// Copyright 2026 chinese-ime-lm contributors. Licensed under Apache-2.0.

#include "ime_lm_filter.h"

#include <rime/candidate.h>
#include <rime/commit_history.h>
#include <rime/composition.h>
#include <rime/config.h>
#include <rime/context.h>
#include <rime/deployer.h>
#include <rime/engine.h>
#include <rime/schema.h>
#include <rime/service.h>
#include <rime/translation.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>

#include "chinese_ime_lm.h"

namespace rime {

namespace {

// The same read-ahead librime's own contextual ranking uses (kContextualSearchLimit in gear/contextual_translation.cc). The reranker compares at most nine candidates that answer the key, so this bounds the work per keystroke, not the decision.
constexpr size_t kSearchLimit = 32;

// How much recent committed text to offer as context. The model keeps only the tail that fits its window, so this only has to be at least that long.
constexpr size_t kContextBytes = 128;

// Candidate types that librime's translators produce straight from a dictionary lookup. Covering the whole segment, such a candidate is an exact hit on the key.
bool IsDictionaryType(const string& type) {
  return type == "phrase" || type == "user_phrase" || type == "table" || type == "user_table";
}

path ResolveModelPath(const string& configured) {
  path file(configured);
  if (file.is_absolute()) {
    return file;
  }
  Deployer& deployer = Service::instance().deployer();
  path in_user_data = deployer.user_data_dir / file;
  if (std::filesystem::exists(in_user_data)) {
    return in_user_data;
  }
  return deployer.shared_data_dir / file;
}

std::shared_ptr<cimelm_model> LoadShared(const path& file) {
  static std::mutex mutex;
  static std::map<string, std::weak_ptr<cimelm_model>> loaded;
  std::lock_guard<std::mutex> lock(mutex);
  const string key = file.u8string();
  if (auto model = loaded[key].lock()) {
    return model;
  }
  std::ifstream stream(file, std::ios::binary);
  if (!stream) {
    LOG(ERROR) << "ime_lm: cannot open model file " << key;
    return nullptr;
  }
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  cimelm_model* raw = nullptr;
  if (cimelm_model_load(bytes.data(), bytes.size(), &raw) != CIMELM_STATUS_OK) {
    LOG(ERROR) << "ime_lm: cannot load model file " << key << ": " << cimelm_last_error();
    return nullptr;
  }
  std::shared_ptr<cimelm_model> model(raw, cimelm_model_free);
  loaded[key] = model;
  LOG(INFO) << "ime_lm: loaded " << key << ", licence " << cimelm_model_license(raw) << ", attribution: " << cimelm_model_attribution(raw);
  return model;
}

cimelm_str View(const string& text) {
  return cimelm_str{text.data(), text.size()};
}

// Puts the candidate the model prefers first and leaves every other candidate where the engine put it.
class RerankedTranslation : public PrefetchTranslation {
 public:
  RerankedTranslation(an<Translation> translation, cimelm_reranker* reranker, const string& context, size_t segment_start, size_t segment_end)
      : PrefetchTranslation(translation) {
    vector<an<Candidate>> fetched;
    while (!translation_->exhausted() && fetched.size() < kSearchLimit) {
      fetched.push_back(translation_->Peek());
      translation_->Next();
    }

    vector<cimelm_str> texts;
    vector<uint8_t> answers_key;
    vector<uint8_t> trusted;
    texts.reserve(fetched.size());
    answers_key.reserve(fetched.size());
    trusted.reserve(fetched.size());
    for (const auto& cand : fetched) {
      // Prefixes and predictive completions end short of, or past, the segment: they answer a different key and are not alternatives to the leader.
      const bool answers = cand->start() == segment_start && cand->end() == segment_end;
      // librime's script translator only makes a sentence when no exact, uncorrected dictionary hit covers the key, so a dictionary-typed leader that covers the key is the case the measurements say never to override. Treating every such hit as trusted also declines when the leader is a correction, which gives up a rescue rather than risking a breakage.
      const auto genuine = Candidate::GetGenuineCandidate(cand);
      const bool dictionary = answers && genuine && IsDictionaryType(genuine->type());
      texts.push_back(View(cand->text()));
      answers_key.push_back(answers ? 1 : 0);
      trusted.push_back(dictionary ? 1 : 0);
    }

    size_t promoted = 0;
    if (cimelm_reranker_best_where(reranker, View(context), texts.data(), texts.size(), answers_key.data(), trusted.data(), &promoted) != CIMELM_STATUS_OK) {
      LOG(ERROR) << "ime_lm: reranking failed: " << cimelm_last_error();
      promoted = 0;
    }
    if (promoted > 0 && promoted < fetched.size()) {
      DLOG(INFO) << "ime_lm: promoted '" << fetched[promoted]->text() << "' over '" << fetched[0]->text() << "'";
      auto chosen = fetched[promoted];
      fetched.erase(fetched.begin() + promoted);
      fetched.insert(fetched.begin(), chosen);
    }
    cache_.insert(cache_.end(), fetched.begin(), fetched.end());
  }
};

}  // namespace

ImeLmFilter::ImeLmFilter(const Ticket& ticket) : Filter(ticket) {
  string configured = "sentence-model.safetensors";
  if (ticket.schema) {
    ticket.schema->config()->GetString(name_space_ + "/model", &configured);
  }
  const path file = ResolveModelPath(configured);
  model_ = LoadShared(file);
  if (!model_) {
    LOG(WARNING) << "ime_lm: no model, candidates are left in the engine's order";
    return;
  }
  if (cimelm_reranker_new(model_.get(), &reranker_) != CIMELM_STATUS_OK) {
    LOG(ERROR) << "ime_lm: cannot create a reranker: " << cimelm_last_error();
    reranker_ = nullptr;
  }
}

ImeLmFilter::~ImeLmFilter() {
  cimelm_reranker_free(reranker_);
}

an<Translation> ImeLmFilter::Apply(an<Translation> translation, CandidateList* candidates) {
  if (!reranker_ || !translation || translation->exhausted()) {
    return translation;
  }
  const auto first = translation->Peek();
  if (!first) {
    return translation;
  }
  // The filter is applied while the engine translates one segment of the composition; the segment is the one this translation's candidates start at.
  const Composition& composition = engine_->context()->composition();
  size_t segment_end = first->end();
  for (const Segment& segment : composition) {
    if (segment.start == first->start()) {
      segment_end = segment.end;
      break;
    }
  }
  return New<RerankedTranslation>(translation, reranker_, PrecedingText(first->start()), first->start(), segment_end);
}

string ImeLmFilter::PrecedingText(size_t segment_start) const {
  const Context* context = engine_->context();
  if (segment_start > 0) {
    return context->composition().GetTextBefore(segment_start);
  }
  // Commits are kept oldest first; join the most recent ones until there is more than the model's window can use.
  string text;
  const CommitHistory& history = context->commit_history();
  for (auto record = history.rbegin(); record != history.rend() && text.size() < kContextBytes; ++record) {
    text.insert(0, record->text);
  }
  return text;
}

}  // namespace rime
