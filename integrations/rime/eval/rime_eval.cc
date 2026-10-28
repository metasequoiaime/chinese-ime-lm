// Copyright 2026 chinese-ime-lm contributors. Licensed under Apache-2.0.
//
// Types each case of an eval set into librime through its public C API and prints, per case, the first candidate and the rank of the gold answer among the first 32. Run by run.sh once per configuration; summarize.py compares the outputs.
//
// usage: rime_eval <shared_data_dir> <user_data_dir> <schema_id> <cases.tsv>

#include <rime_api.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kCandidates = 32;

std::vector<std::string> Split(const std::string& line, char separator) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, separator)) {
    fields.push_back(field);
  }
  return fields;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: rime_eval <shared_data_dir> <user_data_dir> <schema_id> <cases.tsv>\n");
    return 2;
  }
  RimeApi* rime = rime_get_api();
  RIME_STRUCT(RimeTraits, traits);
  traits.shared_data_dir = argv[1];
  traits.user_data_dir = argv[2];
  traits.distribution_name = "chinese-ime-lm eval";
  traits.distribution_code_name = "ime-lm-eval";
  traits.distribution_version = "1";
  traits.app_name = "rime.ime-lm-eval";
  // Warnings and errors only, to stderr: a model that failed to load must be visible in the run's output rather than in a log file.
  traits.min_log_level = 1;
  traits.log_dir = "";
  rime->setup(&traits);
  rime->initialize(&traits);
  if (rime->start_maintenance(True)) {
    rime->join_maintenance_thread();
  }

  RimeSessionId session = rime->create_session();
  if (!session || !rime->select_schema(session, argv[3])) {
    std::fprintf(stderr, "cannot select schema %s\n", argv[3]);
    return 1;
  }

  std::ifstream cases(argv[4]);
  if (!cases) {
    std::fprintf(stderr, "cannot read %s\n", argv[4]);
    return 1;
  }
  std::printf("id\tgold\ttop\tgold_rank\tcandidates\n");
  std::string line;
  bool header = true;
  while (std::getline(cases, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    if (header) {
      header = false;
      continue;
    }
    const auto fields = Split(line, '\t');
    if (fields.size() < 3) {
      continue;
    }
    const std::string& id = fields[0];
    const std::string& input = fields[1];
    const std::string& gold = fields[2];

    // Every case starts from an empty composition and an empty commit history, as the cases carry no preceding text.
    rime->clear_composition(session);
    rime->simulate_key_sequence(session, input.c_str());

    std::vector<std::string> texts;
    RimeCandidateListIterator iterator = {0};
    if (rime->candidate_list_begin(session, &iterator)) {
      while (static_cast<int>(texts.size()) < kCandidates && rime->candidate_list_next(&iterator)) {
        texts.push_back(iterator.candidate.text ? iterator.candidate.text : "");
      }
      rime->candidate_list_end(&iterator);
    }
    int gold_rank = -1;
    for (size_t index = 0; index < texts.size(); ++index) {
      if (texts[index] == gold) {
        gold_rank = static_cast<int>(index);
        break;
      }
    }
    std::string shown;
    for (size_t index = 0; index < texts.size() && index < 10; ++index) {
      shown += (index ? "|" : "") + texts[index];
    }
    std::printf("%s\t%s\t%s\t%d\t%s\n", id.c_str(), gold.c_str(), texts.empty() ? "" : texts[0].c_str(), gold_rank, shown.c_str());
  }

  rime->clear_composition(session);
  rime->destroy_session(session);
  rime->finalize();
  return 0;
}
