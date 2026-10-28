// Copyright 2026 chinese-ime-lm contributors. Licensed under Apache-2.0.
//
// A Rime filter that reranks the candidates of the current segment with chinese-ime-lm, through its C ABI.

#ifndef RIME_IME_LM_FILTER_H_
#define RIME_IME_LM_FILTER_H_

#include <rime/filter.h>

#include <memory>
#include <string>

struct cimelm_model;
struct cimelm_reranker;

namespace rime {

class ImeLmFilter : public Filter {
 public:
  explicit ImeLmFilter(const Ticket& ticket);
  ~ImeLmFilter() override;

  an<Translation> Apply(an<Translation> translation, CandidateList* candidates) override;

 private:
  // The committed text the model conditions on: the selected text of earlier segments in this composition, or the session's recent commits when the segment starts the composition.
  std::string PrecedingText(size_t segment_start) const;

  // Shared by every session in the process that names the same model file: the weights are read-only and loading them is the expensive part.
  std::shared_ptr<cimelm_model> model_;
  // One per filter, and Rime creates filters per session, which is the scope the reranker's cache belongs to.
  cimelm_reranker* reranker_ = nullptr;
};

}  // namespace rime

#endif  // RIME_IME_LM_FILTER_H_
