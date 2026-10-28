// Copyright 2026 chinese-ime-lm contributors. Licensed under Apache-2.0.

#include <rime/component.h>
#include <rime/registry.h>
#include <rime_api.h>

#include "ime_lm_filter.h"

using namespace rime;

static void rime_ime_lm_initialize() {
  Registry::instance().Register("ime_lm_filter", new Component<ImeLmFilter>);
}

static void rime_ime_lm_finalize() {}

RIME_REGISTER_MODULE(ime_lm)
