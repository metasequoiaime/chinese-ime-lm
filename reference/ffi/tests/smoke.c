/* Uses the library the way a C input method would, through the committed header and the static library, so that a header that no longer matches the code fails here rather than in someone's engine. Run by tests/smoke.sh. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chinese_ime_lm.h"

#define CHECK(call)                                                                        \
    do {                                                                                   \
        cimelm_status status_ = (call);                                                    \
        if (status_ != CIMELM_STATUS_OK) {                                                 \
            fprintf(stderr, "%s failed with %d: %s\n", #call, (int)status_, cimelm_last_error()); \
            return 1;                                                                      \
        }                                                                                  \
    } while (0)

static cimelm_str str(const char *text) {
    cimelm_str value = {text, strlen(text)};
    return value;
}

static unsigned char *read_file(const char *path, size_t *len) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    unsigned char *bytes = NULL;
    if (fseek(file, 0, SEEK_END) == 0) {
        long size = ftell(file);
        if (size > 0 && fseek(file, 0, SEEK_SET) == 0) {
            bytes = malloc((size_t)size);
            if (bytes != NULL && fread(bytes, 1, (size_t)size, file) == (size_t)size) {
                *len = (size_t)size;
            } else {
                free(bytes);
                bytes = NULL;
            }
        }
    }
    fclose(file);
    return bytes;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: smoke <model.safetensors>\n");
        return 2;
    }
    printf("chinese-ime-lm-ffi %s\n", cimelm_version());

    size_t len = 0;
    unsigned char *bytes = read_file(argv[1], &len);
    if (bytes == NULL) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    cimelm_model *model = NULL;
    CHECK(cimelm_model_load(bytes, len, &model));
    /* The model owns everything it needs once loaded. */
    free(bytes);
    printf("license: %s\nattribution: %s\n", cimelm_model_license(model), cimelm_model_attribution(model));

    cimelm_reranker *reranker = NULL;
    CHECK(cimelm_reranker_new(model, &reranker));
    /* The reranker holds its own reference to the weights. */
    cimelm_model_free(model);

    cimelm_str candidates[2] = {str("你好"), str("我很")};
    uint8_t answers_key[2] = {1, 1};
    uint8_t trusted[2] = {0, 0};
    size_t index = 99;
    CHECK(cimelm_reranker_best_where(reranker, str("你好吗"), candidates, 2, answers_key, trusted, &index));
    float scores[2] = {0};
    CHECK(cimelm_reranker_log_probabilities(reranker, str("你好吗"), candidates, 2, scores));
    printf("best: %zu, scores: %f %f\n", index, scores[0], scores[1]);
    /* Both candidates answer the key and neither is a trusted hit, so the decision is the higher score, with the leader keeping a tie. */
    size_t expected = scores[1] > scores[0] ? 1 : 0;
    if (index != expected) {
        fprintf(stderr, "best_where chose %zu but the scores favour %zu\n", index, expected);
        return 1;
    }

    /* A trusted dictionary hit in first place is never overridden. */
    trusted[0] = 1;
    CHECK(cimelm_reranker_best_where(reranker, str("你好吗"), candidates, 2, answers_key, trusted, &index));
    if (index != 0) {
        fprintf(stderr, "a trusted leader was overridden by candidate %zu\n", index);
        return 1;
    }

    unsigned char garbage[16];
    memset(garbage, 0xff, sizeof garbage);
    cimelm_model *rejected = NULL;
    cimelm_status status = cimelm_model_load(garbage, sizeof garbage, &rejected);
    if (status != CIMELM_STATUS_INVALID_MODEL || rejected != NULL) {
        fprintf(stderr, "malformed bytes were not rejected: status %d\n", (int)status);
        return 1;
    }
    printf("rejected malformed model: %s\n", cimelm_last_error());

    cimelm_reranker_free(reranker);
    printf("ok\n");
    return 0;
}
