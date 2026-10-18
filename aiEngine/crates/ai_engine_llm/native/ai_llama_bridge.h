#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ai_llama_handle ai_llama_handle;

ai_llama_handle * ai_llama_create(
    const char * model_path,
    uint32_t context_length,
    int32_t threads,
    int32_t gpu_layers,
    uint32_t max_sequences,
    char * error_message,
    size_t error_message_size);

void ai_llama_free(ai_llama_handle * handle);

int32_t ai_llama_count_candidate_tokens(
    ai_llama_handle * handle,
    const char * context,
    const char * candidate,
    uint32_t * token_count,
    char * error_message,
    size_t error_message_size);

int32_t ai_llama_score_candidate(
    ai_llama_handle * handle,
    const char * context,
    const char * candidate,
    uint32_t token_limit,
    float * score,
    char * error_message,
    size_t error_message_size);

// Scores the complete context+candidate sequence and the model EOS token.
// Unlike ai_llama_score_candidate, this never truncates the candidate token
// sequence to a shared depth.
int32_t ai_llama_score_candidate_to_eos(
    ai_llama_handle * handle,
    const char * context,
    const char * candidate,
    float * score,
    char * error_message,
    size_t error_message_size);

// Scores independent candidates in one decoder batch.  Each candidate is
// assigned its own sequence id, so candidates cannot reuse one another's KV
// state.  The temporary KV state is cleared after the batch completes.
// When include_eos is non-zero, the EOS probability after each candidate is
// included; otherwise all candidates are truncated to the shortest candidate
// token depth when token_limit is zero.
int32_t ai_llama_score_candidates_batch(
    ai_llama_handle * handle,
    const char * context,
    const char * const * candidates,
    uint32_t candidate_count,
    uint32_t token_limit,
    uint8_t include_eos,
    float * scores,
    char * error_message,
    size_t error_message_size);

// Scores every token of each complete candidate, starting from the model BOS
// when context is empty, but leaves EOS open. Each score is the arithmetic
// mean of the target-token softmax probabilities (not log-probabilities).
int32_t ai_llama_score_candidates_mean_softmax_open_batch(
    ai_llama_handle * handle,
    const char * context,
    const char * const * candidates,
    uint32_t candidate_count,
    float * scores,
    uint32_t * decode_batch_count,
    uint32_t * max_sequences_per_decode_batch,
    char * error_message,
    size_t error_message_size);

// Greedily generates up to max_tokens after prompt and writes UTF-8 pieces
// into output. Generation stops early on an end-of-generation token. When
// japanese_only is non-zero, non-Japanese token pieces are masked from the
// greedy choice (ASCII whitespace and Japanese punctuation remain allowed).
int32_t ai_llama_generate_greedy(
    ai_llama_handle * handle,
    const char * prompt,
    uint32_t max_tokens,
    uint8_t japanese_only,
    char * output,
    size_t output_size,
    uint32_t * generated_tokens,
    char * error_message,
    size_t error_message_size);

#ifdef __cplusplus
}
#endif
