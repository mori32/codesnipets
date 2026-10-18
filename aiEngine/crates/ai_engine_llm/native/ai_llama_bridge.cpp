#include "ai_llama_bridge.h"

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

struct ai_llama_handle {
    llama_model * model;
    llama_context * context;
    const llama_vocab * vocab;
    std::vector<uint8_t> japanese_token_allowed;
};

struct prepared_candidate {
    std::vector<llama_token> tokens;
    size_t first_candidate_token = 0;
    size_t candidate_token_count = 0;
};

namespace {

void set_error(char * buffer, size_t size, const char * message) {
    if (buffer == nullptr || size == 0) {
        return;
    }
    const size_t length = std::min(size - 1, std::strlen(message));
    std::memcpy(buffer, message, length);
    buffer[length] = '\0';
}

void set_error(char * buffer, size_t size, const std::string & message) {
    set_error(buffer, size, message.c_str());
}

bool valid_text(const char * text, char * error, size_t error_size) {
    if (text == nullptr) {
        set_error(error, error_size, "text pointer is null");
        return false;
    }
    if (std::strlen(text) > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        set_error(error, error_size, "text is too long for llama.cpp tokenization");
        return false;
    }
    return true;
}

bool tokenize(
    const llama_vocab * vocab,
    const char * text,
    std::vector<llama_token> & tokens,
    char * error,
    size_t error_size) {
    if (!valid_text(text, error, error_size)) {
        return false;
    }
    const auto text_length = static_cast<int32_t>(std::strlen(text));
    int32_t required = llama_tokenize(vocab, text, text_length, nullptr, 0, false, false);
    if (required < 0) {
        if (required == std::numeric_limits<int32_t>::min()) {
            set_error(error, error_size, "token count overflow");
            return false;
        }
        required = -required;
    }
    if (required == 0) {
        tokens.clear();
        return true;
    }
    tokens.resize(static_cast<size_t>(required));
    int32_t actual = llama_tokenize(
        vocab,
        text,
        text_length,
        tokens.data(),
        required,
        false,
        false);
    if (actual < 0) {
        set_error(error, error_size, "llama.cpp tokenization failed");
        return false;
    }
    tokens.resize(static_cast<size_t>(actual));
    return true;
}

bool is_japanese_code_point(uint32_t code_point) {
    // SentencePiece uses U+2581 (LOWER ONE EIGHTH BLOCK) as its visible
    // whitespace marker.  It is metadata in a token piece, not non-Japanese
    // output, so keep it when the Japanese-only generation filter is active.
    if (code_point == 0x2581) {
        return true;
    }
    return (code_point >= 0x3000 && code_point <= 0x303f) ||
        (code_point >= 0x3040 && code_point <= 0x30ff) ||
        (code_point >= 0x31f0 && code_point <= 0x31ff) ||
        (code_point >= 0x3400 && code_point <= 0x4dbf) ||
        (code_point >= 0x4e00 && code_point <= 0x9fff) ||
        (code_point >= 0xf900 && code_point <= 0xfaff) ||
        (code_point >= 0x20000 && code_point <= 0x2fa1f) ||
        (code_point >= 0xff01 && code_point <= 0xff0f) ||
        (code_point >= 0xff1a && code_point <= 0xff20) ||
        (code_point >= 0xff3b && code_point <= 0xff40) ||
        (code_point >= 0xff5b && code_point <= 0xff65);
}

bool is_japanese_token_piece(const llama_vocab * vocab, llama_token token) {
    char piece_buffer[256];
    int32_t piece_size = llama_token_to_piece(
        vocab,
        token,
        piece_buffer,
        static_cast<int32_t>(sizeof(piece_buffer)),
        0,
        false);
    std::vector<char> piece_storage;
    if (piece_size < 0) {
        piece_storage.resize(static_cast<size_t>(-piece_size));
        piece_size = llama_token_to_piece(
            vocab,
            token,
            piece_storage.data(),
            static_cast<int32_t>(piece_storage.size()),
            0,
            false);
    }
    if (piece_size <= 0) {
        return false;
    }

    const auto *bytes = reinterpret_cast<const unsigned char *>(
        piece_storage.empty() ? piece_buffer : piece_storage.data());
    const size_t size = static_cast<size_t>(piece_size);
    size_t index = 0;
    while (index < size) {
        const unsigned char first = bytes[index];
        uint32_t code_point = 0;
        size_t width = 0;
        if (first < 0x80) {
            if (first != ' ' && first != '\t' && first != '\n' && first != '\r') {
                return false;
            }
            ++index;
            continue;
        }
        if ((first & 0xe0) == 0xc0) {
            code_point = first & 0x1f;
            width = 2;
        } else if ((first & 0xf0) == 0xe0) {
            code_point = first & 0x0f;
            width = 3;
        } else if ((first & 0xf8) == 0xf0) {
            code_point = first & 0x07;
            width = 4;
        } else {
            return false;
        }
        if (index + width > size) {
            return false;
        }
        for (size_t offset = 1; offset < width; ++offset) {
            const unsigned char continuation = bytes[index + offset];
            if ((continuation & 0xc0) != 0x80) {
                return false;
            }
            code_point = (code_point << 6) | (continuation & 0x3f);
        }
        const bool overlong = (width == 2 && code_point < 0x80) ||
            (width == 3 && code_point < 0x800) ||
            (width == 4 && code_point < 0x10000);
        if (overlong || code_point > 0x10ffff ||
            (code_point >= 0xd800 && code_point <= 0xdfff) ||
            !is_japanese_code_point(code_point)) {
            return false;
        }
        index += width;
    }
    return true;
}

bool score_tokens(
    ai_llama_handle * handle,
    const std::vector<llama_token> & tokens,
    size_t first_candidate_token,
    size_t token_limit,
    bool include_eos,
    float * score,
    char * error,
    size_t error_size) {
    if (tokens.empty()) {
        set_error(error, error_size, "candidate token sequence is empty");
        return false;
    }
    if (first_candidate_token >= tokens.size()) {
        set_error(error, error_size, "candidate token offset is invalid");
        return false;
    }
    const size_t available_tokens = tokens.size() - first_candidate_token;
    const size_t scored_token_limit = std::min(token_limit, available_tokens);
    if (scored_token_limit == 0) {
        set_error(error, error_size, "candidate token limit is zero");
        return false;
    }
    const size_t score_end = first_candidate_token + scored_token_limit;
    const llama_token eos = include_eos ? llama_vocab_eos(handle->vocab) : LLAMA_TOKEN_NULL;
    if (include_eos && eos == LLAMA_TOKEN_NULL) {
        set_error(error, error_size, "model has no EOS token");
        return false;
    }

    llama_memory_clear(llama_get_memory(handle->context), true);
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    if (batch.token == nullptr || batch.logits == nullptr) {
        llama_batch_free(batch);
        set_error(error, error_size, "llama.cpp could not allocate a batch");
        return false;
    }

    for (size_t index = 0; index < tokens.size(); ++index) {
        batch.token[index] = tokens[index];
        batch.pos[index] = static_cast<llama_pos>(index);
        batch.n_seq_id[index] = 1;
        batch.seq_id[index][0] = 0;
        batch.logits[index] = 1;
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());

    const int32_t result = llama_decode(handle->context, batch);
    if (result != 0) {
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(handle->context), true);
        set_error(error, error_size, "llama.cpp decode failed");
        return false;
    }

    const int32_t vocabulary_size = llama_vocab_n_tokens(handle->vocab);
    if (vocabulary_size <= 0) {
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(handle->context), true);
        set_error(error, error_size, "llama.cpp returned an invalid vocabulary size");
        return false;
    }

    double total_log_probability = 0.0;
    size_t scored_tokens = 0;
    for (size_t token_index = std::max<size_t>(first_candidate_token, 1);
         token_index < score_end;
         ++token_index) {
        const float * logits = llama_get_logits_ith(
            handle->context,
            static_cast<int32_t>(token_index - 1));
        if (logits == nullptr) {
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error, error_size, "llama.cpp did not return logits");
            return false;
        }
        float max_logit = logits[0];
        for (int32_t vocabulary_index = 1; vocabulary_index < vocabulary_size; ++vocabulary_index) {
            max_logit = std::max(max_logit, logits[vocabulary_index]);
        }
        double denominator = 0.0;
        for (int32_t vocabulary_index = 0; vocabulary_index < vocabulary_size; ++vocabulary_index) {
            denominator += std::exp(static_cast<double>(logits[vocabulary_index] - max_logit));
        }
        const auto target = static_cast<int32_t>(tokens[token_index]);
        if (target < 0 || target >= vocabulary_size || denominator <= 0.0) {
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error, error_size, "llama.cpp returned an invalid target token");
            return false;
        }
        const double log_probability =
            static_cast<double>(logits[target] - max_logit) - std::log(denominator);
        total_log_probability += log_probability;
        ++scored_tokens;
    }

    if (include_eos) {
        // EOS is the token predicted immediately after the final candidate
        // token. Its logits are therefore at the final candidate position.
        const float * logits = llama_get_logits_ith(
            handle->context,
            static_cast<int32_t>(score_end - 1));
        if (logits == nullptr) {
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error, error_size, "llama.cpp did not return EOS logits");
            return false;
        }
        float max_logit = logits[0];
        for (int32_t vocabulary_index = 1; vocabulary_index < vocabulary_size; ++vocabulary_index) {
            max_logit = std::max(max_logit, logits[vocabulary_index]);
        }
        double denominator = 0.0;
        for (int32_t vocabulary_index = 0; vocabulary_index < vocabulary_size; ++vocabulary_index) {
            denominator += std::exp(static_cast<double>(logits[vocabulary_index] - max_logit));
        }
        const auto eos_index = static_cast<int32_t>(eos);
        if (eos_index < 0 || eos_index >= vocabulary_size || denominator <= 0.0) {
            llama_batch_free(batch);
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error, error_size, "llama.cpp returned an invalid EOS token");
            return false;
        }
        total_log_probability +=
            static_cast<double>(logits[eos_index] - max_logit) - std::log(denominator);
        ++scored_tokens;
    }
    llama_batch_free(batch);
    llama_memory_clear(llama_get_memory(handle->context), true);

    if (scored_tokens == 0) {
        set_error(error, error_size, "candidate has no scoreable tokens");
        return false;
    }
    // All candidates use the same shortest token depth. Keep the sum rather
    // than dividing by that depth so the score matches the sequence log-prob
    // contract and remains calibrated for the classical-score weight.
    *score = static_cast<float>(total_log_probability);
    return std::isfinite(*score);
}

bool prepare_candidate_tokens(
    ai_llama_handle * handle,
    const char * context,
    const char * candidate,
    std::vector<llama_token> & combined_tokens,
    size_t & first_candidate_token,
    size_t & candidate_token_count,
    bool prepend_bos,
    char * error,
    size_t error_size) {
    std::vector<llama_token> context_tokens;
    std::vector<llama_token> candidate_tokens;
    if (!tokenize(handle->vocab, context, context_tokens, error, error_size) ||
        !tokenize(handle->vocab, candidate, candidate_tokens, error, error_size)) {
        return false;
    }

    std::string combined = context;
    combined += candidate;
    if (!tokenize(handle->vocab, combined.c_str(), combined_tokens, error, error_size)) {
        return false;
    }
    if (candidate_tokens.empty() || combined_tokens.size() < candidate_tokens.size()) {
        set_error(error, error_size, "candidate token sequence is invalid");
        return false;
    }
    if (combined_tokens.size() < context_tokens.size() ||
        !std::equal(context_tokens.begin(), context_tokens.end(), combined_tokens.begin())) {
        set_error(
            error,
            error_size,
            "context tokenization is not a prefix of the combined tokenization");
        return false;
    }
    const size_t suffix_start = combined_tokens.size() - candidate_tokens.size();
    if (suffix_start >= context_tokens.size() &&
        std::equal(
            candidate_tokens.begin(),
            candidate_tokens.end(),
            combined_tokens.begin() + suffix_start)) {
        // BPE-style tokenizers generally preserve the candidate as a suffix.
        candidate_token_count = candidate_tokens.size();
    } else {
        // SentencePiece may retokenize the boundary: a token can contain the
        // final character(s) of the context and the first character(s) of the
        // candidate.  In that case the separately tokenized candidate cannot
        // be used to locate the scored range.  When the context itself is a
        // prefix of the combined token sequence, score every token after the
        // context boundary instead.  This preserves the continuation
        // direction without rejecting valid SPM GGUF models.
        if (context_tokens.size() > combined_tokens.size() ||
            !std::equal(
                context_tokens.begin(),
                context_tokens.end(),
                combined_tokens.begin())) {
            set_error(
                error,
                error_size,
                "tokenization does not preserve a context boundary");
            return false;
        }
        first_candidate_token = context_tokens.size();
        candidate_token_count = combined_tokens.size() - first_candidate_token;
        if (candidate_token_count == 0) {
            set_error(error, error_size, "candidate has no tokens after context boundary");
            return false;
        }
    }

    if (context_tokens.empty() && prepend_bos) {
        const llama_token bos = llama_vocab_bos(handle->vocab);
        if (bos == LLAMA_TOKEN_NULL) {
            set_error(error, error_size, "model has no BOS token for empty context");
            return false;
        }
        combined_tokens.insert(combined_tokens.begin(), bos);
    }
    first_candidate_token = combined_tokens.size() - candidate_token_count;
    return true;
}

bool score_candidates_batch(
    ai_llama_handle * handle,
    const std::vector<prepared_candidate> & candidates,
    size_t token_limit,
    bool include_eos,
    bool full_open_candidate,
    bool mean_softmax_probability,
    float * scores,
    char * error,
    size_t error_size) {
    if (candidates.empty()) {
        return true;
    }
    if (candidates.size() > static_cast<size_t>(llama_n_seq_max(handle->context))) {
        set_error(error, error_size, "candidate batch exceeds llama.cpp sequence capacity");
        return false;
    }

    size_t total_tokens = 0;
    size_t max_sequence_length = 0;
    size_t shortest_candidate_tokens = std::numeric_limits<size_t>::max();
    for (const auto & candidate : candidates) {
        if (candidate.tokens.empty() || candidate.candidate_token_count == 0 ||
            candidate.first_candidate_token >= candidate.tokens.size()) {
            set_error(error, error_size, "candidate token sequence is invalid");
            return false;
        }
        max_sequence_length = std::max(max_sequence_length, candidate.tokens.size());
        shortest_candidate_tokens = std::min(
            shortest_candidate_tokens, candidate.candidate_token_count);
    }

    if (max_sequence_length > std::numeric_limits<size_t>::max() / candidates.size()) {
        set_error(error, error_size, "batch token count overflow");
        return false;
    }
    // Right-pad all independent sequences to the same length.  This keeps the
    // decoder in one no-reuse batch (the padding logits are disabled), which
    // is the llama.cpp equivalent of the padded ONNX batch used by the
    // reference evaluator.
    total_tokens = max_sequence_length * candidates.size();

    // Open-ended prefix scoring deliberately uses the shortest common depth.
    // Closed EOS scoring, however, must score each candidate through its own
    // final token and then EOS.  Using the common shortest depth for both
    // modes silently truncated multi-token candidates and made batch results
    // differ from the single-candidate scorer.
    const size_t common_score_limit = token_limit == 0
        ? shortest_candidate_tokens
        : std::min(token_limit, shortest_candidate_tokens);
    if (common_score_limit == 0 && !full_open_candidate) {
        set_error(error, error_size, "batch candidate token limit is zero");
        return false;
    }
    if (total_tokens > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        total_tokens > static_cast<size_t>(llama_n_batch(handle->context))) {
        set_error(error, error_size, "candidate batch exceeds llama.cpp batch capacity");
        return false;
    }

    llama_memory_clear(llama_get_memory(handle->context), true);
    llama_batch batch = llama_batch_init(
        static_cast<int32_t>(total_tokens),
        0,
        static_cast<int32_t>(candidates.size()));
    if (batch.token == nullptr || batch.logits == nullptr || batch.seq_id == nullptr) {
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(handle->context), true);
        set_error(error, error_size, "llama.cpp could not allocate candidate batch");
        return false;
    }

    std::vector<std::vector<int32_t>> batch_indices;
    batch_indices.reserve(candidates.size());
    size_t batch_index = 0;
    for (const auto & candidate : candidates) {
        batch_indices.emplace_back(candidate.tokens.size());
    }
    // Add positions in lockstep.  This is the layout used by llama.cpp's
    // parallel-sequence examples and makes each causal position available for
    // every sequence before advancing to the next position.
    for (size_t position = 0; position < max_sequence_length; ++position) {
        for (size_t sequence = 0; sequence < candidates.size(); ++sequence) {
            const auto & candidate = candidates[sequence];
            const bool is_padding = position >= candidate.tokens.size();
            const llama_token padding_token = llama_vocab_pad(handle->vocab) != LLAMA_TOKEN_NULL
                ? llama_vocab_pad(handle->vocab)
                : (llama_vocab_eos(handle->vocab) != LLAMA_TOKEN_NULL
                    ? llama_vocab_eos(handle->vocab)
                    : 0);
            batch.token[batch_index] = is_padding ? padding_token : candidate.tokens[position];
            batch.pos[batch_index] = static_cast<llama_pos>(position);
            batch.n_seq_id[batch_index] = 1;
            batch.seq_id[batch_index][0] = static_cast<llama_seq_id>(sequence);
            const size_t candidate_score_limit = full_open_candidate
                ? candidate.candidate_token_count
                : include_eos
                ? (token_limit == 0
                    ? candidate.candidate_token_count
                    : std::min(token_limit, candidate.candidate_token_count))
                : common_score_limit;
            if (candidate_score_limit == 0) {
                llama_batch_free(batch);
                llama_memory_clear(llama_get_memory(handle->context), true);
                set_error(error, error_size, "batch candidate token limit is zero");
                return false;
            }
            const size_t score_logit_start = candidate.first_candidate_token > 0
                ? candidate.first_candidate_token - 1
                : 0;
            const size_t score_logit_end = candidate.first_candidate_token + candidate_score_limit;
            batch.logits[batch_index] = !is_padding &&
                position >= score_logit_start && position < score_logit_end;
            if (!is_padding) {
                batch_indices[sequence][position] = static_cast<int32_t>(batch_index);
            }
            ++batch_index;
        }
    }
    batch.n_tokens = static_cast<int32_t>(batch_index);

    const auto cleanup = [&]() {
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(handle->context), true);
    };

    if (llama_decode(handle->context, batch) != 0) {
        cleanup();
        set_error(error, error_size, "llama.cpp candidate batch decode failed");
        return false;
    }

    const int32_t vocabulary_size = llama_vocab_n_tokens(handle->vocab);
    if (vocabulary_size <= 0) {
        cleanup();
        set_error(error, error_size, "llama.cpp returned an invalid vocabulary size");
        return false;
    }
    const llama_token eos = include_eos ? llama_vocab_eos(handle->vocab) : LLAMA_TOKEN_NULL;
    if (include_eos && eos == LLAMA_TOKEN_NULL) {
        cleanup();
        set_error(error, error_size, "model has no EOS token");
        return false;
    }

    const auto log_probability = [&](const float * logits, llama_token target, double & result) {
        if (logits == nullptr || target < 0 || target >= vocabulary_size) {
            return false;
        }
        float max_logit = logits[0];
        for (int32_t vocabulary_index = 1; vocabulary_index < vocabulary_size; ++vocabulary_index) {
            max_logit = std::max(max_logit, logits[vocabulary_index]);
        }
        double denominator = 0.0;
        for (int32_t vocabulary_index = 0; vocabulary_index < vocabulary_size; ++vocabulary_index) {
            denominator += std::exp(static_cast<double>(logits[vocabulary_index] - max_logit));
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator)) {
            return false;
        }
        result += static_cast<double>(logits[target] - max_logit) - std::log(denominator);
        return true;
    };

    for (size_t sequence = 0; sequence < candidates.size(); ++sequence) {
        const auto & candidate = candidates[sequence];
        const size_t score_limit = full_open_candidate
            ? candidate.candidate_token_count
            : include_eos
            ? (token_limit == 0
                ? candidate.candidate_token_count
                : std::min(token_limit, candidate.candidate_token_count))
            : common_score_limit;
        const size_t score_end = candidate.first_candidate_token + score_limit;
        double total_log_probability = 0.0;
        double total_softmax_probability = 0.0;
        for (size_t token_index = std::max<size_t>(candidate.first_candidate_token, 1);
             token_index < score_end;
             ++token_index) {
            const float * logits = llama_get_logits_ith(
                handle->context, batch_indices[sequence][token_index - 1]);
            double token_log_probability = 0.0;
            if (!log_probability(logits, candidate.tokens[token_index], token_log_probability)) {
                cleanup();
                set_error(error, error_size, "llama.cpp returned invalid candidate logits");
                return false;
            }
            if (mean_softmax_probability) {
                total_softmax_probability += std::exp(token_log_probability);
            } else {
                total_log_probability += token_log_probability;
            }
        }
        if (include_eos) {
            const float * logits = llama_get_logits_ith(
                handle->context, batch_indices[sequence][score_end - 1]);
            if (!log_probability(logits, eos, total_log_probability)) {
                cleanup();
                set_error(error, error_size, "llama.cpp returned invalid EOS logits");
                return false;
            }
        }
        if (!std::isfinite(total_log_probability)) {
            cleanup();
            set_error(error, error_size, "candidate log probability is not finite");
            return false;
        }
        scores[sequence] = mean_softmax_probability
            ? static_cast<float>(total_softmax_probability / static_cast<double>(score_limit))
            : static_cast<float>(total_log_probability);
    }

    cleanup();
    return true;
}

} // namespace

extern "C" ai_llama_handle * ai_llama_create(
    const char * model_path,
    uint32_t context_length,
    int32_t threads,
    int32_t gpu_layers,
    uint32_t max_sequences,
    char * error_message,
    size_t error_message_size) {
    if (model_path == nullptr || std::strlen(model_path) == 0) {
        set_error(error_message, error_message_size, "model path is empty");
        return nullptr;
    }

    llama_backend_init();
    if (gpu_layers != 0 && !llama_supports_gpu_offload()) {
        set_error(
            error_message,
            error_message_size,
            "GPU layers were requested, but this llama.cpp build has no GPU backend; rebuild with CUDA or Vulkan support");
        return nullptr;
    }
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = gpu_layers;
    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (model == nullptr) {
        set_error(error_message, error_message_size, "llama.cpp could not load the model");
        return nullptr;
    }

    llama_context_params context_params = llama_context_default_params();
    // The batch scorer evaluates independent candidate sequences together.
    // Keep the configured context length available to each sequence; with a
    // non-unified KV buffer llama.cpp divides n_ctx by n_seq_max, which made
    // longer stateless generation prompts fail despite fitting in n_ctx.
    if (max_sequences == 0) {
        set_error(error_message, error_message_size, "maximum sequence count must be positive");
        llama_model_free(model);
        return nullptr;
    }
    context_params.n_seq_max = max_sequences;
    context_params.kv_unified = true;
    if (context_length > 0) {
        context_params.n_ctx = context_length;
        context_params.n_batch = std::max(context_params.n_batch, context_length);
    }
    if (threads > 0) {
        context_params.n_threads = threads;
        context_params.n_threads_batch = threads;
    }
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        llama_model_free(model);
        set_error(error_message, error_message_size, "llama.cpp could not create a context");
        return nullptr;
    }

    auto * handle = new ai_llama_handle { model, context, llama_model_get_vocab(model) };
    const int32_t vocabulary_size = llama_vocab_n_tokens(handle->vocab);
    if (vocabulary_size <= 0) {
        delete handle;
        llama_free(context);
        llama_model_free(model);
        set_error(error_message, error_message_size, "llama.cpp returned an invalid vocabulary size");
        return nullptr;
    }
    handle->japanese_token_allowed.resize(static_cast<size_t>(vocabulary_size));
    for (int32_t vocabulary_index = 0; vocabulary_index < vocabulary_size; ++vocabulary_index) {
        handle->japanese_token_allowed[static_cast<size_t>(vocabulary_index)] =
            is_japanese_token_piece(handle->vocab, static_cast<llama_token>(vocabulary_index));
    }
    return handle;
}

extern "C" void ai_llama_free(ai_llama_handle * handle) {
    if (handle == nullptr) {
        return;
    }
    llama_free(handle->context);
    llama_model_free(handle->model);
    delete handle;
}

extern "C" int32_t ai_llama_count_candidate_tokens(
    ai_llama_handle * handle,
    const char * context,
    const char * candidate,
    uint32_t * token_count,
    char * error_message,
    size_t error_message_size) {
    if (handle == nullptr || token_count == nullptr) {
        set_error(error_message, error_message_size, "invalid llama.cpp token count arguments");
        return -1;
    }
    std::vector<llama_token> combined_tokens;
    size_t first_candidate_token = 0;
    size_t candidate_token_count = 0;
    if (!prepare_candidate_tokens(
            handle,
            context,
            candidate,
            combined_tokens,
            first_candidate_token,
            candidate_token_count,
            true,
            error_message,
            error_message_size)) {
        return -2;
    }
    static_cast<void>(combined_tokens);
    static_cast<void>(first_candidate_token);
    if (candidate_token_count > std::numeric_limits<uint32_t>::max()) {
        set_error(error_message, error_message_size, "candidate token count is too large");
        return -3;
    }
    *token_count = static_cast<uint32_t>(candidate_token_count);
    return 0;
}

extern "C" int32_t ai_llama_score_candidate(
    ai_llama_handle * handle,
    const char * context,
    const char * candidate,
    uint32_t token_limit,
    float * score,
    char * error_message,
    size_t error_message_size) {
    if (handle == nullptr || score == nullptr) {
        set_error(error_message, error_message_size, "invalid llama.cpp score arguments");
        return -1;
    }
    std::vector<llama_token> combined_tokens;
    size_t candidate_start = 0;
    size_t candidate_token_count = 0;
    if (!prepare_candidate_tokens(
            handle,
            context,
            candidate,
            combined_tokens,
            candidate_start,
            candidate_token_count,
            true,
            error_message,
            error_message_size)) {
        return -2;
    }
    const size_t score_limit = token_limit == 0
        ? candidate_token_count
        : std::min<size_t>(token_limit, candidate_token_count);
    return score_tokens(
        handle,
        combined_tokens,
        candidate_start,
        score_limit,
        false,
        score,
        error_message,
        error_message_size)
        ? 0
        : -5;
}

extern "C" int32_t ai_llama_score_candidate_to_eos(
    ai_llama_handle * handle,
    const char * context,
    const char * candidate,
    float * score,
    char * error_message,
    size_t error_message_size) {
    if (handle == nullptr || score == nullptr) {
        set_error(error_message, error_message_size, "invalid llama.cpp EOS score arguments");
        return -1;
    }
    std::vector<llama_token> combined_tokens;
    size_t candidate_start = 0;
    size_t candidate_token_count = 0;
    if (!prepare_candidate_tokens(
            handle,
            context,
            candidate,
            combined_tokens,
            candidate_start,
            candidate_token_count,
            false,
            error_message,
            error_message_size)) {
        return -2;
    }
    return score_tokens(
        handle,
        combined_tokens,
        candidate_start,
        candidate_token_count,
        true,
        score,
        error_message,
        error_message_size)
        ? 0
        : -5;
}

extern "C" int32_t ai_llama_score_candidates_batch(
    ai_llama_handle * handle,
    const char * context,
    const char * const * candidates,
    uint32_t candidate_count,
    uint32_t token_limit,
    uint8_t include_eos,
    float * scores,
    char * error_message,
    size_t error_message_size) {
    if (handle == nullptr || context == nullptr || candidates == nullptr || scores == nullptr ||
        candidate_count == 0) {
        set_error(error_message, error_message_size, "invalid llama.cpp candidate batch arguments");
        return -1;
    }

    std::vector<prepared_candidate> prepared;
    prepared.reserve(candidate_count);
    for (uint32_t index = 0; index < candidate_count; ++index) {
        if (candidates[index] == nullptr) {
            set_error(error_message, error_message_size, "candidate pointer is null");
            return -2;
        }
        prepared_candidate candidate;
        if (!prepare_candidate_tokens(
                handle,
                context,
                candidates[index],
                candidate.tokens,
                candidate.first_candidate_token,
                candidate.candidate_token_count,
                !include_eos,
                error_message,
                error_message_size)) {
            return -3;
        }
        prepared.emplace_back(std::move(candidate));
    }

    return score_candidates_batch(
        handle,
        prepared,
        token_limit,
        include_eos != 0,
        false,
        false,
        scores,
        error_message,
        error_message_size)
        ? 0
        : -4;
}

extern "C" int32_t ai_llama_score_candidates_mean_softmax_open_batch(
    ai_llama_handle * handle,
    const char * context,
    const char * const * candidates,
    uint32_t candidate_count,
    float * scores,
    uint32_t * decode_batch_count,
    uint32_t * max_sequences_per_decode_batch,
    char * error_message,
    size_t error_message_size) {
    if (handle == nullptr || context == nullptr || candidates == nullptr || scores == nullptr ||
        decode_batch_count == nullptr || max_sequences_per_decode_batch == nullptr ||
        candidate_count == 0) {
        set_error(error_message, error_message_size, "invalid mean-softmax candidate batch arguments");
        return -1;
    }

    std::vector<prepared_candidate> prepared;
    prepared.reserve(candidate_count);
    for (uint32_t index = 0; index < candidate_count; ++index) {
        if (candidates[index] == nullptr) {
            set_error(error_message, error_message_size, "candidate pointer is null");
            return -2;
        }
        prepared_candidate candidate;
        if (!prepare_candidate_tokens(
                handle,
                context,
                candidates[index],
                candidate.tokens,
                candidate.first_candidate_token,
                candidate.candidate_token_count,
                true,
                error_message,
                error_message_size)) {
            return -3;
        }
        prepared.emplace_back(std::move(candidate));
    }

    // Fit as many independent sequences as the model's decoder batch permits.
    // N-best paths differ in token length, so split at candidate boundaries
    // instead of truncating every path to the shortest one.
    const size_t batch_capacity = static_cast<size_t>(llama_n_batch(handle->context));
    const size_t sequence_capacity = static_cast<size_t>(llama_n_seq_max(handle->context));
    *decode_batch_count = 0;
    *max_sequences_per_decode_batch = 0;
    if (batch_capacity == 0 || sequence_capacity == 0) {
        set_error(error_message, error_message_size, "llama.cpp returned invalid batch capacities");
        return -4;
    }
    size_t begin = 0;
    while (begin < prepared.size()) {
        size_t end = begin;
        size_t max_length = 0;
        while (end < prepared.size() && end - begin < sequence_capacity) {
            const size_t next_length = prepared[end].tokens.size();
            const size_t next_max_length = std::max(max_length, next_length);
            const size_t next_count = end - begin + 1;
            if (next_max_length > batch_capacity / next_count) {
                break;
            }
            max_length = next_max_length;
            ++end;
        }
        if (end == begin) {
            set_error(error_message, error_message_size, "one candidate exceeds llama.cpp batch capacity");
            return -5;
        }
        ++*decode_batch_count;
        *max_sequences_per_decode_batch = std::max(
            *max_sequences_per_decode_batch,
            static_cast<uint32_t>(end - begin));
        if (!score_candidates_batch(
                handle,
                std::vector<prepared_candidate>(prepared.begin() + static_cast<std::ptrdiff_t>(begin),
                    prepared.begin() + static_cast<std::ptrdiff_t>(end)),
                0,
                false,
                true,
                true,
                scores + begin,
                error_message,
                error_message_size)) {
            return -6;
        }
        begin = end;
    }
    return 0;
}

extern "C" int32_t ai_llama_generate_greedy(
    ai_llama_handle * handle,
    const char * prompt,
    uint32_t max_tokens,
    uint8_t japanese_only,
    char * output,
    size_t output_size,
    uint32_t * generated_tokens,
    char * error_message,
    size_t error_message_size) {
    if (handle == nullptr || generated_tokens == nullptr || output == nullptr || output_size == 0) {
        set_error(error_message, error_message_size, "invalid llama.cpp generation arguments");
        return -1;
    }
    std::vector<llama_token> tokens;
    if (!tokenize(handle->vocab, prompt, tokens, error_message, error_message_size)) {
        return -2;
    }
    if (tokens.empty()) {
        set_error(error_message, error_message_size, "generation prompt token sequence is empty");
        return -3;
    }

    // GPT-NeoX SentencePiece checkpoints expect the BOS token before the
    // prompt.  BPE models (for example Qwen) generally do not, so limit this
    // compatibility insertion to SPM vocabularies.
    if (llama_vocab_type(handle->vocab) == LLAMA_VOCAB_TYPE_SPM) {
        const llama_token bos = llama_vocab_bos(handle->vocab);
        if (bos != LLAMA_TOKEN_NULL && tokens.front() != bos) {
            tokens.insert(tokens.begin(), bos);
        }
    }

    llama_memory_clear(llama_get_memory(handle->context), true);
    const uint32_t physical_batch_size = llama_n_ubatch(handle->context);
    if (physical_batch_size == 0) {
        set_error(error_message, error_message_size, "llama.cpp returned an invalid physical batch size");
        return -4;
    }
    const size_t prompt_chunk_size = std::min(tokens.size(), static_cast<size_t>(physical_batch_size));
    llama_batch prompt_batch = llama_batch_init(static_cast<int32_t>(prompt_chunk_size), 0, 1);
    if (prompt_batch.token == nullptr || prompt_batch.logits == nullptr) {
        llama_batch_free(prompt_batch);
        set_error(error_message, error_message_size, "llama.cpp could not allocate prompt batch");
        return -4;
    }
    for (size_t offset = 0; offset < tokens.size(); offset += prompt_chunk_size) {
        const size_t chunk_size = std::min(prompt_chunk_size, tokens.size() - offset);
        for (size_t index = 0; index < chunk_size; ++index) {
            const size_t token_index = offset + index;
            prompt_batch.token[index] = tokens[token_index];
            prompt_batch.pos[index] = static_cast<llama_pos>(token_index);
            prompt_batch.n_seq_id[index] = 1;
            prompt_batch.seq_id[index][0] = 0;
            prompt_batch.logits[index] = token_index + 1 == tokens.size();
        }
        prompt_batch.n_tokens = static_cast<int32_t>(chunk_size);
        if (llama_decode(handle->context, prompt_batch) != 0) {
            llama_batch_free(prompt_batch);
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error_message, error_message_size, "llama.cpp prompt decode failed");
            return -5;
        }
    }
    llama_batch_free(prompt_batch);

    const int32_t vocabulary_size = llama_vocab_n_tokens(handle->vocab);
    if (vocabulary_size <= 0) {
        llama_memory_clear(llama_get_memory(handle->context), true);
        set_error(error_message, error_message_size, "llama.cpp returned an invalid vocabulary size");
        return -6;
    }

    std::string generated;
    uint32_t count = 0;
    const size_t last_prompt_chunk_size = tokens.size() % prompt_chunk_size == 0
        ? prompt_chunk_size
        : tokens.size() % prompt_chunk_size;
    int32_t logits_index = static_cast<int32_t>(last_prompt_chunk_size - 1);
    while (count < max_tokens) {
        const float * logits = llama_get_logits_ith(handle->context, logits_index);
        if (logits == nullptr) {
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error_message, error_message_size, "llama.cpp did not return generation logits");
            return -7;
        }
        llama_token next = LLAMA_TOKEN_NULL;
        float best_logit = -std::numeric_limits<float>::infinity();
        const llama_token pad = llama_vocab_pad(handle->vocab);
        const llama_token eos = llama_vocab_eos(handle->vocab);
        for (int32_t vocabulary_index = 0; vocabulary_index < vocabulary_size; ++vocabulary_index) {
            const llama_token candidate = static_cast<llama_token>(vocabulary_index);
            if (candidate == pad && candidate != eos) {
                continue;
            }
            if (japanese_only != 0 && !llama_vocab_is_eog(handle->vocab, candidate) &&
                handle->japanese_token_allowed[static_cast<size_t>(vocabulary_index)] == 0) {
                continue;
            }
            if (logits[vocabulary_index] > best_logit) {
                best_logit = logits[vocabulary_index];
                next = candidate;
            }
        }
        if (next == LLAMA_TOKEN_NULL) {
            if (japanese_only != 0) {
                break;
            }
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error_message, error_message_size, "llama.cpp returned no greedy token");
            return -8;
        }
        // Some older GPT-NeoX GGUF files mark the padding token as EOG.  It
        // was excluded from the candidate scan above because padding is not
        // a generation terminator.
        if (llama_vocab_is_eog(handle->vocab, next)) {
            break;
        }

        char piece_buffer[256];
        int32_t piece_size = llama_token_to_piece(
            handle->vocab,
            next,
            piece_buffer,
            static_cast<int32_t>(sizeof(piece_buffer)),
            0,
            false);
        std::vector<char> piece_storage;
        if (piece_size < 0) {
            piece_storage.resize(static_cast<size_t>(-piece_size));
            piece_size = llama_token_to_piece(
                handle->vocab,
                next,
                piece_storage.data(),
                static_cast<int32_t>(piece_storage.size()),
                0,
                false);
        }
        if (piece_size < 0) {
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error_message, error_message_size, "llama.cpp token piece conversion failed");
            return -9;
        }
        if (piece_storage.empty()) {
            generated.append(piece_buffer, static_cast<size_t>(piece_size));
        } else {
            generated.append(piece_storage.data(), static_cast<size_t>(piece_size));
        }
        tokens.push_back(next);
        ++count;

        llama_batch next_batch = llama_batch_init(1, 0, 1);
        if (next_batch.token == nullptr || next_batch.logits == nullptr) {
            llama_batch_free(next_batch);
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error_message, error_message_size, "llama.cpp could not allocate generation batch");
            return -10;
        }
        next_batch.token[0] = next;
        next_batch.pos[0] = static_cast<llama_pos>(tokens.size() - 1);
        next_batch.n_seq_id[0] = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.logits[0] = 1;
        next_batch.n_tokens = 1;
        if (llama_decode(handle->context, next_batch) != 0) {
            llama_batch_free(next_batch);
            llama_memory_clear(llama_get_memory(handle->context), true);
            set_error(error_message, error_message_size, "llama.cpp generation decode failed");
            return -11;
        }
        llama_batch_free(next_batch);
        logits_index = 0;
    }

    llama_memory_clear(llama_get_memory(handle->context), true);
    if (generated.size() + 1 > output_size) {
        set_error(error_message, error_message_size, "generation output buffer is too small");
        return -12;
    }
    std::memcpy(output, generated.data(), generated.size());
    output[generated.size()] = '\0';
    *generated_tokens = count;
    return 0;
}
