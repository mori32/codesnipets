//! llama.cpp candidate evaluator.
//!
//! The native bridge is compiled only with the llama-cpp feature. The
//! bridge owns the model and context; Rust serializes calls so one context is
//! never decoded concurrently. The public scorer still implements the same
//! model-independent `ContextScorer` contract as the test scorers.

use std::ffi::{CStr, CString, NulError, c_char, c_void};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::{CandidatePrefix, ContextScorer, ScoreError};

unsafe extern "C" {
    fn ai_llama_create(
        model_path: *const c_char,
        context_length: u32,
        threads: i32,
        gpu_layers: i32,
        max_sequences: u32,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> *mut c_void;

    fn ai_llama_free(handle: *mut c_void);

    fn ai_llama_score_candidates_batch(
        handle: *mut c_void,
        context: *const c_char,
        candidates: *const *const c_char,
        candidate_count: u32,
        token_limit: u32,
        include_eos: u8,
        scores: *mut f32,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> i32;

    fn ai_llama_score_candidates_mean_softmax_open_batch(
        handle: *mut c_void,
        context: *const c_char,
        candidates: *const *const c_char,
        candidate_count: u32,
        scores: *mut f32,
        decode_batch_count: *mut u32,
        max_sequences_per_decode_batch: *mut u32,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> i32;

    fn ai_llama_count_candidate_tokens(
        handle: *mut c_void,
        context: *const c_char,
        candidate: *const c_char,
        token_count: *mut u32,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> i32;

    fn ai_llama_score_candidate(
        handle: *mut c_void,
        context: *const c_char,
        candidate: *const c_char,
        token_limit: u32,
        score: *mut f32,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> i32;

    fn ai_llama_score_candidate_to_eos(
        handle: *mut c_void,
        context: *const c_char,
        candidate: *const c_char,
        score: *mut f32,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> i32;

    fn ai_llama_generate_greedy(
        handle: *mut c_void,
        prompt: *const c_char,
        max_tokens: u32,
        japanese_only: u8,
        output: *mut c_char,
        output_size: usize,
        generated_tokens: *mut u32,
        error_message: *mut c_char,
        error_message_size: usize,
    ) -> i32;
}

/// Configuration used when loading a GGUF model through llama.cpp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaCppConfig {
    /// Context window requested from llama.cpp. Zero uses the model default.
    pub context_length: u32,
    /// Number of CPU threads. Zero lets llama.cpp choose.
    pub threads: i32,
    /// Number of layers to offload. Zero is CPU-only; -1 requests all layers.
    pub gpu_layers: i32,
    /// Maximum number of independent candidate sequences decoded together.
    /// This is capped again by the context's token batch capacity.
    pub max_sequences: u32,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            context_length: 2048,
            threads: 0,
            gpu_layers: 0,
            max_sequences: 8,
        }
    }
}

/// Actual batching performed for one candidate-scoring call.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CandidateBatchStats {
    pub candidate_count: usize,
    pub decode_batch_count: usize,
    pub max_sequences_per_decode_batch: usize,
}

/// A llama.cpp-backed evaluator.
///
/// `ContextScorer` uses summed log-probability through the common shortest
/// candidate-token depth, which avoids tokenization-length bias during live
/// IME prefix reranking. `score_complete_sentences_to_eos` is the separate
/// sentence/segment endpoint mode: it scores every token and then EOS with no
/// length penalty, matching the Qiita-style comparison.
#[derive(Debug)]
pub struct LlamaCppScorer {
    handle: Mutex<NativeHandle>,
}

impl LlamaCppScorer {
    /// Loads a GGUF model and creates one reusable llama.cpp context.
    ///
    /// # Errors
    ///
    /// Returns an error when the path contains an embedded NUL, the native
    /// bridge cannot load the model, or the context cannot be created.
    pub fn load(path: impl AsRef<Path>, config: LlamaCppConfig) -> Result<Self, ScoreError> {
        let path = CString::new(path.as_ref().to_string_lossy().as_bytes())
            .map_err(|error| ScoreError::new(format!("model path contains NUL: {error}")))?;
        let mut error_message = vec![0_u8; 1024];
        let handle = unsafe {
            ai_llama_create(
                path.as_ptr(),
                config.context_length,
                config.threads,
                config.gpu_layers,
                config.max_sequences,
                error_message.as_mut_ptr().cast(),
                error_message.len(),
            )
        };
        let handle = NonNull::new(handle)
            .ok_or_else(|| ScoreError::new(native_error(&error_message, "model load failed")))?;
        Ok(Self {
            handle: Mutex::new(NativeHandle(handle)),
        })
    }

    /// Scores complete sentence or IME-segment alternatives through EOS.
    ///
    /// This intentionally does not truncate to the shortest candidate and
    /// does not apply length normalization. It is suitable when each input is
    /// a closed segment (word/phrase) rather than an open-ended prefix.
    ///
    /// # Errors
    ///
    /// Returns an error when a sentence contains an embedded NUL or the
    /// native bridge cannot score the sequence through EOS.
    pub fn score_complete_sentences_to_eos(
        &self,
        sentences: &[CandidatePrefix],
    ) -> Result<Vec<f32>, ScoreError> {
        self.score_segments_to_eos("", sentences)
    }

    /// Scores closed IME segments after an already committed context.
    ///
    /// The returned score is the sum for the complete candidate token
    /// sequence followed by EOS. Candidates are not truncated to a shared
    /// token depth and are not length-normalized.
    ///
    /// # Errors
    ///
    /// Returns an error when the context/candidate contains an embedded NUL
    /// or the native bridge cannot score the sequence through EOS.
    pub fn score_segments_to_eos(
        &self,
        context: &str,
        candidates: &[CandidatePrefix],
    ) -> Result<Vec<f32>, ScoreError> {
        let context = CString::new(context).map_err(|error| string_nul_error(&error))?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| ScoreError::new("llama.cpp scorer lock is poisoned"))?;
        let sentence_strings: Vec<_> = candidates
            .iter()
            .map(|sentence| {
                CString::new(sentence.text.as_str()).map_err(|error| string_nul_error(&error))
            })
            .collect::<Result<_, _>>()?;
        if sentence_strings.is_empty() {
            return Ok(Vec::new());
        }

        if std::env::var("AI_ENGINE_LLM_SCORING").ok().as_deref() != Some("single") {
            let candidate_ptrs: Vec<_> = sentence_strings
                .iter()
                .map(|value| value.as_ptr())
                .collect();
            let mut scores = vec![0.0_f32; sentence_strings.len()];
            let mut error_message = vec![0_u8; 1024];
            let result = unsafe {
                ai_llama_score_candidates_batch(
                    handle.0.as_ptr(),
                    context.as_ptr(),
                    candidate_ptrs.as_ptr(),
                    candidate_ptrs.len() as u32,
                    0,
                    1,
                    scores.as_mut_ptr(),
                    error_message.as_mut_ptr().cast(),
                    error_message.len(),
                )
            };
            if result != 0 {
                return Err(ScoreError::new(native_error(
                    &error_message,
                    "complete sentence EOS batch scoring failed",
                )));
            }
            return Ok(scores);
        }

        let mut scores = Vec::with_capacity(sentence_strings.len());
        for sentence in &sentence_strings {
            let mut score = 0.0_f32;
            let mut error_message = vec![0_u8; 1024];
            let result = unsafe {
                ai_llama_score_candidate_to_eos(
                    handle.0.as_ptr(),
                    context.as_ptr(),
                    sentence.as_ptr(),
                    &raw mut score,
                    error_message.as_mut_ptr().cast(),
                    error_message.len(),
                )
            };
            if result != 0 {
                return Err(ScoreError::new(native_error(
                    &error_message,
                    "complete sentence EOS scoring failed",
                )));
            }
            scores.push(score);
        }
        Ok(scores)
    }

    /// Scores every candidate token from BOS (when `context` is empty) to the
    /// final candidate token, leaving EOS open. Each returned value is the
    /// arithmetic mean of the target token's softmax probability over that
    /// candidate's own complete token sequence. Unlike `score_prefixes`, this
    /// does not truncate candidates to a shared token depth; unlike the EOS
    /// methods, it does not score EOS or return log-probability sums.
    ///
    /// # Errors
    ///
    /// Returns an error when text contains an embedded NUL, the candidate
    /// batch exceeds the native argument limit, or llama.cpp cannot score a
    /// full candidate sequence.
    pub fn score_open_candidate_mean_softmax(
        &self,
        context: &str,
        candidates: &[CandidatePrefix],
    ) -> Result<Vec<f32>, ScoreError> {
        self.score_open_candidate_mean_softmax_with_stats(context, candidates)
            .map(|(scores, _)| scores)
    }

    /// Scores open candidates and reports how many independent sequences the
    /// native decoder put into its largest single decode batch.
    pub fn score_open_candidate_mean_softmax_with_stats(
        &self,
        context: &str,
        candidates: &[CandidatePrefix],
    ) -> Result<(Vec<f32>, CandidateBatchStats), ScoreError> {
        let context = CString::new(context).map_err(|error| string_nul_error(&error))?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| ScoreError::new("llama.cpp scorer lock is poisoned"))?;
        let candidate_strings: Vec<_> = candidates
            .iter()
            .map(|candidate| {
                CString::new(candidate.text.as_str()).map_err(|error| string_nul_error(&error))
            })
            .collect::<Result<_, _>>()?;
        if candidate_strings.is_empty() {
            return Ok((Vec::new(), CandidateBatchStats::default()));
        }
        let candidate_ptrs = candidate_strings
            .iter()
            .map(|candidate| candidate.as_ptr())
            .collect::<Vec<_>>();
        let mut scores = vec![0.0_f32; candidate_strings.len()];
        let mut decode_batch_count = 0_u32;
        let mut max_sequences_per_decode_batch = 0_u32;
        let mut error_message = vec![0_u8; 1024];
        let result = unsafe {
            ai_llama_score_candidates_mean_softmax_open_batch(
                handle.0.as_ptr(),
                context.as_ptr(),
                candidate_ptrs.as_ptr(),
                candidate_ptrs.len() as u32,
                scores.as_mut_ptr(),
                &raw mut decode_batch_count,
                &raw mut max_sequences_per_decode_batch,
                error_message.as_mut_ptr().cast(),
                error_message.len(),
            )
        };
        if result != 0 {
            return Err(ScoreError::new(native_error(
                &error_message,
                "full candidate mean-softmax scoring failed",
            )));
        }
        Ok((
            scores,
            CandidateBatchStats {
                candidate_count: candidate_strings.len(),
                decode_batch_count: decode_batch_count as usize,
                max_sequences_per_decode_batch: max_sequences_per_decode_batch as usize,
            },
        ))
    }

    /// Greedily generates up to `max_tokens` after a prompt.
    ///
    /// Generation stops early when llama.cpp selects an end-of-generation
    /// token. The returned string is the decoded UTF-8 token-piece sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt contains an embedded NUL or the
    /// native bridge cannot decode or generate the requested sequence.
    pub fn generate_greedy(&self, prompt: &str, max_tokens: u32) -> Result<String, ScoreError> {
        self.generate_greedy_with_filter(prompt, max_tokens, false)
    }

    /// Greedily generates up to `max_tokens`, allowing only Japanese text.
    ///
    /// The native sampler masks token pieces containing Latin letters,
    /// digits, ASCII punctuation, Markdown markers, and other non-Japanese
    /// characters. ASCII whitespace and Japanese punctuation are retained so
    /// normal sentence boundaries remain possible. This is a lexical output
    /// constraint, not a grammar or semantic constraint.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt contains an embedded NUL or the
    /// native bridge cannot decode or generate the requested sequence.
    pub fn generate_greedy_japanese_only(
        &self,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<String, ScoreError> {
        self.generate_greedy_with_filter(prompt, max_tokens, true)
    }

    fn generate_greedy_with_filter(
        &self,
        prompt: &str,
        max_tokens: u32,
        japanese_only: bool,
    ) -> Result<String, ScoreError> {
        let prompt = CString::new(prompt).map_err(|error| string_nul_error(&error))?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| ScoreError::new("llama.cpp scorer lock is poisoned"))?;
        let mut output = vec![0_u8; 256 * 1024];
        let mut generated_tokens = 0_u32;
        let mut error_message = vec![0_u8; 1024];
        let result = unsafe {
            ai_llama_generate_greedy(
                handle.0.as_ptr(),
                prompt.as_ptr(),
                max_tokens,
                u8::from(japanese_only),
                output.as_mut_ptr().cast(),
                output.len(),
                &raw mut generated_tokens,
                error_message.as_mut_ptr().cast(),
                error_message.len(),
            )
        };
        if result != 0 {
            return Err(ScoreError::new(native_error(
                &error_message,
                "greedy generation failed",
            )));
        }
        let _ = generated_tokens;
        let end = output
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(output.len());
        String::from_utf8(output[..end].to_vec())
            .map_err(|error| ScoreError::new(format!("generated output is not UTF-8: {error}")))
    }
}

impl ContextScorer for LlamaCppScorer {
    fn score_prefixes(
        &self,
        context: &str,
        candidates: &[CandidatePrefix],
    ) -> Result<Vec<f32>, ScoreError> {
        let context = CString::new(context).map_err(|error| string_nul_error(&error))?;
        let handle = self
            .handle
            .lock()
            .map_err(|_| ScoreError::new("llama.cpp scorer lock is poisoned"))?;
        let candidate_strings: Vec<_> = candidates
            .iter()
            .map(|candidate| {
                CString::new(candidate.text.as_str()).map_err(|error| string_nul_error(&error))
            })
            .collect::<Result<_, _>>()?;
        if candidate_strings.is_empty() {
            return Ok(Vec::new());
        }

        if std::env::var("AI_ENGINE_LLM_SCORING").ok().as_deref() != Some("single") {
            let candidate_ptrs: Vec<_> = candidate_strings
                .iter()
                .map(|value| value.as_ptr())
                .collect();
            let mut scores = vec![0.0_f32; candidate_strings.len()];
            let mut error_message = vec![0_u8; 1024];
            let result = unsafe {
                ai_llama_score_candidates_batch(
                    handle.0.as_ptr(),
                    context.as_ptr(),
                    candidate_ptrs.as_ptr(),
                    candidate_ptrs.len() as u32,
                    0,
                    0,
                    scores.as_mut_ptr(),
                    error_message.as_mut_ptr().cast(),
                    error_message.len(),
                )
            };
            if result != 0 {
                return Err(ScoreError::new(native_error(
                    &error_message,
                    "candidate batch scoring failed",
                )));
            }
            return Ok(scores);
        }

        let mut candidate_token_counts = Vec::with_capacity(candidates.len());
        for candidate in &candidate_strings {
            let mut token_count = 0_u32;
            let mut error_message = vec![0_u8; 1024];
            let result = unsafe {
                ai_llama_count_candidate_tokens(
                    handle.0.as_ptr(),
                    context.as_ptr(),
                    candidate.as_ptr(),
                    &raw mut token_count,
                    error_message.as_mut_ptr().cast(),
                    error_message.len(),
                )
            };
            if result != 0 {
                return Err(ScoreError::new(native_error(
                    &error_message,
                    "candidate token counting failed",
                )));
            }
            if token_count == 0 {
                return Err(ScoreError::new("candidate has no tokens"));
            }
            candidate_token_counts.push(token_count);
        }
        let token_limit = candidate_token_counts.iter().copied().min().unwrap_or(0);

        let mut scores = Vec::with_capacity(candidates.len());
        for candidate in &candidate_strings {
            let mut score = 0.0_f32;
            let mut error_message = vec![0_u8; 1024];
            let result = unsafe {
                ai_llama_score_candidate(
                    handle.0.as_ptr(),
                    context.as_ptr(),
                    candidate.as_ptr(),
                    token_limit,
                    &raw mut score,
                    error_message.as_mut_ptr().cast(),
                    error_message.len(),
                )
            };
            if result != 0 {
                return Err(ScoreError::new(native_error(
                    &error_message,
                    "candidate scoring failed",
                )));
            }
            scores.push(score);
        }
        Ok(scores)
    }
}

#[derive(Debug)]
struct NativeHandle(NonNull<c_void>);

// The native handle is only accessed while LlamaCppScorer::handle is locked.
unsafe impl Send for NativeHandle {}

impl Drop for NativeHandle {
    fn drop(&mut self) {
        unsafe { ai_llama_free(self.0.as_ptr()) };
    }
}

fn string_nul_error(error: &NulError) -> ScoreError {
    ScoreError::new(format!("text contains NUL: {error}"))
}

fn native_error(buffer: &[u8], fallback: &str) -> String {
    CStr::from_bytes_until_nul(buffer)
        .ok()
        .and_then(|message| (!message.to_bytes().is_empty()).then(|| message.to_string_lossy()))
        .map_or_else(|| fallback.to_owned(), std::borrow::Cow::into_owned)
}
