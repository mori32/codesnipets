# CRF N-best local LLM alpha experiment

## Goal and score definition

This experiment adds a local GGUF language-model signal to the existing CRF N-best reranker and estimates only one scalar, `llm_alpha`. For a rendered candidate surface (y), tokenize the whole candidate from an empty context with BOS supplied, and for each candidate token (t_i) read its target-token probability from the softmax after the preceding prefix:

\[
q(y) = \frac{1}{T_y}\sum_{i=1}^{T_y} p(t_i\mid \mathrm{BOS},t_{<i}).
\]

The final candidate token is scored; EOS is not appended or scored. BOS is context, not a scored token. This is the arithmetic mean of raw token softmax probabilities, not mean log probability, sequence likelihood, or perplexity. To combine features with different units, center each score within the sentence's N-best list and divide by that feature's pooled within-list RMS, estimated from the training examples:

\[
s(y) = z_{\mathrm{CRF}}(y) + \alpha z_{\mathrm{LLM}}(y), \qquad
s_{\mathrm{CRF}} = s_{\mathrm{first-order}} + s_{\mathrm{trigram}}.
\]

Here `z_CRf=(s_CRf-mean_N(s_CRf))/R_CRf` and `z_LLM=(q-mean_N(q))/R_LLM`. Means are per sentence; each `R` is the RMS of these centered values pooled across training lists. At inference, the per-list means may be omitted because they add only a candidate-independent constant. Fit an unregularized one-dimensional listwise negative log likelihood over the training candidate sets using safeguarded scalar Newton steps; the combined list receives the usual softmax normalization inside that objective. The learned alpha is dimensionless. All CRF feature weights remain frozen. Alpha starts at zero for this experiment.

## Inputs and locations

The experiment uses the completed chunk003 CRF run and its normalized data, not a prior LLM checkpoint:

- CRF input model: `E:\lmwork-target\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01\model\model.json`
- Normalized training cache: `E:\lmwork-int\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01\work\page-blocks`
- Fixed 5,000-sentence test sample: `E:\lmwork-int\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01-posthoc-ablation\work\test-sample.json`
- GGUF directory: `D:\Git\codesnipets\aiEngine\tests\fixtures\GGUF`
- Selected initial GGUF: `qwen2.5-0.5b-instruct-q8_0.gguf`
- Current verified model output: `E:\lmwork-target\crf-wiki003-llm-alpha-qwen25-05b-20261007-run03-batch128-4threads\model`
- Current work data and candidate-score audit: `E:\lmwork-int\crf-wiki003-llm-alpha-qwen25-05b-20261007-run03-batch128-4threads\work`

The earlier 10,000/5,000 attempt was stopped during feature scoring and is preserved separately in run02; it is not used to initialize or continue this run.

The first 1,000/500 run03 pass produced the 1,000 training candidate-feature rows but stopped partway through the held-out pass when raw N-best omitted a gold path. That partial test output was preserved. The corrected pass reuses only those training rows to refit the normalized scalar alpha and scores the exact held-out 500 sentences again; it does not reuse partial test scores. The interrupted 4-thread pass that used an 8-sequence decoder limit is also preserved separately and is not used for the final metrics.

Training examples are selected deterministically and stratified across page-cache blocks (100 from each block for the default 10,000). The fixed test sample is never used for alpha fitting. The selected GGUF is passed as a path; to try another file from the same folder, pass that filename, or pass the directory plus `--model-name <exact-filename>`. Runs are separate per GGUF; the CLI does not silently load or train every model in the directory.

Each run-level output directory immediately below `E:\lmwork-int` or `E:\lmwork-target` may have one `README.md`; do not add README files to deeper `work`/`model` subfolders. The source corpus, old CRF model, page cache, test sample, and GGUF fixtures are read-only inputs. Generated data and model output stay outside the repository. Cargo executable build products are the only generated binaries kept in the repository tree.

## Build and run

`third_party\llama.cpp` is the source tree. The default setup is CPU-only and stays under `third_party\llama.cpp-install`; the CUDA build has separate build/install trees so it does not replace the CPU build. For CUDA, install the NVIDIA CUDA Toolkit first (the display driver alone does not provide `nvcc` or the build libraries), then run:

```powershell
.\scripts\setup-llama-cpp.ps1 -Backend CUDA
. .\third_party\llama.cpp-env-cuda.ps1
cargo build --release -p ai_engine_crf --features llama-cpp --bin crf-llm-alpha
```

The setup script requires `CUDA_PATH` and `nvcc.exe`, builds llama.cpp with `GGML_CUDA=ON`, and writes CUDA artifacts to `third_party\llama.cpp-build-cuda` and `third_party\llama.cpp-install-cuda`. The regular CPU setup remains:

```powershell
.\scripts\setup-llama-cpp.ps1
. .\third_party\llama.cpp-env.ps1
cargo build --release -p ai_engine_crf --features llama-cpp --bin crf-llm-alpha
```

Example command for the selected Qwen model (PowerShell):

```powershell
. .\third_party\llama.cpp-env-cuda.ps1
cargo run --release -p ai_engine_crf --features llama-cpp --bin crf-llm-alpha -- `
  --model-in E:\lmwork-target\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01\model\model.json `
  --page-cache E:\lmwork-int\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01\work\page-blocks `
  --test-sample E:\lmwork-int\crf-wiki003-n128-primary-listwise-affix100-crossbi-20261007-run01-posthoc-ablation\work\test-sample.json `
  --llama-model D:\Git\codesnipets\aiEngine\tests\fixtures\GGUF\qwen2.5-0.5b-instruct-q8_0.gguf `
  --train-sentences 1000 --test-sentences 500 --nbest 128 --threads 4 --gpu-layers -1 `
  --context-length 32768 `
  --training-features E:\lmwork-int\crf-wiki003-llm-alpha-qwen25-05b-20261007-run03-1000-500\work\training-features.jsonl `
  --model-out-dir E:\lmwork-target\crf-wiki003-llm-alpha-qwen25-05b-20261007-run03-batch128-4threads\model `
  --work-dir E:\lmwork-int\crf-wiki003-llm-alpha-qwen25-05b-20261007-run03-batch128-4threads\work
```

To choose another GGUF from the folder, pass the folder itself and its exact filename, for example `--llama-model ...\GGUF --model-name Qwen2.5-1.5B-Instruct-Q8_0.gguf`, and use new, empty target/work directories for that run. `--threads` defaults to 8, `--gpu-layers` defaults to 0 (CPU); set it to `-1` to request all model layers on the GPU, or a positive number to offload that many layers. A GPU-layer request now fails immediately if the loaded llama.cpp build has no GPU-offload backend, instead of silently behaving like a CPU-only build. The alpha CLI defaults to a 32,768-token context so a wide N-best batch can fit. The CLI rejects a non-empty output directory to avoid overwriting prior results.

For this N-best reranker, `n_seq_max` is set to `--nbest`; the token batch capacity is set from `--context-length`. Candidate strings that render identically are deduplicated before LLM scoring because they receive the same LLM score. Long sequences can still split a requested batch when padded token count exceeds context capacity. Progress and final `metrics.json` report the actual maximum sequences per decoder call, decoder-call count, unique candidate count, and how many test sentences reached the requested full width. The 4-thread verification run recorded one decoder call per sentence, 88.4 unique surfaces per sentence on average, a maximum width of 128, and 5/500 sentences at the full width.

## Outputs and interpretation

The work directory records the run configuration, sampled training and fixed-test candidate features, alpha-fit summary, and held-out metrics. The model directory contains the resulting `model.json` and a save/reload verification. The output model stores `llm_alpha` plus the training-derived CRF and LLM RMS normalizers needed by the combined score.

The fixed-test report includes raw N=128 gold coverage, first-order bigram top-1, CRF post-trigram top-1, CRF+LLM top-1, and Unicode-scalar Levenshtein character accuracy for each relevant stage. The current quick experiment evaluates on 500 held-out sentences; the fit loss/accuracy is in-sample and must not be described as test performance. Because the test and training examples use the same MeCab-derived pseudo-gold conventions as the CRF run, report this as a controlled ranking experiment, not an independently human-verified accuracy estimate.

## Verified 1,000/500 result (2026-10-07)

Alpha was fitted on the separate 1,000 training feature rows, then tested on the first 500 fixed test sentences. CRF parameters remained frozen. The fit used per-list mean-centering and training-derived pooled RMS scales; the held-out test was not used to fit alpha.

| Metric | CRF-only | CRF + local LLM | Change |
|---|---:|---:|---:|
| Raw N=128 gold coverage | 93.8% | 93.8% | — |
| First-order/bigram top-1 | 70.6% | — | — |
| Post-trigram top-1 | 70.4% | 71.8% | +1.4 pp vs post-trigram |
| Post-trigram character accuracy | 68.774% | 70.552% | +1.778 pp |

The first-order character accuracy was 68.122%; CRF+LLM improves that by 2.430 pp. The test contained 469/500 gold paths in raw N-best; the 31 absent gold paths remained misses (they were not injected into evaluation). The learned alpha was `0.49123458717782864`. Save/reload verification passed, and `verification.json` confirms the CRF weights were unchanged.

This wide batch changed the actual decoder sequence capacity from 8 to 128 and the context/token batch capacity to 32,768. It did **not** make each of the 500 held-out sentences faster: total test scoring took 3,125.6 seconds (0.160 sentence/s), so the change primarily fixes the batching behavior, not demonstrated throughput. The run used 4 CPU threads and no GPU layers.

## Middle-slice 1,000/500 result (2026-10-08)

To check the unusually low first-500 character score, this run used the central 1,000 rows (zero-based `500..1500`) of the already scored 2,000-row training-feature file, and the central 500 sentences (zero-based `2250..2750`) of the unchanged fixed 5,000-sentence test sample. It did not rescore the training candidates. The CLI records both offsets in `run.json`; `training-features.jsonl` in this run contains the selected 1,000 rows. The previous run's full training-feature file and fixed test sample remain unchanged.

Run-level work/model directories are immediate children of `E:\lmwork-int` and `E:\lmwork-target`, respectively, so their README files are at the requested depth; no README was created in nested subfolders. The run used one CPU thread, `gpu_layers=-1`, Qwen2.5-0.5B Q8_0 on the RTX 3060, N=128, and 32,768 context. The CUDA log identified the GPU. Model save/reload passed and verification confirms CRF weights were unchanged.

| Metric | CRF-only | CRF + local LLM | Change |
|---|---:|---:|---:|
| Raw N=128 gold coverage | 95.4% | 95.4% | — |
| First-order/bigram top-1 | 70.6% | — | — |
| Post-trigram top-1 | 70.6% | 68.8% | −1.8 pp |
| Bigram character accuracy | 83.2517% | — | — |
| Post-trigram character accuracy | 83.3918% | 83.2329% | −0.1599 pp |

The fitted alpha was `0.43382766755376784`; the one-dimensional fit converged in 3 iterations on the 1,000 training rows. This is an optimizer convergence result, not evidence that the coefficient transfers across samples. Test scoring took about 911 seconds; all 500 calls had a maximum sequence width of 128, with 6/500 sentences reaching all 128 unique candidate surfaces.

The middle test slice is still clustered: it contains 500 sentences from only 9 pages, and three pages account for 424/500 sentences. Thus the move from the first-500 character accuracy (68.7741%) to the middle-500 result (83.3918%) is consistent with strong page/order effects; neither 500-sentence slice should be treated as representative of the full 5,000-sentence test. The full-test CRF-only character accuracy remains 96.2618%.

### Middle-slice composition diagnosis

The fixed 5,000-sentence sample contains 87 distinct pages; its three most frequent pages account for 27.8% of rows. The contiguous middle 500 instead contains only 9 pages, with `愛知県立岡崎北高等学校` (194 rows), `宮木汐音` (116), and `パブリック・エナミー` (114) accounting for 424/500 (84.8%). This is a strong page-order sampling bias, not a balanced 500-sentence subset. Mean input length is 30.1 Unicode scalars (median 28), versus 26.2 (median 21) across all 5,000. Section-heading rows are 12.8% versus 12.3% overall, so headings alone do not explain the low score.

Recomputed from the saved 500 candidate-feature rows with the saved training RMS normalizers and alpha:

| Slice | Rows | CRF top-1 | CRF+LLM top-1 | CRF char acc. | CRF+LLM char acc. |
|---|---:|---:|---:|---:|---:|
| Three dominant pages | 424 | 317/424 (74.8%) | 306/424 (72.2%) | 81.972% | 81.878% |
| Other six pages | 76 | 36/76 (47.4%) | 38/76 (50.0%) | 90.849% | 90.352% |
| All middle rows | 500 | 353/500 (70.6%) | 344/500 (68.8%) | 83.392% | 83.233% |

The dominant pages contribute 2,287 of the CRF's 2,508 character edit operations (91.2%), though they are 84.8% of rows. The reranker changes 66/500 selected paths; 19 CRF-correct selections become incorrect and 10 incorrect selections become correct, for a net loss of 9 top-1 sentences. The absolute character accuracy is therefore strongly affected by which pages the contiguous slice happens to include; on this fixed slice, adding the LLM slightly worsens both metrics.

Composition counts came from the unchanged 5,000-row `test-sample.json`; per-page and per-slice prediction metrics were recomputed from `test-features.jsonl`, `fit-metrics.json`, and the saved alpha. No model, sample, or feature rows were modified.

Artifacts:

- Work/config/features/metrics: `E:\lmwork-int\crf-wiki003-llm-alpha-qwen25-05b-20261008-run02-middle1000-500`
- Alpha model and verification: `E:\lmwork-target\crf-wiki003-llm-alpha-qwen25-05b-20261008-run02-middle1000-500`
- CLI selection options: `--train-start-sentence`, `--train-stride`, `--test-start-sentence`, and `--test-stride`; starts default to zero and strides to one.

## Evenly spaced 1,000/500 result (2026-10-08)

To avoid the severe page clustering in the contiguous middle-500 slice, the training selection took every second row (`0, 2, ..., 1998`) from the prior 2,000-row training-feature pool. That pool was not the corpus prefix: it had been selected deterministically across all 100 normalized page-cache blocks, with 20 rows per block. The resulting training rows cover 119 distinct pages, with at most 10 rows from any one page. The fixed test selected every tenth row (`0, 10, ..., 4990`) from the unchanged 5,000-row range. Validation matched all 1,000 training rows and all 500 test rows to those exact source indices.

The spaced test covers 83 distinct pages; its top three pages account for 27.8% of rows, matching the 27.8% top-three share in the full 5,000-row range (versus 84.8% in the contiguous middle 500). The fit/test split stayed separate; CRF weights were frozen. The run used the same Qwen2.5-0.5B Q8_0 GGUF on RTX 3060 with one CPU thread, full GPU offload, N=128, and context 32,768. Test scoring took 811.1 seconds, with 45,295 unique candidate strings, 500 decoder calls, and 8/500 sentences reaching all 128 unique candidates.

| Metric | CRF-only | CRF + local LLM | Change |
|---|---:|---:|---:|
| Raw N=128 gold coverage | 92.0% | 92.0% | — |
| First-order/bigram top-1 | 63.2% | — | — |
| Post-trigram top-1 | 63.8% | 62.6% | −1.2 pp |
| Bigram character accuracy | 86.722% | — | — |
| Post-trigram character accuracy | 86.952% | 86.826% | −0.126 pp |

The fitted alpha was `0.41374998157339266`; the one-dimensional fit converged in three iterations (training listwise NLL 2.080028→2.001935). On the more page-representative held-out sample, CRF-only character accuracy was 86.952%, 3.560 pp above the contiguous middle-500 result; the LLM reranker still slightly reduced both held-out top-1 and character accuracy. Model save/reload passed and the verification records that CRF weights were unchanged.

Artifacts:

- Work/config/features/metrics: `E:\lmwork-int\crf-wiki003-llm-alpha-qwen25-05b-20261008-run03-even1000-500`
- Alpha model and verification: `E:\lmwork-target\crf-wiki003-llm-alpha-qwen25-05b-20261008-run03-even1000-500`

## Implementation entry points

- Native BOS-to-open-candidate mean-softmax scorer: `crates/ai_engine_llm/native/ai_llama_bridge.cpp`
- Rust scorer API: `crates/ai_engine_llm/src/llama_cpp.rs`
- Alpha-aware path score and persisted alpha: `crates/ai_engine_crf/src/lib.rs`, `crates/ai_engine_crf/src/output.rs`
- Scalar fitting and evaluation CLI: `crates/ai_engine_crf/src/bin/crf-llm-alpha.rs`
