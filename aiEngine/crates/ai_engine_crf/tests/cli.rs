//! End-to-end offline binary test with a compressed Japanese corpus.
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use ai_engine_crf::{Model, corpus, output};
use ai_engine_dict::{TrainingSeeds, TrainingWord};

#[derive(Debug)]
struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn capacity_cutoff_retries_without_duplicate_updates_and_stops_if_strong_weights_cannot_fit() {
    use ai_engine_crf::{Feature, cutoff};
    let directory = std::env::temp_dir().join(format!(
        "ai-engine-crf-cutoff-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let temp = Temp(directory);
    let mut seed = Model::from_seeds(
        TrainingSeeds {
            pos_names: vec!["[BOS]".into(), "[EOS]".into(), "名詞,一般,*,*,*,*,*".into()],
            pos_bigrams: vec![(1, 3, 0.0), (3, 2, 0.0)],
            words: (0..256)
                .map(|n| TrainingWord {
                    surface: if n == 0 {
                        "阿".into()
                    } else if n == 1 {
                        "あ".into()
                    } else {
                        format!("語{n}")
                    },
                    reading: if n < 2 {
                        "あ".into()
                    } else {
                        format!("よみ{n}")
                    },
                    pos_id: 3,
                    log_unigram: -1.0,
                    model_only: false,
                })
                .collect(),
        },
        1e-8,
    )
    .unwrap();
    seed.first.scale = 0.5;
    seed.third.scale = 0.25;
    let count = u32::try_from(seed.words.len()).unwrap();
    for a in 3..count {
        for b in 3..count {
            seed.first.values.insert(Feature::Bigram(a, b), 1e-5);
            seed.third.values.insert([1, a, b], -1e-5);
        }
    }
    seed.first.values.insert(Feature::Word(3), 0.4);
    seed.first.values.insert(Feature::Pos(1, 3), -0.2);
    let source = temp.0.join("source.json");
    output::save(&seed, &source).unwrap();
    let input = temp.0.join("input.jsonl.bz2");
    let train = (1..100).find(|&p| !corpus::validation_page(p)).unwrap();
    let dev = (1..100).find(|&p| corpus::validation_page(p)).unwrap();
    let mut writer = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
    for page in [train, train, train, train, train, train, train, train, dev] {
        serde_json::to_writer(&mut writer, &serde_json::json!({"page_id":page,"title":"容量試験",
            "sentence":"阿", "tokens":[{"surface":"阿","feature":"名詞,一般,*,*,*,*,阿,ア,ア","reading":"ア","unknown":false}]})).unwrap();
        writeln!(writer).unwrap();
    }
    std::fs::write(&input, writer.finish().unwrap()).unwrap();
    let invoke = |name: &str, model: &std::path::Path, enabled: bool| {
        Command::new(env!("CARGO_BIN_EXE_ai_engine_crf"))
            .arg("--input")
            .arg(&input)
            .arg("--model-in")
            .arg(model)
            .arg("--output")
            .arg(temp.0.join(name))
            .args([
                "--epochs",
                "1",
                "--threads",
                "8",
                "--batch-size",
                "2",
                "--shuffle-buffer",
                "4",
                "--page-block-size",
                "1",
                "--nbest",
                "8",
                "--max-sentences",
                "8",
                "--max-validation-sentences",
                "1",
                "--max-eval-sentences",
                "1",
                "--max-parameter-mib",
                "1",
                "--delta-cutoff",
                if enabled { "0.0001" } else { "0" },
                "--delta-cutoff-max",
                "0.01",
            ])
            .output()
            .unwrap()
    };
    let recovered = invoke("recovered", &source, true);
    assert!(
        recovered.status.success(),
        "{}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    let events = std::fs::read_to_string(temp.0.join("recovered/delta-cutoff.jsonl")).unwrap();
    assert_eq!(events.lines().count(), 1);
    let event: serde_json::Value = serde_json::from_str(events.lines().next().unwrap()).unwrap();
    assert_eq!(event["committed_sentences"], 0);
    assert_eq!(event["cutoff"]["removed_total"], 131_072);
    assert!(event["cutoff"]["table_bytes_after"].as_u64().unwrap() < 1 << 20);
    assert_eq!(event["validation_before"]["sentences"], 1);
    assert_eq!(event["validation_after"]["sentences"], 1);
    assert!(!temp.0.join("recovered/training-stop.json").exists());
    let metrics: serde_json::Value =
        serde_json::from_slice(&std::fs::read(temp.0.join("recovered/metrics.json")).unwrap())
            .unwrap();
    assert_eq!(metrics[1]["training_sentences_processed"], 8);
    assert_eq!(metrics[1]["training_batches"], 4);
    assert_eq!(metrics[1]["cutoff_events"].as_array().unwrap().len(), 1);
    // The retry's updates must match training a manually cut model from scratch.
    let mut reference = seed.clone();
    cutoff::apply(
        &mut reference,
        cutoff::Policy {
            minimum_delta: 1e-4,
            maximum_delta: 1e-2,
            target_fraction: 0.75,
        },
        1 << 20,
    )
    .unwrap();
    let compact = temp.0.join("compact.json");
    output::save(&reference, &compact).unwrap();
    let result = invoke("reference", &compact, false);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read(temp.0.join("recovered/model.json")).unwrap(),
        std::fs::read(temp.0.join("reference/model.json")).unwrap()
    );
    // No semantic cutoff is hidden when disabled.
    let disabled = invoke("disabled", &source, false);
    assert!(!disabled.status.success());
    assert!(!temp.0.join("disabled/delta-cutoff.jsonl").exists());
    assert_eq!(
        output::fingerprint(&output::load(&temp.0.join("disabled/stopped-model.json")).unwrap()),
        output::fingerprint(&seed)
    );
    // Even the maximum cutoff cannot delete substantial positive/negative deltas.
    for value in seed.first.values.values_mut() {
        *value = 0.2;
    }
    for value in seed.third.values.values_mut() {
        *value = -0.2;
    }
    let strong_source = temp.0.join("strong.json");
    output::save(&seed, &strong_source).unwrap();
    let failed = invoke("strong", &strong_source, true);
    assert!(!failed.status.success());
    let logs = std::fs::read_to_string(temp.0.join("strong/delta-cutoff.jsonl")).unwrap();
    assert_eq!(logs.lines().count(), 1); // Bounded retry, not an infinite loop.
    let partial: serde_json::Value = serde_json::from_slice(
        &std::fs::read(temp.0.join("strong/partial-training.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(partial["training_sentences_processed"], 0);
    assert_eq!(partial["exact_resume"], true);
    assert_eq!(
        output::fingerprint(&output::load(&temp.0.join("strong/stopped-model.json")).unwrap()),
        output::fingerprint(&seed)
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn exact_resume_keeps_shuffled_pending_batches_and_matches_uninterrupted_model() {
    use ai_engine_crf::blocks::{Block, PageBlocks};
    let directory = std::env::temp_dir().join(format!(
        "ai-engine-crf-resume-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let temp = Temp(directory);
    let dummy = PageBlocks {
        directory: PathBuf::new(),
        pages_per_block: 1,
        training_sentences: 8,
        blocks: (0..3)
            .map(|n| Block {
                file: n.to_string(),
                pages: 1,
                sentences: 0,
                source_first_record: n + 1,
                source_last_record: n + 1,
                compressed_bytes: 0,
            })
            .collect(),
    };
    let seed = (1..100_u64)
        .find(|&s| dummy.order(s, 1) == [0, 1, 2])
        .unwrap();
    let mut state = seed;
    corpus::shuffle(&mut [0, 1], &mut state);
    let mut order = [0, 1, 2, 3];
    corpus::shuffle(&mut order, &mut state);
    // Fail at the second batch of an already shuffled 4-sentence window.
    let hard = order[1];
    let seed_model = temp.0.join("seed.json");
    let model = Model::from_seeds(
        TrainingSeeds {
            pos_names: vec!["[BOS]".into(), "[EOS]".into(), "名詞,一般,*,*,*,*,*".into()],
            pos_bigrams: vec![(1, 3, 0.0), (3, 3, 0.0), (3, 2, 0.0)],
            words: [("阿", "あ"), ("あ", "あ"), ("伊", "い"), ("愛", "あい")]
                .into_iter()
                .map(|(s, r)| TrainingWord {
                    surface: s.into(),
                    reading: r.into(),
                    pos_id: 3,
                    log_unigram: -1.0,
                    model_only: false,
                })
                .collect(),
        },
        1e-8,
    )
    .unwrap();
    output::save(&model, &seed_model).unwrap();
    let train_pages = (1..100)
        .filter(|&p| !corpus::validation_page(p))
        .take(2)
        .collect::<Vec<_>>();
    let dev = (1..100).find(|&p| corpus::validation_page(p)).unwrap();
    let input = temp.0.join("input.jsonl.bz2");
    let mut writer = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
    for (page, two) in [(train_pages[0], false), (train_pages[0], false)]
        .into_iter()
        .chain((0..6).map(|i| (train_pages[1], i == hard)))
        .chain([(dev, false)])
    {
        let mut tokens = vec![
            serde_json::json!({"surface":"阿","feature":"名詞,一般,*,*,*,*,阿,ア,ア","reading":"ア","unknown":false}),
        ];
        if two {
            tokens.push(serde_json::json!({"surface":"伊","feature":"名詞,一般,*,*,*,*,伊,イ,イ","reading":"イ","unknown":false}));
        }
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({"page_id":page,"title":"再開試験",
            "sentence":if two {"阿伊"} else {"阿"},"tokens":tokens}),
        )
        .unwrap();
        writeln!(writer).unwrap();
    }
    std::fs::write(&input, writer.finish().unwrap()).unwrap();
    let invoke = |out: &std::path::Path, max_nodes: &str, resume: Option<&std::path::Path>| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ai_engine_crf"));
        command
            .arg("--input")
            .arg(&input)
            .arg("--output")
            .arg(out)
            .args([
                "--epochs",
                "1",
                "--threads",
                "8",
                "--batch-size",
                "1",
                "--shuffle-buffer",
                "4",
                "--page-block-size",
                "1",
                "--nbest",
                "8",
                "--max-sentences",
                "8",
                "--max-validation-sentences",
                "1",
                "--max-eval-sentences",
                "1",
                "--max-nodes",
                max_nodes,
                "--seed",
            ])
            .arg(seed.to_string());
        if let Some(path) = resume {
            command.arg("--resume-from").arg(path);
        } else {
            command.arg("--model-in").arg(&seed_model);
        }
        command.output().unwrap()
    };
    let uninterrupted = temp.0.join("uninterrupted");
    let completed = invoke(&uninterrupted, "100", None);
    assert!(
        completed.status.success(),
        "{}",
        String::from_utf8_lossy(&completed.stderr)
    );
    let stopped = temp.0.join("stopped");
    let failed = invoke(&stopped, "2", None);
    assert!(!failed.status.success());
    let stop: serde_json::Value =
        serde_json::from_slice(&std::fs::read(stopped.join("training-stop.json")).unwrap())
            .unwrap();
    assert_eq!(stop["pending_window"].as_array().unwrap().len(), 3);
    let partial: serde_json::Value =
        serde_json::from_slice(&std::fs::read(stopped.join("partial-training.json")).unwrap())
            .unwrap();
    assert_eq!(partial["training_sentences_processed"], 3);
    let resumed = temp.0.join("resumed");
    let result = invoke(&resumed, "100", Some(&stopped));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let resumed_model = output::load(&resumed.join("model.json")).unwrap();
    let uninterrupted_model = output::load(&uninterrupted.join("model.json")).unwrap();
    // Sparse checkpoint record order is intentionally unspecified so large
    // models can stream without a second sorted copy of the parameter table.
    assert_eq!(
        output::fingerprint(&resumed_model),
        output::fingerprint(&uninterrupted_model)
    );
    let metrics: serde_json::Value =
        serde_json::from_slice(&std::fs::read(resumed.join("metrics.json")).unwrap()).unwrap();
    assert_eq!(metrics[1]["training_sentences_processed"], 8);
    assert_eq!(metrics[1]["training_batches"], 8);
    assert_eq!(metrics[1]["training_preupdate_top1_evaluated_sentences"], 8);
    assert!(
        metrics[1]["training_preupdate_top1_gold_hits"]
            .as_u64()
            .unwrap()
            <= 8
    );
    assert!(metrics[1]["training_preupdate_top1_gold_hit_rate"].is_number());
    assert_eq!(
        metrics[1]["nbest_gold_hits"],
        8 - metrics[1]["training_gold_missing_nbest"].as_u64().unwrap()
    );
    assert!(metrics[1]["training_nbest_gold_hit_rate"].is_number());
    let curve = std::fs::read_to_string(resumed.join("training-curve.jsonl")).unwrap();
    let last_curve: serde_json::Value =
        serde_json::from_str(curve.lines().last().unwrap()).unwrap();
    assert!(last_curve["training_preupdate_top1_gold_hit_rate"].is_number());
    assert!(last_curve["training_nbest_gold_hit_rate"].is_number());
    assert!(String::from_utf8_lossy(&result.stderr).contains("train_preupdate_top1="));
    assert!(String::from_utf8_lossy(&result.stderr).contains("train_nbest_hit="));
    assert!(!resumed.join("page-blocks").exists()); // Reuse, do not rebuild/re-normalize.
    // A changed pending record must be detected before any additional update.
    let mut corrupt = stop;
    corrupt["pending_window"][0]["sentence"]["reading"] = "誤改変".into();
    std::fs::write(
        stopped.join("training-stop.json"),
        serde_json::to_vec(&corrupt).unwrap(),
    )
    .unwrap();
    let rejected_path = temp.0.join("rejected");
    let rejected = invoke(&rejected_path, "100", Some(&stopped));
    assert!(!rejected.status.success());
    let old = output::load(&stopped.join("stopped-model.json")).unwrap();
    let unchanged = output::load(&rejected_path.join("emergency-model.json")).unwrap();
    assert_eq!(output::fingerprint(&old), output::fingerprint(&unchanged));
}

fn assert_streaming_metrics(directory: &std::path::Path) {
    let metrics: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(directory.join("metrics.json")).unwrap())
            .unwrap();
    assert_eq!(metrics.as_array().unwrap().len(), 3);
    for epoch in 1..=2 {
        assert_eq!(metrics[epoch]["training_sentences_processed"], 6);
        assert_eq!(metrics[epoch]["training_batches"], 3);
        assert_eq!(metrics[epoch]["train"]["sentences"], 1);
        let mut order = metrics[epoch]["page_block_order"]
            .as_array()
            .unwrap()
            .clone();
        order.sort_by_key(|v| v.as_u64().unwrap());
        assert_eq!(order, vec![0, 1, 2, 3]);
        assert!(
            directory
                .join(format!("checkpoint-epoch-{epoch:04}.json"))
                .is_file()
        );
    }
    let input_report: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(directory.join("input-report.json")).unwrap())
            .unwrap();
    assert_eq!(input_report["training_sentences"], 6);
    assert!(
        metrics[2]["train"]["mean_first_nll"].as_f64().unwrap()
            < metrics[0]["train"]["mean_first_nll"].as_f64().unwrap()
    );
}

#[test]
#[allow(clippy::too_many_lines)] // Verifies the complete CLI lifecycle in one isolated directory.
fn compressed_corpus_train_export_reload_and_overwrite_guard() {
    let directory = std::env::temp_dir().join(format!(
        "ai-engine-crf-cli-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let temp = Temp(directory);
    let seeds = TrainingSeeds {
        pos_names: vec!["[BOS]".into(), "[EOS]".into(), "名詞,一般,*,*,*,*,*".into()],
        pos_bigrams: vec![(1, 3, 0.0), (3, 3, 0.0), (3, 2, 0.0)],
        words: [("阿", "あ"), ("あ", "あ"), ("伊", "い"), ("愛", "あい")]
            .into_iter()
            .map(|(s, r)| TrainingWord {
                surface: s.into(),
                reading: r.into(),
                pos_id: 3,
                log_unigram: -1.0,
                model_only: false,
            })
            .collect(),
    };
    let model_path = temp.0.join("seed.json");
    output::save(&Model::from_seeds(seeds, 1e-8).unwrap(), &model_path).unwrap();
    let input_path = temp.0.join("input.jsonl.bz2");
    let mut compressor = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
    let train_pages = (1..100)
        .filter(|&p| !corpus::validation_page(p))
        .take(4)
        .collect::<Vec<_>>();
    let dev_page = (1..100).find(|&p| corpus::validation_page(p)).unwrap();
    let second_dev_page = (1..100)
        .filter(|&p| corpus::validation_page(p))
        .nth(1)
        .unwrap();
    for page in [
        train_pages[0],
        train_pages[0],
        train_pages[1],
        train_pages[1],
        train_pages[2],
        train_pages[2],
        dev_page,
        dev_page,
        train_pages[3],
        train_pages[3],
        second_dev_page,
        second_dev_page,
    ] {
        let record = serde_json::json!({"page_id":page,"title":"試験","sentence":"阿伊","tokens":[
            {"surface":"阿","feature":"名詞,一般,*,*,*,*,阿,ア,ア","reading":"ア","unknown":false},
            {"surface":"伊","feature":"名詞,一般,*,*,*,*,伊,イ,イ","reading":"イ","unknown":false}]});
        serde_json::to_writer(&mut compressor, &record).unwrap();
        writeln!(compressor).unwrap();
    }
    std::fs::write(&input_path, compressor.finish().unwrap()).unwrap();
    let run = |model: &std::path::Path, out: &std::path::Path, epochs: &str, objective: &str| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_ai_engine_crf"));
        command
            .args(["--input"])
            .arg(&input_path)
            .args(["--model-in"])
            .arg(model)
            .args(["--output"])
            .arg(out)
            .args([
                "--epochs",
                epochs,
                "--max-sentences",
                "6",
                "--max-validation-sentences",
                "2",
                "--nbest",
                "8",
                "--shuffle-buffer",
                "2",
                "--max-eval-sentences",
                "1",
                "--checkpoint-every",
                "1",
                "--threads",
                "4",
                "--batch-size",
                "2",
                "--page-block-size",
                "1",
                "--save-initial-model",
            ]);
        if objective == "margin" {
            command.args(["--primary-nbest-margin", "0.0"]);
        } else if objective == "listwise" {
            command.arg("--primary-nbest-listwise");
        }
        command.output().unwrap()
    };
    let trained_path = temp.0.join("trained");
    let result = run(&model_path, &trained_path, "2", "none");
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_streaming_metrics(&trained_path);
    let margin_path = temp.0.join("margin");
    let margin_result = run(&model_path, &margin_path, "1", "margin");
    assert!(
        margin_result.status.success(),
        "{}",
        String::from_utf8_lossy(&margin_result.stderr)
    );
    let margin_metrics: serde_json::Value =
        serde_json::from_slice(&std::fs::read(margin_path.join("metrics.json")).unwrap()).unwrap();
    assert_eq!(
        margin_metrics[1]["mean_update_first_nll"],
        serde_json::Value::Null
    );
    assert!(margin_metrics[1]["mean_update_primary_nbest_margin_loss"].is_number());
    assert!(
        margin_metrics[1]["primary_nbest_margin_active_sentences"]
            .as_u64()
            .unwrap()
            <= margin_metrics[1]["training_sentences_processed"]
                .as_u64()
                .unwrap()
    );
    let update_policy: serde_json::Value = serde_json::from_slice(
        &std::fs::read(margin_path.join("training-update-policy.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        update_policy["primary_objective"],
        "raw-nbest-all-pairwise-hinge"
    );
    assert_eq!(update_policy["primary_nbest_margin_log_score"], 0.0);
    let listwise_path = temp.0.join("listwise");
    let listwise_result = run(&model_path, &listwise_path, "1", "listwise");
    assert!(
        listwise_result.status.success(),
        "{}",
        String::from_utf8_lossy(&listwise_result.stderr)
    );
    let listwise_metrics: serde_json::Value =
        serde_json::from_slice(&std::fs::read(listwise_path.join("metrics.json")).unwrap())
            .unwrap();
    assert_eq!(
        listwise_metrics[1]["mean_update_first_nll"],
        serde_json::Value::Null
    );
    assert!(listwise_metrics[1]["mean_update_primary_nbest_listwise_loss"].is_number());
    assert!(listwise_metrics[1]["primary_nbest_listwise_active_sentences"].is_number());
    let listwise_policy: serde_json::Value = serde_json::from_slice(
        &std::fs::read(listwise_path.join("training-update-policy.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(listwise_policy["primary_objective"], "raw-nbest-listwise");
    assert!(
        listwise_policy["primary_objective_definition"]
            .as_str()
            .unwrap()
            .contains("expected candidate feature counts minus gold feature counts")
    );
    let hogwild_path = temp.0.join("hogwild");
    let hogwild_result = Command::new(env!("CARGO_BIN_EXE_ai_engine_crf"))
        .arg("--input")
        .arg(&input_path)
        .arg("--model-in")
        .arg(&model_path)
        .arg("--output")
        .arg(&hogwild_path)
        .args([
            "--hogwild",
            "--epochs",
            "1",
            "--max-sentences",
            "6",
            "--max-validation-sentences",
            "2",
            "--nbest",
            "8",
            "--shuffle-buffer",
            "4",
            "--max-eval-sentences",
            "1",
            "--threads",
            "4",
            "--batch-size",
            "2",
            "--page-block-size",
            "0",
            "--gold-path-augmentation",
        ])
        .output()
        .unwrap();
    assert!(
        hogwild_result.status.success(),
        "{}",
        String::from_utf8_lossy(&hogwild_result.stderr)
    );
    let hogwild_metrics: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(hogwild_path.join("metrics.json")).unwrap())
            .unwrap();
    assert_eq!(hogwild_metrics[1]["training_sentences_processed"], 6);
    assert_eq!(
        hogwild_metrics[1]["training_preupdate_top1_evaluated_sentences"],
        6
    );
    assert!(hogwild_metrics[1]["training_nbest_gold_hit_rate"].is_number());
    let hogwild_stderr = String::from_utf8_lossy(&hogwild_result.stderr);
    assert!(hogwild_stderr.contains("online_top1="));
    assert!(hogwild_stderr.contains("online_nbest_hit="));
    assert!(hogwild_stderr.matches("hogwild_window=").count() >= 2);
    assert!(hogwild_stderr.contains("--max-parameter-mib is not enforced"));
    let hogwild_verification: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(hogwild_path.join("verification.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        hogwild_verification["execution_mode"],
        "hogwild-scc-per-sentence"
    );
    assert_eq!(hogwild_verification["parameter_budget_enforced"], false);
    assert!(
        hogwild_verification["gradient_reduction"]
            .as_str()
            .unwrap()
            .contains("no batch mean")
    );
    assert!(
        !output::load(&hogwild_path.join("model.json"))
            .unwrap()
            .third
            .values
            .is_empty()
    );
    let initial = output::load(&trained_path.join("initial-model.json")).unwrap();
    assert!(initial.first.values.is_empty());
    assert!(initial.third.values.is_empty());
    let learned = output::load(&trained_path.join("model.json")).unwrap();
    assert!(!learned.first.values.is_empty());
    assert!(!learned.third.values.is_empty()); // N-best word-trigram keys learn deltas.
    assert!(!learned.third_pos.values.is_empty());
    assert!(!learned.third_reading_length.values.is_empty());
    let old_checkpoint = std::fs::read(trained_path.join("model.json")).unwrap();
    let refused = run(&model_path, &trained_path, "1", "none");
    assert!(!refused.status.success());
    assert_eq!(
        std::fs::read(trained_path.join("model.json")).unwrap(),
        old_checkpoint
    );
    let evaluation_path = temp.0.join("evaluation-only");
    let evaluated = run(
        &trained_path.join("model.json"),
        &evaluation_path,
        "0",
        "none",
    );
    assert!(
        evaluated.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let reloaded = output::load(&evaluation_path.join("model.json")).unwrap();
    assert_eq!(reloaded.first.values, learned.first.values);
    assert_eq!(reloaded.third.values, learned.third.values);
    let full_path = temp.0.join("full-corpus");
    let full = Command::new(env!("CARGO_BIN_EXE_ai_engine_crf"))
        .arg("--input")
        .arg(&input_path)
        .arg("--model-in")
        .arg(&model_path)
        .arg("--output")
        .arg(&full_path)
        .args([
            "--full-corpus",
            "--epochs",
            "1",
            "--threads",
            "8",
            "--batch-size",
            "2",
            "--page-block-size",
            "1",
            "--shuffle-buffer",
            "2",
            "--nbest",
            "8",
            "--max-sentences",
            "1",
            "--max-records",
            "1",
            "--max-eval-sentences",
            "1",
            "--max-validation-sentences",
            "2",
        ])
        .output()
        .unwrap();
    assert!(
        full.status.success(),
        "{}",
        String::from_utf8_lossy(&full.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(full_path.join("input-report.json")).unwrap())
            .unwrap();
    assert_eq!(report["records_read"], 12);
    assert_eq!(report["training_sentences"], 8);
    assert_eq!(report["heldout_sentences"], 4);
    assert_eq!(report["validation_sentences"], 2);
    assert_eq!(report["source_training_pages"], 4);
    assert_eq!(report["source_heldout_pages"], 2);
    assert_eq!(report["stopped_at_limit"], false);
    let split: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(full_path.join("corpus-split.json")).unwrap())
            .unwrap();
    assert_eq!(split["page_id_disjoint"], true);
    let mut reader = corpus::open(&full_path.join("test-corpus.normalized.jsonl.bz2")).unwrap();
    let mut line = String::new();
    let mut heldout = Vec::new();
    while let Some(entry) = ai_engine_crf::blocks::read_cached(&mut *reader, &mut line).unwrap() {
        assert!(corpus::validation_page(entry.sentence.page_id));
        heldout.push(entry.record);
    }
    assert_eq!(heldout, vec![7, 8, 11, 12]);
    let full_metrics: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(full_path.join("metrics.json")).unwrap())
            .unwrap();
    assert_eq!(full_metrics[1]["training_sentences_processed"], 8);
    assert_eq!(full_metrics[1]["training_batches"], 4);
    // A later oversized lattice must preserve the earlier committed batch.
    let fail_input = temp.0.join("lattice-limit.jsonl");
    let mut fail_writer = std::fs::File::create(&fail_input).unwrap();
    for (page, two_words) in [
        (train_pages[0], false),
        (train_pages[1], true),
        (dev_page, false),
    ] {
        let mut tokens = vec![
            serde_json::json!({"surface":"阿","feature":"名詞,一般,*,*,*,*,阿,ア,ア","reading":"ア","unknown":false}),
        ];
        if two_words {
            tokens.push(serde_json::json!({"surface":"伊","feature":"名詞,一般,*,*,*,*,伊,イ,イ","reading":"イ","unknown":false}));
        }
        serde_json::to_writer(
            &mut fail_writer,
            &serde_json::json!({"page_id":page,"title":"停止試験",
            "sentence":if two_words {"阿伊"} else {"阿"},"tokens":tokens}),
        )
        .unwrap();
        writeln!(fail_writer).unwrap();
    }
    drop(fail_writer);
    let fail_path = temp.0.join("lattice-limit");
    let failed = Command::new(env!("CARGO_BIN_EXE_ai_engine_crf"))
        .arg("--input")
        .arg(&fail_input)
        .arg("--model-in")
        .arg(&model_path)
        .arg("--output")
        .arg(&fail_path)
        .args([
            "--epochs",
            "1",
            "--max-sentences",
            "2",
            "--max-validation-sentences",
            "1",
            "--threads",
            "8",
            "--batch-size",
            "1",
            "--shuffle-buffer",
            "1",
            "--page-block-size",
            "0",
            "--nbest",
            "8",
            "--max-nodes",
            "2",
            "--max-eval-sentences",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    let stopped = output::load(&fail_path.join("stopped-model.json")).unwrap();
    assert!(!stopped.first.values.is_empty());
    let stop: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(fail_path.join("training-stop.json")).unwrap())
            .unwrap();
    assert_eq!(stop["kind"], "lattice-error");
    let partial: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(fail_path.join("partial-training.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(partial["training_sentences_processed"], 1);
    assert_eq!(
        output::fingerprint(&stopped),
        output::fingerprint(&output::load(&fail_path.join("model.json")).unwrap())
    );
    let mut invalid: serde_json::Value = serde_json::from_slice(&old_checkpoint).unwrap();
    invalid["first_values"] = serde_json::json!([[{"Word":999_999},0.2]]);
    let invalid_path = temp.0.join("invalid.json");
    std::fs::write(&invalid_path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    assert!(output::load(&invalid_path).is_err());
}

#[test]
fn evaluation_sample_mode_measures_raw_nbest_without_injecting_gold() {
    let directory = std::env::temp_dir().join(format!(
        "ai-engine-crf-fixed-eval-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let temp = Temp(directory);
    let model = Model::from_seeds(
        TrainingSeeds {
            pos_names: vec!["[BOS]".into(), "[EOS]".into(), "名詞,一般,*,*,*,*,*".into()],
            pos_bigrams: vec![],
            words: [("阿", "あ"), ("あ", "あ")]
                .into_iter()
                .map(|(surface, reading)| TrainingWord {
                    surface: surface.into(),
                    reading: reading.into(),
                    pos_id: 3,
                    log_unigram: -1.0,
                    model_only: false,
                })
                .collect(),
        },
        1e-8,
    )
    .unwrap();
    let word = u32::try_from(
        model
            .words
            .iter()
            .position(|candidate| candidate.surface == "阿")
            .unwrap(),
    )
    .unwrap();
    let model_path = temp.0.join("initial-model.json");
    output::save(&model, &model_path).unwrap();
    let sample_path = temp.0.join("fixed-sample.json");
    std::fs::write(
        &sample_path,
        serde_json::to_vec(&vec![corpus::Sentence {
            page_id: 1,
            title: "評価".into(),
            sentence: "阿".into(),
            reading: "あ".into(),
            spans: vec![corpus::Span {
                start: 0,
                end: 3,
                word,
                script: 4,
                forced: false,
            }],
        }])
        .unwrap(),
    )
    .unwrap();
    let input_path = temp.0.join("unused-input.jsonl");
    std::fs::write(
        &input_path,
        b"this path is required but fixed-sample mode must not read it\n",
    )
    .unwrap();
    let output_path = temp.0.join("eval");
    let result = Command::new(env!("CARGO_BIN_EXE_ai_engine_crf"))
        .arg("--input")
        .arg(&input_path)
        .arg("--model-in")
        .arg(&model_path)
        .arg("--evaluation-sample-in")
        .arg(&sample_path)
        .arg("--output")
        .arg(&output_path)
        .args(["--epochs", "0", "--nbest", "32", "--gold-path-augmentation"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let evaluation: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output_path.join("evaluation.json")).unwrap())
            .unwrap();
    assert_eq!(evaluation["gold_path_augmentation_applied"], false);
    assert_eq!(evaluation["metrics"]["nbest"], 32);
    assert_eq!(evaluation["metrics"]["sentences"], 1);
    assert_eq!(evaluation["metrics"]["nbest_gold_recall"], 1.0);
    assert_eq!(evaluation["metrics"]["gold_present_sentences"], 1);
    assert_eq!(
        evaluation["metrics"]["first_exact_accuracy_given_gold_present"],
        1.0
    );
    assert_eq!(
        evaluation["metrics"]["reranked_exact_accuracy_given_gold_present"],
        1.0
    );
    assert_eq!(evaluation["metrics"]["rerank_top1_helped_sentences"], 0);
    assert_eq!(evaluation["metrics"]["rerank_top1_harmed_sentences"], 0);
}
