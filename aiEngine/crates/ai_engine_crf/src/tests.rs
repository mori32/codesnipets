use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use ai_engine_dict::{TrainingSeeds, TrainingWord};

use super::*;
use crate::corpus::{Index, Limits, Record, Token};

fn close(a: f64, b: f64, tolerance: f64) {
    assert!((a - b).abs() <= tolerance, "{a:.15} != {b:.15}");
}

#[test]
fn static_word_bigrams_are_log_probability_mixture_offsets_and_keep_both_signs() {
    let seeds = TrainingSeeds {
        pos_names: vec!["[BOS]".into(), "[EOS]".into(), "名詞,一般,*,*,*,*,*".into()],
        pos_bigrams: vec![(3, 3, 0.5_f64.ln())],
        words: (0..4)
            .map(|n| TrainingWord {
                surface: format!("語{n}"),
                reading: format!("ご{n}"),
                pos_id: 3,
                model_only: false,
                log_unigram: 0.25_f64.ln(),
            })
            .collect(),
    };
    let model = Model::from_seeds_with_bigrams(
        seeds,
        vec![
            ai_engine_dict::TrainingBigram {
                source: 0,
                target: 2,
                count: 50,
                outgoing_count: 100,
            },
            ai_engine_dict::TrainingBigram {
                source: 1,
                target: 3,
                count: 1,
                outgoing_count: 100,
            },
        ],
        1e-8,
        0.9,
    )
    .unwrap();
    let base: f64 = 0.5 * 0.25;
    let positive_mix = 0.9 * base + 0.1 * 0.5;
    let negative_mix = 0.9 * base + 0.1 * 0.01;
    close(
        model.initial(Feature::Bigram(3, 5)),
        (positive_mix / base).ln(),
        1e-12,
    );
    close(
        model.initial(Feature::Bigram(4, 6)),
        (negative_mix / base).ln(),
        1e-12,
    );
    assert!(model.initial(Feature::Bigram(3, 5)) > 0.0);
    assert!(model.initial(Feature::Bigram(4, 6)) < 0.0);
    assert!(model.initial_bigrams.get(&Feature::Bigram(3, 5)).is_some());
    assert!(model.third.values.is_empty());
}

#[test]
fn static_word_trigrams_are_log_interpolation_offsets_and_missing_triples_stay_zero() {
    let seeds = TrainingSeeds {
        pos_names: vec!["[BOS]".into(), "[EOS]".into(), "名詞,一般,*,*,*,*,*".into()],
        pos_bigrams: vec![(3, 3, 0.5_f64.ln())],
        words: (0..3)
            .map(|n| TrainingWord {
                surface: format!("語{n}"),
                reading: format!("ご{n}"),
                pos_id: 3,
                model_only: false,
                log_unigram: 0.25_f64.ln(),
            })
            .collect(),
    };
    let model = Model::from_seeds_with_ngrams(
        seeds,
        Vec::new(),
        vec![ai_engine_dict::TrainingTrigram {
            first: 0,
            middle: 1,
            target: 2,
            count: 20,
            parent_count: 100,
        }],
        1e-8,
        0.9,
        0.1,
    )
    .unwrap();
    let p2: f64 = 0.5 * 0.25;
    let p3: f64 = 20.0 / 100.0;
    close(
        model.trigram_weight([3, 4, 5]),
        ((0.9 * p2 + 0.1 * p3) / p2).ln(),
        1e-12,
    );
    close(model.trigram_weight([3, 4, 3]), 0.0, 0.0);
    assert_eq!(model.initial_trigrams.len(), 1);
    assert!(model.third.values.is_empty());
}

#[test]
fn word_trigram_key_delta_learns_independently_and_rejects_nonfinite_updates() {
    let (mut model, _) = fixture();
    let key = [BOS, 3, EOS];
    model.initial_trigrams.insert(key, 0.5);
    let before = model.trigram_weight(key);
    let mut gradient = TriGradient::new();
    gradient.insert(key, -2.0);
    model.third.update(&gradient, 0.03, 0.1).unwrap();
    assert!(model.trigram_weight(key) > before);
    assert_eq!(model.third.values.len(), 1);
    gradient.insert(key, f64::INFINITY);
    assert!(model.third.update(&gradient, 0.03, 0.1).is_err());
}

fn fixture() -> (Model, Lattice) {
    let pos = vec![
        "[BOS]",
        "[EOS]",
        "名詞,一般,*,*,*,*,*",
        "未知語",
        "名詞,数,*,*,*,*,*",
        "英単語",
        "記号,一般,*,*,*,*,*",
        "助詞,格助詞,一般,*,*,*,を",
    ];
    let words = [
        ("阿", "あ", 3, false),
        ("あ", "あ", 3, false),
        ("伊", "い", 3, false),
        ("愛", "あい", 3, false),
        ("い", "い", 3, false),
        ("", "", 4, true),
        ("", "", 5, true),
        ("", "", 6, true),
        ("。", "。", 7, false),
        ("！", "！", 7, false),
        ("を", "を", 8, false),
    ]
    .into_iter()
    .map(|(surface, reading, pos_id, model_only)| TrainingWord {
        surface: surface.to_owned(),
        reading: reading.to_owned(),
        pos_id,
        model_only,
        log_unigram: -0.2,
    })
    .collect();
    let seeds = TrainingSeeds {
        words,
        pos_names: pos.into_iter().map(str::to_owned).collect(),
        pos_bigrams: vec![],
    };
    let mut model = Model::from_seeds(seeds, 0.5).unwrap();
    model.initial_pos.fill(0.0);
    let nodes = vec![
        Node {
            start: 0,
            end: 0,
            word: BOS,
            script: 0,
        },
        Node {
            start: 0,
            end: 1,
            word: 3,
            script: 4,
        },
        Node {
            start: 0,
            end: 1,
            word: 4,
            script: 2,
        },
        Node {
            start: 0,
            end: 2,
            word: 6,
            script: 4,
        },
        Node {
            start: 1,
            end: 2,
            word: 5,
            script: 4,
        },
        Node {
            start: 1,
            end: 2,
            word: 7,
            script: 2,
        },
        Node {
            start: 2,
            end: 2,
            word: EOS,
            script: 1,
        },
    ];
    let graph = Lattice::new(nodes, vec![0, 1, 4, 6], 100).unwrap();
    (model, graph)
}

fn enumerate(model: &Model, graph: &Lattice) -> Vec<Path> {
    fn walk(model: &Model, graph: &Lattice, path: Vec<usize>, output: &mut Vec<Path>) {
        let a = *path.last().unwrap();
        if a == graph.nodes.len() - 1 {
            output.push(Path {
                score: model.path_score(graph, &path),
                nodes: path,
            });
            return;
        }
        for &e in &graph.outgoing[a] {
            let mut next = path.clone();
            next.push(graph.edges[e].to);
            walk(model, graph, next, output);
        }
    }
    let mut results = vec![];
    walk(model, graph, vec![0], &mut results);
    results.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.nodes.cmp(&b.nodes))
    });
    results
}

#[test]
fn forward_backward_matches_exhaustive_paths_and_flow() {
    let (mut model, graph) = fixture();
    model.first.values.insert(Feature::ScriptBigram(4, 2), 0.37);
    let all = enumerate(&model, &graph);
    assert_eq!(all.len(), 5);
    let exact = all
        .iter()
        .fold(f64::NEG_INFINITY, |z, p| log_add(z, p.score));
    let result = first_order(&model, &graph).unwrap();
    close(result.log_z, exact, 1e-12);
    for (edge_id, edge) in graph.edges.iter().enumerate() {
        let expected: f64 = all
            .iter()
            .filter(|p| p.nodes.windows(2).any(|pair| pair == [edge.from, edge.to]))
            .map(|p| (p.score - exact).exp())
            .sum();
        close(result.edge_marginals[edge_id], expected, 1e-12);
    }
    for node in 1..graph.nodes.len() - 1 {
        let incoming: f64 = graph.incoming[node]
            .iter()
            .map(|&e| result.edge_marginals[e])
            .sum();
        let outgoing: f64 = graph.outgoing[node]
            .iter()
            .map(|&e| result.edge_marginals[e])
            .sum();
        close(incoming, outgoing, 1e-12);
    }
    close(
        graph.outgoing[0]
            .iter()
            .map(|&e| result.edge_marginals[e])
            .sum(),
        1.0,
        1e-12,
    );
}

#[test]
fn first_gradient_all_four_families_matches_finite_difference() {
    let (model, graph) = fixture();
    let analytic = first_order(&model, &graph).unwrap().gradient;
    for (&key, &derivative) in &analytic {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.first.values.insert(key, 1e-5);
        minus.first.values.insert(key, -1e-5);
        let numeric = (first_order(&plus, &graph).unwrap().loss
            - first_order(&minus, &graph).unwrap().loss)
            / 2e-5;
        close(numeric, derivative, 1e-9);
    }
    assert!(
        analytic
            .iter()
            .any(|(f, g)| matches!(f, Feature::Bigram(..)) && *g < 0.0)
    );
    assert!(
        analytic
            .iter()
            .any(|(f, g)| matches!(f, Feature::Bigram(..)) && *g > 0.0)
    );
    for kind in 0..6 {
        assert!(analytic.keys().any(|f| match f {
            Feature::Word(_) => kind == 0,
            Feature::Pos(..) => kind == 1,
            Feature::Bigram(..) => kind == 2,
            Feature::PosWord(..) => kind == 3,
            Feature::WordPos(..) => kind == 4,
            Feature::Script(_) | Feature::ScriptBigram(..) => false,
            Feature::ReadingLengthBigram(..) => kind == 5,
        }));
    }
}

#[test]
fn astar_nbest_matches_exhaustive_viterbi_with_variable_segmentation() {
    let (mut model, graph) = fixture();
    model.first.values.insert(Feature::Word(3), 0.323);
    model.first.values.insert(Feature::Bigram(4, 7), -0.213);
    model.first.values.insert(Feature::Script(2), 0.07);
    let expected = enumerate(&model, &graph);
    let actual = nbest(&model, &graph, 100, 100).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(&expected) {
        assert_eq!(a.nodes, b.nodes);
        close(a.score, b.score, 1e-12);
    }
    assert!(nbest(&model, &graph, 2, 0).is_err());
}

#[test]
fn trigram_gradient_matches_finite_difference_with_bos_eos() {
    let (model, graph) = fixture();
    let paths = enumerate(&model, &graph);
    let result = rerank(&model, &graph, &paths).unwrap();
    for (&key, &derivative) in &result.gradient {
        let mut plus = model.clone();
        let mut minus = model.clone();
        plus.third.values.insert(key, 1e-5);
        minus.third.values.insert(key, -1e-5);
        let numeric = (rerank(&plus, &graph, &paths).unwrap().loss.unwrap()
            - rerank(&minus, &graph, &paths).unwrap().loss.unwrap())
            / 2e-5;
        close(numeric, derivative, 1e-9);
    }
    assert!(result.gradient.contains_key(&[BOS, 3, 5]));
    assert!(result.gradient.contains_key(&[3, 5, EOS]));
}

#[test]
fn no_gold_injection_or_duplicate_paths_and_cached_score_not_trusted() {
    let (model, graph) = fixture();
    let mut paths = nbest(&model, &graph, 1, 100).unwrap();
    assert_ne!(paths[0].nodes, graph.gold);
    let result = rerank(&model, &graph, &paths).unwrap();
    assert!(result.loss.is_none());
    assert!(result.gradient.is_empty());
    paths[0].score = f64::NAN;
    assert!(rerank(&model, &graph, &paths).unwrap().scores[0].is_finite());
    paths.push(paths[0].clone());
    assert!(rerank(&model, &graph, &paths).is_err());
}

#[test]
fn online_update_reduces_both_losses_and_learns_positive_and_negative_pairs() {
    let (mut model, graph) = fixture();
    model.initial_trigrams.insert([BOS, 3, 5], 0.5);
    model.initial_trigrams.insert([3, 5, EOS], 0.5);
    assert!(model.first.values.is_empty() && model.third.values.is_empty());
    close(model.weight(Feature::Bigram(3, 5)), 0.0, 0.0);
    let report = train_sentence(&mut model, &graph, 0.03, 0.0, 100, 100).unwrap();
    assert!(report.first_loss_after < report.first_loss_before);
    assert!(report.rerank_loss_after.unwrap() < report.rerank_loss_before.unwrap());
    assert!(model.first.get(Feature::Bigram(3, 5)) > 0.0);
    assert!(model.first.get(Feature::Bigram(4, 7)) < 0.0);
    assert!(!model.third.values.is_empty());
    assert!(!model.third_pos.values.is_empty());
}

#[test]
fn nbest_is_regenerated_after_first_order_update() {
    let (mut model, graph) = fixture();
    assert_ne!(nbest(&model, &graph, 1, 100).unwrap()[0].nodes, graph.gold);
    let report = train_sentence(&mut model, &graph, 1.0, 0.0, 1, 100).unwrap();
    assert!(report.gold_in_nbest);
    assert_eq!(nbest(&model, &graph, 1, 100).unwrap()[0].nodes, graph.gold);
    assert!(model.third.values.is_empty()); // One candidate: no trigram discrimination.
}

#[test]
fn global_l2_decays_untouched_parameters_toward_initial_not_zero_total_cost() {
    let (mut model, _) = fixture();
    let key = Feature::Word(3);
    model.first.values.insert(key, 2.0);
    model.first.update(&BTreeMap::new(), 0.2, 0.5).unwrap();
    close(model.first.get(key), 1.8, 1e-12);
    close(model.weight(key), model.initial(key) + 1.8, 1e-12);
    assert!(model.first.update(&BTreeMap::new(), 2.0, 0.5).is_err());
}

#[test]
fn stable_log_space_and_repeated_trigram_counts() {
    let (mut model, graph) = fixture();
    for word in &mut model.words {
        word.initial = -1000.0;
    }
    model.initial_pos.fill(-1000.0);
    let result = first_order(&model, &graph).unwrap();
    assert!(result.loss.is_finite());
    assert!(result.gradient.values().all(|g| g.is_finite()));
    assert_eq!(
        Model::trigram_features(&graph, &[1, 1, 1, 1]).get(&[3, 3, 3]),
        Some(&2.0)
    );
    close(log_add(-10000.0, -10000.0), -10000.0 + 2.0_f64.ln(), 1e-12);
}

#[test]
fn trigram_learns_context_that_shared_last_bigram_cannot_discriminate() {
    let (mut model, _) = fixture();
    model.words.iter_mut().for_each(|w| w.initial = 0.0);
    let build = |context: u32, correct: usize| {
        Lattice::new(
            vec![
                Node {
                    start: 0,
                    end: 0,
                    word: BOS,
                    script: 0,
                },
                Node {
                    start: 0,
                    end: 1,
                    word: context,
                    script: 4,
                },
                Node {
                    start: 1,
                    end: 2,
                    word: 13,
                    script: 2,
                },
                Node {
                    start: 2,
                    end: 3,
                    word: 5,
                    script: 4,
                },
                Node {
                    start: 2,
                    end: 3,
                    word: 6,
                    script: 4,
                },
                Node {
                    start: 3,
                    end: 3,
                    word: EOS,
                    script: 1,
                },
            ],
            vec![0, 1, 2, correct, 5],
            100,
        )
        .unwrap()
    };
    let garden = build(3, 3);
    let shop = build(4, 4);
    for _ in 0..100 {
        for graph in [&garden, &shop] {
            let paths = nbest(&model, graph, 2, 100).unwrap();
            let r = rerank(&model, graph, &paths).unwrap();
            model.third.update(&r.gradient, 0.2, 0.0).unwrap();
        }
    }
    assert!(model.first.values.is_empty());
    for graph in [&garden, &shop] {
        let paths = nbest(&model, graph, 2, 100).unwrap();
        assert_eq!(
            paths[rerank(&model, graph, &paths).unwrap().best].nodes,
            graph.gold
        );
    }
}

fn token(surface: &str, reading: Option<&str>, feature: &str, unknown: bool) -> Token {
    Token {
        surface: surface.to_owned(),
        feature: feature.to_owned(),
        reading: reading.map(str::to_owned),
        unknown,
        segment_index: 0,
    }
}

#[test]
fn corpus_builds_full_reading_lattice_not_just_same_boundary_homophones() {
    let (model, _) = fixture();
    let index = Index::new(&model).unwrap();
    let record = Record {
        page_id: 1,
        title: "題".into(),
        sentence: "阿伊".into(),
        tokens: vec![
            token("阿", Some("ア"), "名詞,一般,*,*,*,*,阿,ア,ア", false),
            token("伊", Some("イ"), "名詞,一般,*,*,*,*,伊,イ,イ", false),
        ],
    };
    let prepared = corpus::prepare(
        record,
        &model,
        &index,
        Limits {
            words: 48,
            nodes: 100,
            edges: 100,
        },
    )
    .unwrap();
    assert_eq!(prepared.reading, "あい");
    assert_eq!(nbest(&model, &prepared.graph, 100, 100).unwrap().len(), 5);
    assert!(
        prepared
            .graph
            .nodes
            .iter()
            .any(|n| n.word == 6 && n.start == 0 && n.end == 6)
    );
}

#[test]
fn normalization_collapses_unreviewed_unknowns_numbers_english_and_splits_symbols() {
    let (model, _) = fixture();
    let index = Index::new(&model).unwrap();
    let record = Record {
        page_id: 2,
        title: "試験".into(),
        sentence: "未知漢字123ABC。!を".into(),
        tokens: vec![
            token("未知漢字", None, "名詞,固有名詞,人名,一般,*,*,*", true),
            token("123", None, "名詞,数,*,*,*,*,*", true),
            token("ABC", None, "名詞,一般,*,*,*,*,*", true),
            token("。!", None, "記号,一般,*,*,*,*,*", true),
            token("を", Some("ヲ"), "助詞,格助詞,一般,*,*,*,を,ヲ,ヲ", false),
        ],
    };
    let prepared = corpus::prepare(
        record,
        &model,
        &index,
        Limits {
            words: 48,
            nodes: 100,
            edges: 100,
        },
    )
    .unwrap();
    assert_eq!(prepared.forced_tokens, 3);
    let ids = prepared
        .graph
        .gold
        .iter()
        .map(|&n| prepared.graph.nodes[n].word)
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![BOS, 8, 9, 10, 11, 12, 13, EOS]);
    assert!(prepared.reading.contains("１２３"));
    assert_eq!(
        model.pos_names[model.words[13].pos as usize],
        "助詞,格助詞,一般,*,*,*,を"
    );
}

#[test]
fn surface_categories_include_mixed_scripts_and_are_not_reading_categories() {
    for (surface, expected) in [
        ("かな", 2),
        ("カナー", 3),
        ("漢字", 4),
        ("。！", 5),
        ("ＡBC", 6),
        ("１２３", 7),
        ("買う", 8),
        ("AI技術", 8),
        ("𠮷", 4),
    ] {
        assert_eq!(script(surface), expected, "{surface}");
    }
}

#[test]
fn unknown_surface_script_does_not_score_but_reading_length_transition_is_trainable() {
    let (mut model, _) = fixture();
    let index = Index::new(&model).unwrap();
    let record = Record {
        page_id: 1,
        title: "未知語の表記".into(),
        sentence: "ミチ阿".into(),
        tokens: vec![
            Token {
                surface: "ミチ".into(),
                feature: "名詞,一般,*,*,*,*,*".into(), // This unchecked POS is not used.
                reading: None,
                unknown: true,
                segment_index: 0,
            },
            Token {
                surface: "阿".into(),
                feature: "名詞,一般,*,*,*,*,阿,ア,ア".into(),
                reading: Some("ア".into()),
                unknown: false,
                segment_index: 0,
            },
        ],
    };
    let prepared = corpus::prepare(
        record,
        &model,
        &index,
        Limits {
            words: 10,
            nodes: 100,
            edges: 1000,
        },
    )
    .unwrap();
    let unknown = &prepared.graph.nodes[prepared.graph.gold[1]];
    assert_eq!(unknown.word, 8);
    assert_eq!(unknown.script, 3); // Actual katakana, not generic word's empty surface.
    assert_eq!(model.words[unknown.word as usize].reading_length, 0);
    // The prepared unknown+阿 path has no competing reading-length bucket for
    // the known token. Build a two-way lattice so the length transition is
    // genuinely trainable, while the paths differ in surface script and length.
    let reading_graph = Lattice::new(
        vec![
            Node {
                start: 0,
                end: 0,
                word: BOS,
                script: 0,
            },
            Node {
                start: 0,
                end: 1,
                word: 3,
                script: 4,
            }, // 阿, length 1
            Node {
                start: 0,
                end: 1,
                word: 6,
                script: 2,
            }, // 愛, length 2
            Node {
                start: 1,
                end: 1,
                word: EOS,
                script: 1,
            },
        ],
        vec![0, 1, 3],
        20,
    )
    .unwrap();
    let before = first_order(&model, &reading_graph).unwrap();
    assert!(before.gradient[&Feature::ReadingLengthBigram(0, 1)] < -0.1);
    assert!(before.gradient[&Feature::ReadingLengthBigram(0, 2)] > 0.1);
    model
        .first
        .values
        .insert(Feature::ReadingLengthBigram(0, 1), 0.7);
    let after_07 = first_nll(&model, &reading_graph).unwrap();
    assert!(after_07 < before.loss);
    model
        .first
        .values
        .insert(Feature::ReadingLengthBigram(0, 1), 0.8);
    assert!(first_nll(&model, &reading_graph).unwrap() < after_07);
}

#[test]
fn json_record_reader_rejects_oversized_lines_without_unbounded_read() {
    let bytes = vec![b' '; (8 << 20) + 100];
    let mut reader = std::io::Cursor::new(bytes);
    let mut line = String::new();
    let error = corpus::read_record(&mut reader, &mut line, 1).unwrap_err();
    assert!(error.to_string().contains("8 MiB"));
    assert_eq!(line.len(), (8 << 20) + 1);
    assert_eq!(reader.position(), (8 << 20) + 1);
}

#[derive(Debug)]
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "ai-engine-crf-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn checkpoint_and_feature_families_roundtrip_without_quantization() {
    let temp = Temp::new();
    let (mut model, graph) = fixture();
    model.initial_bigrams.insert(Feature::Bigram(3, 4), -0.25);
    model.initial_trigrams.insert([BOS, 3, EOS], 0.125);
    train_sentence(&mut model, &graph, 0.03, 0.01, 100, 100).unwrap();
    output::export(&model, &temp.0).unwrap();
    let model_path = temp.0.join("model.json");
    let loaded = output::load(&model_path).unwrap();
    assert_eq!(model.first.values, loaded.first.values);
    assert_eq!(model.third.values, loaded.third.values);
    assert_eq!(model.initial_bigrams, loaded.initial_bigrams);
    assert_eq!(model.initial_trigrams, loaded.initial_trigrams);
    close(
        model.trigram_weight([BOS, 3, EOS]),
        loaded.trigram_weight([BOS, 3, EOS]),
        0.0,
    );
    let mut legacy_alpha: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&model_path).unwrap()).unwrap();
    assert!(legacy_alpha.get("word_trigram_alpha").is_none());
    legacy_alpha["word_trigram_alpha"] = serde_json::json!(1234.0);
    let legacy_path = temp.0.join("legacy-alpha.json");
    std::fs::write(&legacy_path, serde_json::to_vec(&legacy_alpha).unwrap()).unwrap();
    let ignores_legacy_alpha = output::load(&legacy_path).unwrap();
    assert_eq!(
        loaded.trigram_weight([BOS, 3, EOS]).to_bits(),
        ignores_legacy_alpha.trigram_weight([BOS, 3, EOS]).to_bits()
    );
    assert_eq!(
        first_order(&model, &graph).unwrap().loss.to_bits(),
        first_order(&loaded, &graph).unwrap().loss.to_bits()
    );
    assert!(output::save(&loaded, &temp.0.join("model.json")).is_err()); // Never overwrite.
    for name in [
        "word-cost.tsv",
        "pos-bigram.tsv",
        "pos-token.tsv",
        "affix-pos-selection.tsv",
        "word-bigram.tsv",
        "pos-word-bigram.tsv",
        "word-pos-bigram.tsv",
        "word-trigram.tsv",
        "reading-length-bigram.tsv",
        "reading-length-trigram.tsv",
    ] {
        assert!(std::fs::metadata(temp.0.join(name)).unwrap().len() > 20);
    }
}

#[test]
fn streamed_checkpoint_rejects_duplicate_and_out_of_range_keys() {
    let temp = Temp::new();
    let (mut model, graph) = fixture();
    train_sentence(&mut model, &graph, 0.03, 0.01, 256, 1000).unwrap();
    // Preserve a frozen legacy per-key word-trigram delta to exercise loader validation.
    model.third.values.insert([BOS, 3, EOS], 0.1);
    let original = temp.0.join("original.json");
    output::save(&model, &original).unwrap();
    let source: serde_json::Value =
        serde_json::from_slice(&std::fs::read(original).unwrap()).unwrap();
    for kind in 0..5 {
        let mut invalid = source.clone();
        match kind {
            0 => {
                let values = invalid["first_values"].as_array_mut().unwrap();
                values.push(values[0].clone());
            }
            1 => {
                let values = invalid["third_values"].as_array_mut().unwrap();
                values.push(values[0].clone());
            }
            2 => {
                invalid["first_values"][0] = serde_json::json!([
                    {"Bigram": [sparse::MAX_ID + 1, 3]}, 0.1,
                ]);
            }
            3 => {
                invalid["third_values"][0] =
                    serde_json::json!([[BOS, sparse::MAX_ID + 1, EOS], 0.1,]);
            }
            _ => {
                invalid["first_values"][0] = serde_json::json!([
                    {"Word": model.words.len() + 1}, 0.1,
                ]);
            }
        }
        let path = temp.0.join(format!("invalid-{kind}.json"));
        std::fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(
            output::load(&path).is_err(),
            "invalid checkpoint kind {kind}"
        );
    }
}

#[test]
fn compressed_reader_handles_concatenated_members_and_utf8() {
    let temp = Temp::new();
    let path = temp.0.join("input.jsonl.bz2");
    let mut bytes = vec![];
    for text in ["日本語\n", "次の文\n"] {
        let mut encoder = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::fast());
        encoder.write_all(text.as_bytes()).unwrap();
        bytes.extend(encoder.finish().unwrap());
    }
    std::fs::write(&path, bytes).unwrap();
    let mut text = String::new();
    corpus::open(&path)
        .unwrap()
        .read_to_string(&mut text)
        .unwrap();
    assert_eq!(text, "日本語\n次の文\n");
    let corrupt = temp.0.join("bad.bz2");
    std::fs::write(&corrupt, b"not bzip2").unwrap();
    assert!(
        corpus::open(&corrupt)
            .unwrap()
            .read_to_string(&mut String::new())
            .is_err()
    );
}

#[test]
fn page_split_and_shuffle_are_deterministic() {
    let mut a = (0..100).collect::<Vec<_>>();
    let mut b = a.clone();
    corpus::shuffle(&mut a, &mut 123);
    corpus::shuffle(&mut b, &mut 123);
    assert_eq!(a, b);
    assert!((0..100).any(corpus::validation_page));
    assert!((0..100).any(|p| !corpus::validation_page(p)));
}

#[test]
fn fast_online_path_is_bit_identical_to_diagnostic_update() {
    let (mut diagnostic, graph) = fixture();
    let mut online = diagnostic.clone();
    for _ in 0..5 {
        let measured = train_sentence(&mut diagnostic, &graph, 0.03, 0.01, 32, 1000).unwrap();
        let fast = train_online(&mut online, &graph, 0.03, 0.01, 32, 1000, 1 << 20).unwrap();
        assert_eq!(
            measured.first_loss_before.to_bits(),
            fast.first_loss.to_bits()
        );
        assert_eq!(diagnostic.first.values, online.first.values);
        assert_eq!(diagnostic.third.values, online.third.values);
        assert_eq!(
            diagnostic.first.scale.to_bits(),
            online.first.scale.to_bits()
        );
        assert_eq!(
            diagnostic.third.scale.to_bits(),
            online.third.scale.to_bits()
        );
        assert_eq!(
            first_nll(&online, &graph).unwrap().to_bits(),
            first_order(&online, &graph).unwrap().loss.to_bits()
        );
    }
}

#[test]
fn memory_limit_does_not_commit_partial_sentence_or_drop_features() {
    let (mut model, graph) = fixture();
    let initial = output::fingerprint(&model);
    let error = train_online(&mut model, &graph, 0.03, 0.01, 256, 1000, 1).unwrap_err();
    assert!(error.downcast_ref::<ParameterLimit>().is_some());
    assert_eq!(output::fingerprint(&model), initial);
    train_online(&mut model, &graph, 0.03, 0.01, 256, 1000, 1 << 20).unwrap();
    assert!(!model.first.values.is_empty());
    assert!(!model.third.values.is_empty());
    assert!(!model.third_pos.values.is_empty());
}

#[test]
fn packed_keys_do_not_collide_and_capacity_estimate_bounds_allocated_tables() {
    use sparse::{Key, MAX_ID, SparseMap};
    let keys = [
        Feature::Word(MAX_ID),
        Feature::Pos(MAX_ID, MAX_ID),
        Feature::Bigram(MAX_ID, MAX_ID),
        Feature::Script(9),
        Feature::ScriptBigram(9, 9),
    ];
    for key in keys {
        assert_eq!(Feature::unpack(key.pack()), key);
    }
    assert_eq!(
        keys.into_iter()
            .map(Key::pack)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        5
    );
    let mut table = SparseMap::<Feature>::default();
    for i in 0..1000 {
        let bound = table.projected_bytes(1);
        table.insert(Feature::Bigram(i, i + 1), f64::from(i));
        assert!(table.projected_bytes(0) <= bound);
        assert_eq!(
            table.get(&Feature::Bigram(i, i + 1)).unwrap().to_bits(),
            f64::from(i).to_bits()
        );
    }
}

#[test]
fn packed_feature_order_equals_reference_enum_order() {
    use sparse::Key;
    let mut keys = vec![];
    for a in [0, 1, 255, 65536, sparse::MAX_ID] {
        keys.push(Feature::Word(a));
        for b in [0, 1, 255, 65536, sparse::MAX_ID] {
            keys.push(Feature::Pos(a, b));
            keys.push(Feature::Bigram(a, b));
        }
    }
    for a in 0..10 {
        keys.push(Feature::Script(a));
        for b in 0..10 {
            keys.push(Feature::ScriptBigram(a, b));
        }
    }
    let mut packed = keys.iter().copied().map(Key::pack).collect::<Vec<_>>();
    packed.sort_unstable();
    keys.sort_unstable();
    assert_eq!(
        keys,
        packed.into_iter().map(Feature::unpack).collect::<Vec<_>>()
    );
}

#[test]
fn sorted_batch_merge_keeps_input_order_cancellation_and_signed_zero() {
    let keys = [Feature::Word(3), Feature::Word(4), Feature::Bigram(3, 5)];
    let values = [
        [1e16, -0.0, 0.0],
        [1.0, -0.0, -0.0],
        [-1e16, -0.0, 0.1],
        [0.3, -0.0, -0.2],
    ];
    let mut sum = Gradient::new();
    let mut gradients = Vec::new();
    for row in values {
        let gradient: Gradient = keys.into_iter().zip(row).collect();
        for (&key, &value) in &gradient {
            *sum.entry(key).or_default() += value;
        }
        gradients.push(optimized::Entries::from_reference(gradient));
    }
    gradients.push(optimized::Entries::from_reference(Gradient::new()));
    for value in sum.values_mut() {
        *value /= 5.0;
    }
    let actual = optimized::average(gradients);
    assert_eq!(
        sum.into_iter()
            .map(|(k, v)| (k, v.to_bits()))
            .collect::<Vec<_>>(),
        actual
            .iter()
            .map(|(k, v)| (k, v.to_bits()))
            .collect::<Vec<_>>(),
    );
    let empty = optimized::average(vec![optimized::Entries::from_reference(Gradient::new())]);
    assert_eq!(empty.iter().count(), 0);
}

#[test]
fn feature_specific_batch_reduction_sums_only_word_bigrams() {
    let rows = [
        (Feature::Word(1), 1.0),
        (Feature::Bigram(2, 3), 2.0),
        (Feature::Pos(4, 5), 3.0),
    ];
    let gradients = [1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(|multiplier| {
            optimized::Entries::from_reference(
                rows.into_iter()
                    .map(|(key, value)| (key, value * multiplier))
                    .collect(),
            )
        })
        .collect();
    let reduced = optimized::average_except(gradients, |feature| {
        matches!(feature, Feature::Bigram(_, _))
    });
    let actual = reduced.iter().collect::<BTreeMap<_, _>>();
    close(actual[&Feature::Word(1)], 2.5, 1e-12);
    close(actual[&Feature::Bigram(2, 3)], 20.0, 1e-12);
    close(actual[&Feature::Pos(4, 5)], 7.5, 1e-12);

    let summed = optimized::sum(vec![optimized::Entries::from_reference(BTreeMap::from([
        ([1_u32, 2, 3], 0.25),
    ]))]);
    let (key, value) = summed.iter().next().expect("one summed trigram entry");
    assert_eq!(key, [1, 2, 3]);
    close(value, 0.25, 1e-12);
}

#[test]
fn packed_worker_gradients_are_bit_identical_including_repeated_keys() {
    let (mut model, graph) = fixture();
    for iteration in 0..20 {
        let reference = first_order(&model, &graph).unwrap();
        let fast = optimized::first(&model, &graph).unwrap();
        assert_eq!(reference.loss.to_bits(), fast.0.to_bits());
        assert_eq!(
            reference
                .gradient
                .into_iter()
                .map(|(k, v)| (k, v.to_bits()))
                .collect::<Vec<_>>(),
            fast.1
                .iter()
                .map(|(k, v)| (k, v.to_bits()))
                .collect::<Vec<_>>(),
        );
        let paths = nbest(&model, &graph, 256, 1000).unwrap();
        let reference = rerank(&model, &graph, &paths).unwrap();
        let fast = optimized::rerank_for_test(&model, &graph, &paths).unwrap();
        assert_eq!(reference.loss.map(f64::to_bits), fast.0.map(f64::to_bits));
        assert_eq!(
            reference
                .gradient
                .iter()
                .map(|(&key, &value)| (key, value.to_bits()))
                .collect::<Vec<_>>(),
            fast.1
                .iter()
                .map(|&(key, value)| (key, value.to_bits()))
                .collect::<Vec<_>>(),
        );
        train_sentence(&mut model, &graph, 0.03, 0.001, 256, 1000).unwrap();
        if iteration == 10 {
            model.initial_pos.fill(-1000.0);
        }
    }
    // Repeated triples must be counted, not just deduplicated during sorting.
    let path = vec![0, 1, 1, 1, 1, 6];
    let paths = vec![Path {
        score: model.path_score(&graph, &path),
        nodes: path,
    }];
    let mut repeated_graph = graph;
    repeated_graph.gold.clone_from(&paths[0].nodes);
    let reference = rerank(&model, &repeated_graph, &paths).unwrap();
    let fast = optimized::rerank_for_test(&model, &repeated_graph, &paths).unwrap();
    assert_eq!(reference.loss, fast.0);
    assert_eq!(reference.gradient.into_iter().collect::<Vec<_>>(), fast.1);
}

#[test]
fn optimized_and_reference_batches_preserve_every_weight_bit() {
    let (initial, graph) = fixture();
    let fast = parallel::Trainer::new(8).unwrap();
    let reference = parallel::Trainer::with_reference_kernels(8, true).unwrap();
    let mut a = initial.clone();
    let mut b = initial;
    let mut other = graph.clone();
    other.gold = vec![0, 3, 6];
    for i in 0..20 {
        let mut c = batch_config();
        c.nbest = if i % 3 == 0 { 1 } else { 256 };
        c.gold_path_augmentation = i % 2 == 0;
        let ra = fast
            .train_batch(&mut a, &[&graph, &other, &graph], c)
            .unwrap();
        let rb = reference
            .train_batch(&mut b, &[&graph, &other, &graph], c)
            .unwrap();
        assert_eq!(ra.first_loss_sum.to_bits(), rb.first_loss_sum.to_bits());
        assert_eq!(ra.rerank_loss_sum.to_bits(), rb.rerank_loss_sum.to_bits());
        assert_eq!(ra.rerank_sentences, rb.rerank_sentences);
        assert_eq!(ra.first_path_gold_hits, rb.first_path_gold_hits);
        assert_eq!(ra.nbest_gold_hits, rb.nbest_gold_hits);
        assert_eq!(ra.gold_augmented_sentences, rb.gold_augmented_sentences);
        assert_eq!(output::fingerprint(&a), output::fingerprint(&b));
    }
}

fn branching_graph() -> Lattice {
    let mut nodes = vec![Node {
        start: 0,
        end: 0,
        word: BOS,
        script: 0,
    }];
    let mut gold = vec![0];
    for position in 0..12 {
        gold.push(nodes.len());
        for variant in 0..5 {
            let width = if variant >= 3 { 2 } else { 1 };
            if position + width > 12 {
                continue;
            }
            nodes.push(Node {
                start: position,
                end: position + width,
                word: 3 + u32::try_from((position + variant) % 5).unwrap(),
                script: 2 + u8::try_from(variant % 3).unwrap(),
            });
        }
    }
    gold.push(nodes.len());
    nodes.push(Node {
        start: 12,
        end: 12,
        word: EOS,
        script: 1,
    });
    Lattice::new(nodes, gold, 10000).unwrap()
}

#[test]
fn arena_nbest_and_local_edge_scores_match_reference_including_ties() {
    let (mut model, _) = fixture();
    let graph = branching_graph();
    assert!(graph.edges.len() >= 100); // Exercise node/shape score sharing.
    let mut workspace = astar::Workspace::default();
    let mut local = optimized::EdgeScores::default();
    for state in 0..4 {
        if state == 0 {
            model.words.iter_mut().for_each(|w| w.initial = 0.0);
        }
        if state == 1 {
            model.words.iter_mut().for_each(|w| w.initial = -0.2);
        }
        if state == 2 {
            model.first.values.insert(Feature::ScriptBigram(2, 4), 0.17);
            model.first.values.insert(Feature::Word(3), -0.04);
            model.first.values.insert(Feature::Bigram(3, 4), 0.22);
        }
        if state == 3 {
            // A previous cached sentence/phase must never retain these old costs.
            model.first.scale = 0.71;
            model.initial_pos.fill(-1000.0);
        }
        local.fill(&model, &graph);
        for (&score, &edge) in local.values.iter().zip(&graph.edges) {
            assert_eq!(score.to_bits(), model.edge_score(&graph, edge).to_bits());
        }
        for count in [1, 32, 256] {
            let expected = nbest(&model, &graph, count, 100_000).unwrap();
            let actual = workspace.nbest(&model, &graph, count, 100_000).unwrap();
            assert_eq!(expected.len(), actual.len());
            for (a, b) in actual.iter().zip(&expected) {
                assert_eq!(a.nodes, b.nodes);
                assert_eq!(a.score.to_bits(), b.score.to_bits());
            }
        }
        assert!(workspace.nbest(&model, &graph, 1, 0).is_err());
    }
}

#[test]
fn larger_variable_segmentation_preserves_worker_and_batch_gradient_bits() {
    let (mut a, _) = fixture();
    let graph = branching_graph();
    let mut b = a.clone();
    let fast = parallel::Trainer::new(8).unwrap();
    let reference = parallel::Trainer::with_reference_kernels(8, true).unwrap();
    let mut c = batch_config();
    c.astar_limit = 100_000;
    for _ in 0..3 {
        let expected = first_order(&a, &graph).unwrap();
        let actual = optimized::first(&a, &graph).unwrap();
        assert_eq!(expected.loss.to_bits(), actual.0.to_bits());
        assert_eq!(
            expected
                .gradient
                .into_iter()
                .map(|(k, v)| (k, v.to_bits()))
                .collect::<Vec<_>>(),
            actual
                .1
                .iter()
                .map(|(k, v)| (k, v.to_bits()))
                .collect::<Vec<_>>(),
        );
        fast.train_batch(&mut a, &[&graph, &graph], c).unwrap();
        reference.train_batch(&mut b, &[&graph, &graph], c).unwrap();
        assert_eq!(output::fingerprint(&a), output::fingerprint(&b));
    }
}

fn batch_config() -> parallel::BatchConfig {
    parallel::BatchConfig {
        rate: 0.03,
        l2: 0.01,
        nbest: 256,
        astar_limit: 1000,
        parameter_bytes: 1 << 20,
        gold_path_augmentation: false,
        primary_nbest_margin: None,
        primary_nbest_listwise: false,
    }
}

#[test]
fn primary_nbest_listwise_updates_a_raw_nbest_miss_with_gold_augmentation() {
    let (mut model, graph) = fixture();
    let mut config = batch_config();
    config.nbest = 1;
    config.primary_nbest_listwise = true;
    let rival = nbest(&model, &graph, config.nbest, config.astar_limit).unwrap()[0].clone();
    assert_ne!(rival.nodes, graph.gold);
    let gold_score = model.path_score(&graph, &graph.gold);
    let rival_score = rival.score;
    let before = gold_score - rival_score;
    let expected_loss = crate::log_add(gold_score, rival_score) - gold_score;

    let report = parallel::Trainer::new(2)
        .unwrap()
        .train_batch(&mut model, &[&graph], config)
        .unwrap();

    assert_eq!(
        report.nbest_gold_hits, 0,
        "raw hit is measured before augmentation"
    );
    assert_eq!(
        report.gold_augmented_sentences, 0,
        "primary loss includes gold independently of trigram augmentation"
    );
    assert_eq!(report.primary_listwise_active_sentences, 1);
    close(report.first_loss_sum, expected_loss, 1e-12);
    let after = model.path_score(&graph, &graph.gold) - model.path_score(&graph, &rival.nodes);
    assert!(
        after > before,
        "gold-vs-candidate gap did not improve: {before} -> {after}"
    );
}

#[test]
fn primary_nbest_pairwise_hinge_updates_a_raw_nbest_miss() {
    let (mut model, graph) = fixture();
    let mut config = batch_config();
    config.nbest = 1;
    config.primary_nbest_margin = Some(0.1);
    let rival = nbest(&model, &graph, config.nbest, config.astar_limit).unwrap()[0].clone();
    assert_ne!(rival.nodes, graph.gold);
    let gold_score = model.path_score(&graph, &graph.gold);
    let rival_score = model.path_score(&graph, &rival.nodes);
    let before = gold_score - rival_score;
    let expected_hinge = (0.1 + rival_score - gold_score).max(0.0);

    let report = parallel::Trainer::new(2)
        .unwrap()
        .train_batch(&mut model, &[&graph], config)
        .unwrap();

    assert_eq!(report.nbest_gold_hits, 0);
    assert_eq!(report.primary_margin_active_sentences, 1);
    close(report.first_loss_sum, expected_hinge, 1e-12);
    let after = model.path_score(&graph, &graph.gold) - model.path_score(&graph, &rival.nodes);
    assert!(
        after > before,
        "gold-vs-Nth gap did not improve: {before} -> {after}"
    );
}

#[test]
fn primary_nbest_pairwise_hinge_updates_wrong_candidates_even_when_gold_is_present() {
    let (mut model, graph) = fixture();
    let paths = enumerate(&model, &graph);
    assert!(paths.len() > 1);
    assert_ne!(paths[0].nodes, graph.gold);
    let mut config = batch_config();
    config.nbest = paths.len();
    config.primary_nbest_margin = Some(0.0);
    let gold_score = model.path_score(&graph, &graph.gold);
    let before_loss: f64 = paths
        .iter()
        .filter(|path| path.nodes != graph.gold)
        .map(|path| (model.path_score(&graph, &path.nodes) - gold_score).max(0.0))
        .sum();
    assert!(before_loss > 0.0);

    let report = parallel::Trainer::new(2)
        .unwrap()
        .train_batch(&mut model, &[&graph], config)
        .unwrap();

    assert_eq!(report.nbest_gold_hits, 1);
    assert_eq!(report.primary_margin_active_sentences, 1);
    close(report.first_loss_sum, before_loss, 1e-12);
    assert!(!model.first.values.is_empty());
    let gold_score = model.path_score(&graph, &graph.gold);
    let after_loss: f64 = paths
        .iter()
        .filter(|path| path.nodes != graph.gold)
        .map(|path| (model.path_score(&graph, &path.nodes) - gold_score).max(0.0))
        .sum();
    assert!(
        after_loss < before_loss,
        "pairwise negative correction did not reduce summed wrong-path hinge: {before_loss} -> {after_loss}"
    );
}

#[test]
fn one_sentence_batch_uses_same_preupdate_state_for_both_gradient_families() {
    let (initial, graph) = fixture();
    let c = batch_config();
    let first = first_order(&initial, &graph).unwrap();
    let paths = nbest(&initial, &graph, c.nbest, c.astar_limit).unwrap();
    let third = rerank(&initial, &graph, &paths).unwrap();
    let mut expected = initial.clone();
    expected
        .first
        .update(&first.gradient, c.rate, c.l2)
        .unwrap();
    expected
        .third
        .update(&third.gradient, c.rate, c.l2)
        .unwrap();
    expected
        .third_pos
        .update(&third.pos_gradient, c.rate, c.l2)
        .unwrap();
    expected
        .third_reading_length
        .update(&third.reading_length_gradient, c.rate, c.l2)
        .unwrap();

    let mut actual = initial;
    let report = parallel::Trainer::new(16)
        .unwrap()
        .train_batch(&mut actual, &[&graph], c)
        .unwrap();
    assert_eq!(report.sentences, 1);
    assert_eq!(output::fingerprint(&expected), output::fingerprint(&actual));
}

#[test]
fn thread_count_cannot_change_batch_weights_or_l2_scales() {
    let (initial, graph) = fixture();
    let mut other = graph.clone();
    other.gold = vec![0, 2, 5, 6];
    let mut third = graph.clone();
    third.gold = vec![0, 3, 6];
    let graphs = [&graph, &other, &third, &other];
    let mut reference = None;
    for threads in [1, 2, 8, 16] {
        let trainer = parallel::Trainer::new(threads).unwrap();
        let mut model = initial.clone();
        for _ in 0..4 {
            trainer
                .train_batch(&mut model, &graphs, batch_config())
                .unwrap();
        }
        let fingerprint = output::fingerprint(&model);
        if let Some(expected) = &reference {
            assert_eq!(&fingerprint, expected);
        } else {
            reference = Some(fingerprint);
        }
    }
}

#[test]
fn batch_sums_all_word_bigram_composition_and_word_trigrams_but_means_other_features() {
    let (initial, graph) = fixture();
    let mut other = graph.clone();
    other.gold = vec![0, 2, 5, 6];
    let graphs = [&graph, &other, &graph];
    let c = batch_config();
    let mut manual = initial.clone();
    let mut gradient = Gradient::new();
    for g in &graphs {
        for (key, value) in first_order(&manual, g).unwrap().gradient {
            *gradient.entry(key).or_default() += value;
        }
    }
    for (key, value) in &mut gradient {
        if !matches!(
            *key,
            Feature::Bigram(_, _) | Feature::PosWord(_, _) | Feature::WordPos(_, _)
        ) {
            *value /= 3.0;
        }
    }
    let mut pos_third = PosTriGradient::new();
    let mut reading_length_third = ReadingLengthTriGradient::new();
    let mut word_third = TriGradient::new();
    for g in &graphs {
        let paths = nbest(&initial, g, c.nbest, c.astar_limit).unwrap();
        let rerank = rerank(&initial, g, &paths).unwrap();
        for (key, value) in rerank.gradient {
            *word_third.entry(key).or_default() += value;
        }
        for (key, value) in rerank.pos_gradient {
            *pos_third.entry(key).or_default() += value;
        }
        for (key, value) in rerank.reading_length_gradient {
            *reading_length_third.entry(key).or_default() += value;
        }
    }
    for value in pos_third.values_mut() {
        *value /= 3.0;
    }
    for value in reading_length_third.values_mut() {
        *value /= 3.0;
    }
    manual.first.update(&gradient, c.rate, c.l2).unwrap();
    manual.third.update(&word_third, c.rate, c.l2).unwrap();
    manual.third_pos.update(&pos_third, c.rate, c.l2).unwrap();
    manual
        .third_reading_length
        .update(&reading_length_third, c.rate, c.l2)
        .unwrap();
    let mut actual = initial;
    parallel::Trainer::new(16)
        .unwrap()
        .train_batch(&mut actual, &graphs, c)
        .unwrap();
    assert_eq!(output::fingerprint(&manual), output::fingerprint(&actual));
    close(actual.first.scale, 1.0 - c.rate * c.l2, 1e-15);
}

#[test]
fn parallel_batch_failure_rolls_back_first_phase_and_preserves_all_features() {
    let trainer = parallel::Trainer::new(16).unwrap();
    let (mut model, graph) = fixture();
    train_sentence(&mut model, &graph, 0.03, 0.01, 32, 1000).unwrap();
    for rebase in [false, true] {
        if rebase {
            model.first.scale = 1e-101;
        }
        let initial = output::fingerprint(&model);
        let mut c = batch_config();
        c.astar_limit = 1;
        let error = trainer
            .train_batch(&mut model, &[&graph, &graph], c)
            .unwrap_err();
        assert!(error.to_string().contains("A* expansion budget"));
        assert_eq!(initial, output::fingerprint(&model));
        c = batch_config();
        c.parameter_bytes = 1;
        let error = trainer
            .train_batch(&mut model, &[&graph, &graph], c)
            .unwrap_err();
        assert!(error.downcast_ref::<ParameterLimit>().is_some());
        assert_eq!(initial, output::fingerprint(&model));
    }
}

#[test]
fn delta_cutoff_uses_effective_magnitudes_keeps_boundaries_and_releases_capacity() {
    let (mut model, _) = fixture();
    let initial_words = serde_json::to_value(&model.words).unwrap();
    let initial_pos = model.initial_pos.clone();
    model.first.scale = 0.5;
    model.third.scale = 0.25;
    for n in 0..2000 {
        model.first.values.insert(Feature::Bigram(n, 5), -1e-5);
        model.third.values.insert([n, 4, 5], 1e-5);
    }
    let small = [
        Feature::Word(3),
        Feature::Pos(3, 3),
        Feature::Script(4),
        Feature::ScriptBigram(2, 4),
    ];
    for key in small {
        model.first.values.insert(key, 1e-5);
    }
    let survivors = [
        (Feature::Word(4), 0.2),
        (Feature::Pos(1, 3), -0.3),
        (Feature::Bigram(3, 4), 0.002),
    ];
    for (key, raw) in survivors {
        model.first.values.insert(key, raw);
    }
    model.third.values.insert([3, 4, 6], -0.004);
    let report = cutoff::apply(
        &mut model,
        cutoff::Policy {
            minimum_delta: 0.001,
            maximum_delta: 0.001,
            target_fraction: 0.75,
        },
        1 << 20,
    )
    .unwrap();
    assert_eq!(report.selected_delta.to_bits(), 0.001_f64.to_bits());
    assert_eq!(report.removed_total, 4004);
    assert!(report.table_bytes_after * 100 < report.table_bytes_before);
    assert!(report.target_reached);
    for key in small {
        assert_eq!(model.first.get(key).to_bits(), 0.0_f64.to_bits());
    }
    for (key, raw) in survivors {
        assert_eq!(
            model.first.values.get(&key).unwrap().to_bits(),
            raw.to_bits()
        );
    }
    assert_eq!(
        model.third.values.get(&[3, 4, 6]).unwrap().to_bits(),
        (-0.004_f64).to_bits()
    );
    assert_eq!(model.first.scale.to_bits(), 0.5_f64.to_bits());
    assert_eq!(model.third.scale.to_bits(), 0.25_f64.to_bits());
    assert_eq!(serde_json::to_value(&model.words).unwrap(), initial_words);
    assert_eq!(model.initial_pos, initial_pos);
    assert_eq!(report.families["pos-bigram"].removed, 1);
    assert_eq!(report.families["word"].removed, 1);
    // Removed keys are not blacklisted: a subsequent gradient can learn them.
    model
        .first
        .update(&BTreeMap::from([(Feature::Word(3), -1.0)]), 0.1, 0.0)
        .unwrap();
    close(model.first.get(Feature::Word(3)), 0.1, 0.0);
}

#[test]
fn pressure_cutoff_selects_smallest_configured_band_and_never_exceeds_maximum() {
    let (mut model, _) = fixture();
    for n in 0..2000 {
        model
            .first
            .values
            .insert(Feature::Bigram(n, 5), if n < 1800 { 0.0005 } else { 0.1 });
    }
    let policy = cutoff::Policy {
        minimum_delta: 1e-4,
        maximum_delta: 1e-2,
        target_fraction: 0.75,
    };
    let report = cutoff::apply(&mut model, policy, 20_000).unwrap();
    assert_eq!(report.selected_delta.to_bits(), 1e-3_f64.to_bits());
    assert_eq!(report.removed_total, 1800);
    assert!(report.target_reached);
    assert_eq!(model.first.values.len(), 200);
    let saved = output::fingerprint(&model);
    let impossible = cutoff::apply(&mut model, policy, 1).unwrap();
    assert_eq!(impossible.selected_delta.to_bits(), 1e-2_f64.to_bits());
    assert_eq!(impossible.removed_total, 0);
    assert!(!impossible.target_reached);
    assert_eq!(output::fingerprint(&model), saved);
    for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
        assert!(
            cutoff::apply(
                &mut model,
                cutoff::Policy {
                    minimum_delta: bad,
                    ..policy
                },
                1000
            )
            .is_err()
        );
        assert_eq!(output::fingerprint(&model), saved);
    }
}

#[test]
fn pressure_cutoff_reuses_hashmap_capacity_for_the_pending_batch() {
    let (mut model, _) = fixture();
    for n in 0..2000 {
        model
            .first
            .values
            .insert(Feature::Bigram(n, 5), if n < 1800 { 0.0005 } else { 0.1 });
    }
    let bytes_before = model.first.values.projected_bytes(0)
        + model.third.values.projected_bytes(0)
        + model.third_pos.values.projected_bytes(0)
        + model.third_reading_length.values.projected_bytes(0);
    let allocation_before = model.first.values.projected_bytes(0);
    let report = cutoff::apply_for_retry(
        &mut model,
        cutoff::Policy {
            minimum_delta: 1e-4,
            maximum_delta: 1e-2,
            target_fraction: 0.75,
        },
        80_000,
        [16, 32, 0, 0],
    )
    .unwrap();
    assert_eq!(report.selected_delta.to_bits(), 1e-3_f64.to_bits());
    assert_eq!(report.removed_total, 1800);
    assert!(report.target_reached);
    assert!(report.retry_capacity_reached);
    assert_eq!(model.first.values.projected_bytes(0), allocation_before);
    assert_eq!(report.table_bytes_after, bytes_before);
    assert_eq!(model.first.values.len(), 200);
}

#[test]
fn private_pool_really_uses_requested_workers_and_retains_result_order() {
    let trainer = parallel::Trainer::new(16).unwrap();
    assert_eq!(trainer.threads(), 16);
    let barrier = std::sync::Barrier::new(16);
    let values = trainer
        .map(&(0..16).collect::<Vec<_>>(), |&i| {
            let id = std::thread::current().id();
            barrier.wait();
            Ok((i, id))
        })
        .unwrap();
    assert_eq!(
        values.iter().map(|p| p.0).collect::<Vec<_>>(),
        (0..16).collect::<Vec<_>>()
    );
    assert_eq!(
        values
            .iter()
            .map(|p| p.1)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        16
    );
    assert!(parallel::Trainer::new(0).is_err());
}

#[test]
fn page_blocks_count_pages_not_sentences_and_keep_pages_intact() {
    let temp = Temp::new();
    let mut builder = blocks::Builder::new(&temp.0.join("blocks"), 1000).unwrap();
    for page in 1..=1001 {
        for repetition in 0..2 {
            let record = usize::try_from((page - 1) * 2 + repetition + 1).unwrap();
            builder.observe(page, record).unwrap();
            builder
                .add(
                    record,
                    &corpus::Sentence {
                        page_id: page,
                        title: "ページ".into(),
                        sentence: "阿".into(),
                        reading: "あ".into(),
                        spans: vec![corpus::Span {
                            start: 0,
                            end: 3,
                            word: 3,
                            script: 4,
                            forced: false,
                        }],
                    },
                )
                .unwrap();
        }
    }
    let cache = builder.finish().unwrap();
    assert_eq!(cache.blocks.len(), 2);
    assert_eq!(cache.blocks[0].pages, 1000);
    assert_eq!(cache.blocks[0].sentences, 2000);
    assert_eq!(cache.blocks[1].pages, 1);
    assert_eq!(cache.blocks[1].sentences, 2);
    for block in 0..2 {
        let mut reader = cache.open(block).unwrap();
        let mut line = String::new();
        let mut pages = Vec::new();
        while let Some(sentence) = blocks::read_cached(&mut *reader, &mut line).unwrap() {
            pages.push(sentence.sentence.page_id);
        }
        assert_eq!(pages.len(), cache.blocks[block].sentences);
        for pair in pages.chunks(2) {
            assert_eq!(pair[0], pair[1]);
        }
        assert!(
            pages
                .iter()
                .all(|&p| if block == 0 { p <= 1000 } else { p == 1001 })
        );
    }
    assert!(blocks::Builder::new(&temp.0.join("blocks"), 1000).is_err());
}

#[test]
fn page_block_order_is_seeded_complete_and_epoch_specific() {
    let cache = blocks::PageBlocks {
        directory: PathBuf::new(),
        pages_per_block: 1000,
        training_sentences: 0,
        blocks: (0..100)
            .map(|n| blocks::Block {
                file: n.to_string(),
                pages: 1000,
                sentences: 0,
                source_first_record: n,
                source_last_record: n,
                compressed_bytes: 0,
            })
            .collect(),
    };
    let order = cache.order(20_261_003, 1);
    assert_eq!(order, cache.order(20_261_003, 1));
    assert_ne!(order, cache.order(20_261_003, 2));
    assert_ne!(order, cache.order(20_261_004, 1));
    let mut sorted = order;
    sorted.sort_unstable();
    assert_eq!(sorted, (0..100).collect::<Vec<_>>());
}

#[test]
fn page_cache_rejects_noncontiguous_page_records_and_wrong_page_gold() {
    let temp = Temp::new();
    let mut builder = blocks::Builder::new(&temp.0.join("blocks"), 1000).unwrap();
    builder.observe(1, 1).unwrap();
    builder.observe(2, 2).unwrap();
    assert!(builder.observe(1, 3).is_err());
    assert!(
        builder
            .add(
                4,
                &corpus::Sentence {
                    page_id: 999,
                    title: String::new(),
                    sentence: String::new(),
                    reading: String::new(),
                    spans: vec![],
                }
            )
            .is_err()
    );
}

#[test]
fn batch_trigram_gold_missing_does_not_inject_gold_or_apply_hidden_decay() {
    let (mut model, graph) = fixture();
    model.third.values.insert([BOS, 6, EOS], 0.5);
    let previous = model.third.clone();
    let mut c = batch_config();
    c.nbest = 1;
    let trainer = parallel::Trainer::new(16).unwrap();
    let report = trainer
        .train_batch(&mut model, &[&graph, &graph], c)
        .unwrap();
    assert_eq!(report.rerank_sentences, 0);
    close(report.rerank_loss_sum, 0.0, 1e-15);
    assert_eq!(model.third.values, previous.values);
    assert_eq!(model.third.scale.to_bits(), previous.scale.to_bits());
}

#[test]
fn batch_gold_augmentation_trains_trigram_but_keeps_raw_nbest_hit_metric() {
    let (mut model, graph) = fixture();
    let previous = model.third.clone();
    let mut c = batch_config();
    c.nbest = 1;
    c.gold_path_augmentation = true;
    let report = parallel::Trainer::new(4)
        .unwrap()
        .train_batch(&mut model, &[&graph, &graph], c)
        .unwrap();
    assert_eq!(report.nbest_gold_hits, 0);
    assert_eq!(report.gold_augmented_sentences, 2);
    assert_eq!(report.rerank_sentences, 2);
    assert!(report.rerank_loss_sum > 0.0);
    assert_ne!(model.third.values, previous.values);
    assert!(!model.third_pos.values.is_empty() || !model.third_reading_length.values.is_empty());
}

#[test]
fn whole_corpus_reservoir_is_bounded_seeded_and_not_a_prefix() {
    let collect = |seed| {
        let mut sample = corpus::Reservoir::new(20, seed);
        for value in 0..10_000 {
            sample.push(value);
        }
        sample.into_values()
    };
    let first = collect(20_261_003);
    assert_eq!(first.len(), 20);
    assert_eq!(first, collect(20_261_003));
    assert_ne!(first, collect(20_261_004));
    assert!(first.iter().any(|&value| value >= 9000));
    assert_eq!(
        first
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        20
    );
    let mut small = corpus::Reservoir::new(20, 1);
    for value in 0..3 {
        small.push(value);
    }
    assert_eq!(small.into_values(), vec![0, 1, 2]);
    let mut empty = corpus::Reservoir::new(0, 0);
    empty.push(1);
    assert!(empty.into_values().is_empty());
}
