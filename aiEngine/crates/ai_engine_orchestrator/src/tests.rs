use super::*;
use std::sync::Arc;

fn word(reading: &str, surface: &str, score: f64) -> QueryWord {
    QueryWord {
        reading: reading.into(),
        surface: surface.into(),
        score,
    }
}
fn dictionary() -> MemoryDictionary {
    MemoryDictionary::new([
        word("にほん", "日本", 10.0),
        word("にほんご", "日本語", 30.0),
        word("ご", "語", 5.0),
        word("へんかん", "変換", 20.0),
        word("かう", "買う", 10.0),
        word("かう", "飼う", 9.0),
        word("しゃ", "社", 10.0),
        word("し", "市", 10.0),
        word("ゃ", "ゃ", -10.0),
        word("かな", "仮名", 20.0),
    ])
    .unwrap()
}
fn focus<D: DictionaryQuery>(session: &mut ImeSession<D>) {
    let _ = session.process_event(Event::FocusIn {
        context_id: 1,
        caret: CaretInfo::default(),
        committed_context: String::new(),
    });
}
fn key<D: DictionaryQuery>(session: &mut ImeSession<D>, key: Key) -> EventResult {
    session.process_event(Event::KeyDown {
        key,
        modifiers: Modifiers::default(),
    })
}
fn type_text<D: DictionaryQuery>(session: &mut ImeSession<D>, text: &str) {
    for ch in text.chars() {
        assert!(key(session, Key::Character(ch)).consumed);
    }
}
fn session() -> ImeSession<MemoryDictionary> {
    let mut session = ImeSession::new(1, dictionary(), ControllerConfig::default());
    focus(&mut session);
    session
}

#[test]
fn layered_raw_phonetic_and_kanji_sources_are_preserved() {
    let mut session = session();
    type_text(&mut session, "shi");
    let range = &session.edit_line().ranges()[0];
    assert_eq!(
        (range.text.as_str(), range.layer),
        ("し", TextLayer::Phonetic)
    );
    assert_eq!(
        range
            .sources
            .iter()
            .map(|n| n.text.as_str())
            .collect::<String>(),
        "shi"
    );
    let result = key(&mut session, Key::Space);
    assert!(
        result
            .actions
            .iter()
            .any(|a| matches!(a, Action::UpdateComposition { text, .. } if text == "市"))
    );
    let top = &session.edit_line().ranges()[0];
    assert_eq!((top.text.as_str(), top.layer), ("市", TextLayer::Converted));
    assert_eq!(top.sources[0].text, "し");
    assert_eq!(top.sources[0].sources.len(), 3);
    let _ = key(&mut session, Key::Escape);
    assert_eq!(session.edit_line().composition(), "し");
}

#[test]
fn backspace_deletes_visible_characters_not_raw_keystrokes() {
    let mut session = session();
    type_text(&mut session, "ochame");
    assert_eq!(session.edit_line().composition(), "おちゃめ");
    let _ = key(&mut session, Key::Backspace);
    let _ = key(&mut session, Key::Backspace);
    assert_eq!(session.edit_line().composition(), "おち");
}

#[test]
fn cursor_insertion_respects_phonetic_boundaries() {
    let mut session = session();
    type_text(&mut session, "aaa");
    let _ = key(&mut session, Key::Left);
    let _ = key(&mut session, Key::Left);
    type_text(&mut session, "chai");
    assert_eq!(session.edit_line().composition(), "あちゃいああ");
    assert_eq!(session.edit_line().caret(), (4, 0));
    let _ = key(&mut session, Key::Home);
    let _ = key(&mut session, Key::Delete);
    assert_eq!(session.edit_line().composition(), "ちゃいああ");
}

#[test]
fn n_does_not_consume_phonetic_text_after_caret() {
    let mut session = session();
    type_text(&mut session, "ai");
    let _ = key(&mut session, Key::Left);
    type_text(&mut session, "n");
    assert_eq!(session.edit_line().composition(), "あnい");
    type_text(&mut session, "a");
    assert_eq!(session.edit_line().composition(), "あない");
}

#[test]
fn pending_romaji_longest_match_and_gemination() {
    for (input, expected) in [
        ("kya", "きゃ"),
        ("shi", "し"),
        ("si", "し"),
        ("chi", "ち"),
        ("ti", "ち"),
        ("kikkake", "きっかけ"),
        ("konnichiha", "こんにちは"),
        ("kamennraida-", "かめんらいだー"),
        ("www", "www"),
        ("n", "n"),
        ("n'a", "んあ"),
    ] {
        let mut session = session();
        type_text(&mut session, input);
        assert_eq!(session.edit_line().composition(), expected, "{input}");
    }
    let mut session = session();
    type_text(&mut session, "kan");
    let result = key(&mut session, Key::Enter);
    assert!(result.actions.contains(&Action::Commit {
        text: "かん".into()
    }));
}

#[test]
fn kana_range_can_be_split_without_losing_provenance() {
    let mut line = EditLine::default();
    line.insert_raw("sha");
    line.normalize_before_caret(false);
    line.set_caret(1, 1).unwrap();
    line.replace("", TextLayer::Raw);
    assert_eq!(line.composition(), "し");
    let fragment = &line.ranges()[0];
    assert_eq!(fragment.source_slice, Some(0..1));
    assert_eq!(fragment.sources[0].text, "しゃ");
    assert_eq!(
        fragment.sources[0]
            .sources
            .iter()
            .map(|n| n.text.as_str())
            .collect::<String>(),
        "sha"
    );
}

#[test]
fn dictionary_matches_across_ranges_and_competing_word_boundaries() {
    let mut line = EditLine::default();
    line.insert_raw("nihongo");
    line.normalize_before_caret(false);
    let output = dictionary().query(&line, "").unwrap();
    assert!(
        output
            .edges
            .iter()
            .any(|e| e.start == 0 && e.end == 4 && e.word.surface == "日本語")
    );
    assert!(
        output
            .edges
            .iter()
            .any(|e| e.start == 0 && e.end == 3 && e.word.surface == "日本")
    );
    let paths = output.nbest(&line.input_lattice(), 5).unwrap();
    assert_eq!(paths[0].text, "日本語");
}

#[test]
fn debug_string_query_creates_the_same_input_lattice() {
    let mut line = EditLine::default();
    line.replace("にほんご", TextLayer::Phonetic);
    assert_eq!(
        dictionary().query_reading("にほんご").unwrap(),
        dictionary().query(&line, "").unwrap()
    );
}

#[test]
fn converted_query_binds_reading_and_surface() {
    let mut line = EditLine::default();
    line.replace("かう", TextLayer::Phonetic);
    line.apply_words(&[(0..2, "飼う".into(), "かう".into())])
        .unwrap();
    let input = line.input_lattice();
    assert_eq!(input.edges()[0].display.as_deref(), Some("飼う"));
    assert_eq!(input.edges()[0].reading, "かう");
    let output = dictionary().query(&line, "").unwrap();
    assert!(
        output
            .edges
            .iter()
            .any(|e| e.word.surface == "飼う" && !e.fallback)
    );
    assert!(output.edges.iter().all(|e| e.word.surface != "買う"));
    assert_eq!(output.nbest(&input, 5).unwrap()[0].text, "飼う");
}

#[test]
fn bound_unknown_surface_is_kept_without_unbinding_it() {
    let mut line = EditLine::default();
    line.replace("かう", TextLayer::Phonetic);
    line.apply_words(&[(0..2, "架宇".into(), "かう".into())])
        .unwrap();
    let output = dictionary().query(&line, "").unwrap();
    assert_eq!(
        output.nbest(&line.input_lattice(), 5).unwrap()[0].text,
        "架宇"
    );
    assert!(output.edges.iter().all(|e| e.fallback));
}

#[test]
fn dictionary_walks_alternative_input_edges_without_flattening() {
    let node = Arc::new(TextRange {
        text: "か".into(),
        layer: TextLayer::Phonetic,
        sources: Vec::new(),
        source_slice: None,
    });
    let make = |start, end, reading: &str| InputEdge {
        start,
        end,
        reading: reading.into(),
        display: None,
        searchable: true,
        source: Arc::clone(&node),
        source_slice: 0..1,
    };
    let input = InputLattice::new(
        2,
        vec![make(0, 1, "か"), make(0, 1, "し"), make(1, 2, "う")],
    )
    .unwrap();
    let dict =
        MemoryDictionary::new([word("かう", "買う", 10.0), word("しう", "試右", 11.0)]).unwrap();
    let output = dict.query_lattice(&input).unwrap();
    assert!(
        output
            .edges
            .iter()
            .any(|e| e.word.surface == "買う" && e.input_edges == [0, 2])
    );
    assert!(
        output
            .edges
            .iter()
            .any(|e| e.word.surface == "試右" && e.input_edges == [1, 2])
    );
    assert_eq!(output.nbest(&input, 2).unwrap()[0].text, "試右");
}

#[test]
fn selected_candidate_is_the_visible_layer_and_commit_source() {
    let mut session = session();
    type_text(&mut session, "kau");
    let _ = key(&mut session, Key::Space);
    assert_eq!(session.edit_line().composition(), "買う");
    let _ = key(&mut session, Key::Space);
    assert_eq!(session.edit_line().composition(), "飼う");
    let result = key(&mut session, Key::Enter);
    assert!(result.actions.contains(&Action::Commit {
        text: "飼う".into()
    }));
    assert!(
        result
            .actions
            .iter()
            .any(|a| matches!(a, Action::LearnSelection { words } if words[0].surface == "飼う"))
    );
    assert!(session.edit_line().is_empty());
    assert_eq!(session.committed_context(), "飼う");
}

#[test]
fn invalid_or_cross_session_candidates_never_commit() {
    let mut session = session();
    type_text(&mut session, "kau");
    let _ = key(&mut session, Key::Space);
    let id = session.candidates()[0].id;
    assert!(
        !session
            .process_event(Event::CandidateSelected {
                id: CandidateId {
                    session_id: 2,
                    ..id
                }
            })
            .consumed
    );
    let _ = key(&mut session, Key::Escape);
    let _ = key(&mut session, Key::Space);
    assert!(
        !session
            .process_event(Event::CandidateSelected { id })
            .consumed
    );
    let result = session.process_event(Event::CandidateSelected {
        id: session.candidates()[0].id,
    });
    assert!(result.actions.contains(&Action::Commit {
        text: "買う".into()
    }));
}

#[derive(Debug)]
struct FailedDictionary;
impl DictionaryQuery for FailedDictionary {
    fn query(&self, _input: &EditLine, _context: &str) -> Result<OutputLattice, String> {
        Err("reader unavailable".into())
    }
}
#[test]
fn dictionary_failure_leaves_reading_committable() {
    let mut session = ImeSession::new(3, FailedDictionary, ControllerConfig::default());
    focus(&mut session);
    type_text(&mut session, "kana");
    let result = key(&mut session, Key::Space);
    assert_eq!(session.phase(), Phase::Composing);
    assert!(result.actions.contains(&Action::Error {
        message: "reader unavailable".into()
    }));
    assert!(
        key(&mut session, Key::Enter)
            .actions
            .contains(&Action::Commit {
                text: "かな".into()
            })
    );
}

#[test]
fn selection_replace_unicode_and_cancel_preserve_edit_invariants() {
    let mut session = session();
    type_text(&mut session, "あ😀いう");
    assert!(
        session
            .process_event(Event::SetSelection {
                position: 1,
                length: 2
            })
            .consumed
    );
    type_text(&mut session, "え");
    assert_eq!(session.edit_line().composition(), "あえう");
    assert!(
        !session
            .process_event(Event::SetSelection {
                position: usize::MAX,
                length: 1
            })
            .consumed
    );
    let _ = key(&mut session, Key::Escape);
    assert_eq!(session.phase(), Phase::Direct);
    assert!(session.edit_line().is_empty());
}

#[test]
fn shortcuts_idle_and_focus_loss_are_not_commits() {
    let mut session = ImeSession::new(1, dictionary(), ControllerConfig::default());
    assert!(!key(&mut session, Key::Character('a')).consumed);
    focus(&mut session);
    assert!(!key(&mut session, Key::Enter).consumed);
    assert!(
        !session
            .process_event(Event::KeyDown {
                key: Key::Character('c'),
                modifiers: Modifiers {
                    control: true,
                    ..Modifiers::default()
                }
            })
            .consumed
    );
    type_text(&mut session, "kana");
    let result = session.process_event(Event::FocusOut);
    assert!(
        !result
            .actions
            .iter()
            .any(|a| matches!(a, Action::Commit { .. }))
    );
    assert!(session.edit_line().is_empty());
}

#[test]
fn predictions_are_explicit_and_stale_results_are_rejected() {
    let mut session = ImeSession::new(
        7,
        dictionary(),
        ControllerConfig {
            prediction_enabled: true,
            ..ControllerConfig::default()
        },
    );
    focus(&mut session);
    type_text(&mut session, "kana");
    let _ = key(&mut session, Key::Tab);
    let generation = session.generation();
    let wrong = session.process_event(Event::PredictionReady {
        session_id: 8,
        generation,
        result: Ok("です".into()),
    });
    assert!(wrong.actions.is_empty());
    let ready = session.process_event(Event::PredictionReady {
        session_id: 7,
        generation,
        result: Ok("です".into()),
    });
    assert!(ready.actions.contains(&Action::ShowGhost {
        generation,
        text: "です".into()
    }));
    assert!(
        !ready
            .actions
            .iter()
            .any(|a| matches!(a, Action::Commit { .. }))
    );
    assert!(
        key(&mut session, Key::Tab)
            .actions
            .contains(&Action::Commit {
                text: "かなです".into()
            })
    );
    assert!(
        session
            .process_event(Event::PredictionReady {
                session_id: 7,
                generation,
                result: Ok("古い".into())
            })
            .actions
            .is_empty()
    );
}

#[test]
fn context_edit_and_typing_cancel_predictions() {
    let mut session = ImeSession::new(
        7,
        dictionary(),
        ControllerConfig {
            prediction_enabled: true,
            ..ControllerConfig::default()
        },
    );
    focus(&mut session);
    let _ = key(&mut session, Key::Tab);
    let generation = session.generation();
    let result = session.process_event(Event::UpdateContext {
        text: "新文脈".into(),
    });
    assert!(result.actions.contains(&Action::CancelPrediction {
        session_id: 7,
        generation
    }));
    let _ = key(&mut session, Key::Tab);
    let generation = session.generation();
    type_text(&mut session, "a");
    assert!(
        session
            .process_event(Event::PredictionReady {
                session_id: 7,
                generation,
                result: Ok("旧予測".into())
            })
            .actions
            .is_empty()
    );
}

#[test]
fn invalid_word_paths_and_bindings_are_rejected_transactionally() {
    let mut line = EditLine::default();
    line.replace("かな", TextLayer::Phonetic);
    let original = line.clone();
    assert!(
        line.apply_words(&[(0..2, "仮名".into(), "かに".into())])
            .is_err()
    );
    assert_eq!(line, original);
    line.apply_words(&[(0..2, "仮名".into(), "かな".into())])
        .unwrap();
    let input = line.input_lattice();
    let output = OutputLattice {
        end: 2,
        edges: vec![OutputEdge {
            start: 0,
            end: 2,
            word: word("かな", "加奈", 10.0),
            input_edges: vec![0],
            fallback: false,
        }],
    };
    assert!(output.nbest(&input, 1).is_err());
}

#[test]
fn direct_and_fullwidth_modes() {
    let mut session = session();
    let _ = session.process_event(Event::SetMode {
        mode: InputMode::Direct,
    });
    assert!(!key(&mut session, Key::Character('a')).consumed);
    let _ = session.process_event(Event::SetMode {
        mode: InputMode::FullWidthAscii,
    });
    type_text(&mut session, "Ab1");
    assert_eq!(session.edit_line().composition(), "Ａｂ１");
}

#[test]
fn bound_word_and_free_reading_share_reading_frames() {
    let dict = MemoryDictionary::new([
        word("かう", "飼う", 10.0),
        word("かう", "買う", 20.0),
        word("いぬ", "犬", 5.0),
        word("かういぬ", "買う犬", 100.0),
    ])
    .unwrap();
    let mut line = EditLine::default();
    line.replace("かう", TextLayer::Phonetic);
    line.apply_words(&[(0..2, "飼う".into(), "かう".into())])
        .unwrap();
    line.insert_raw("inu");
    line.normalize_before_caret(false);
    assert_eq!(line.composition(), "飼ういぬ");
    let input = line.input_lattice();
    let output = dict.query(&line, "").unwrap();
    assert_eq!(output.nbest(&input, 1).unwrap()[0].text, "飼う犬");
    assert!(
        output
            .edges
            .iter()
            .all(|edge| edge.word.surface != "買う犬")
    );
    line.apply_words(&[
        (0..2, "飼う".into(), "かう".into()),
        (2..4, "犬".into(), "いぬ".into()),
    ])
    .unwrap();
    assert_eq!(line.composition(), "飼う犬");
    line.cancel_conversion();
    assert_eq!(line.composition(), "かういぬ");
}

#[test]
fn bound_multichar_word_has_no_invented_per_kanji_reading() {
    let mut line = EditLine::default();
    line.replace("にほんご", TextLayer::Phonetic);
    line.apply_words(&[(0..4, "日本語".into(), "にほんご".into())])
        .unwrap();
    assert_eq!(line.len(), 3);
    assert_eq!(line.input_lattice().end(), 4);
    assert!(line.set_caret(1, 0).is_err());
    line.backspace();
    assert_eq!(line.composition(), "にほんご");
}

#[test]
fn katakana_preserves_small_kana_and_final_n() {
    let mut session = session();
    let _ = session.process_event(Event::SetMode {
        mode: InputMode::Katakana,
    });
    type_text(&mut session, "kyan");
    assert_eq!(session.edit_line().composition(), "キャn");
    assert!(
        key(&mut session, Key::Enter)
            .actions
            .contains(&Action::Commit {
                text: "キャン".into()
            })
    );
}
