//! Platform-independent kana-kanji conversion core.
//!
//! The Phase 1 implementation intentionally has no external dependencies.
//! It provides a small built-in lexicon, user-dictionary learning, and a
//! beam-bounded N-best lattice search. A larger dictionary and an LLM scorer
//! can be added without changing the public conversion result type.

mod dictionary;
mod engine;
mod normalize;
mod static_dictionary;

pub use dictionary::{Dictionary, Word};
pub use engine::{Candidate, KanaKanjiEngine, RerankedCandidate, rerank_candidates};
pub use normalize::to_hiragana;
#[cfg(test)]
pub use static_dictionary::MozcImportStats;
pub use static_dictionary::{ConnectionMatrix, StaticDictionary};

/// Identifies the core crate at runtime for the CLI and FFI smoke tests.
#[must_use]
pub const fn engine_name() -> &'static str {
    "ai_engine_core"
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{
        Candidate, KanaKanjiEngine, StaticDictionary, engine_name, rerank_candidates, to_hiragana,
    };

    #[test]
    fn exposes_a_stable_core_identity() {
        assert_eq!(engine_name(), "ai_engine_core");
    }

    #[test]
    fn normalizes_common_romaji_and_katakana() {
        assert_eq!(to_hiragana("konnichihasekai"), "こんにちはせかい");
        assert_eq!(to_hiragana("gakkou"), "がっこう");
        assert_eq!(to_hiragana("カタカナ"), "かたかな");
    }

    #[test]
    fn returns_an_expected_top_candidate() {
        let engine = KanaKanjiEngine::new();
        let candidates = engine.convert("にほんごへんかん", 5);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].text, "日本語変換");
        assert_eq!(candidates[0].reading, "にほんごへんかん");
        assert!(candidates[0].segments.len() >= 2);
    }

    #[test]
    fn keeps_unknown_input_convertible() {
        let engine = KanaKanjiEngine::new();
        let candidates = engine.convert("あいうえお", 1);
        assert_eq!(candidates[0].text, "あいうえお");
    }

    #[test]
    fn learned_user_word_is_preferred_and_persisted() {
        let mut engine = KanaKanjiEngine::new();
        engine.add_word("とうきょう", "東京都", 100);
        let candidate = engine.convert("toukyou", 1).remove(0);
        assert_eq!(candidate.text, "東京都");
        engine.select(&candidate);

        let path = std::env::temp_dir().join(format!(
            "ai-engine-test-{}-dictionary.tsv",
            std::process::id()
        ));
        engine
            .save_user_dictionary(&path)
            .expect("save user dictionary");

        let mut restored = KanaKanjiEngine::new();
        restored
            .load_user_dictionary(&path)
            .expect("load user dictionary");
        assert_eq!(restored.convert("とうきょう", 1)[0].text, "東京都");
        fs::remove_file(path).expect("remove temporary dictionary");
    }

    #[test]
    fn loads_external_dictionary_without_persisting_it_as_user_data() {
        let dictionary_path = std::env::temp_dir().join(format!(
            "ai-engine-test-{}-external.tsv",
            std::process::id()
        ));
        let user_path =
            std::env::temp_dir().join(format!("ai-engine-test-{}-user.tsv", std::process::id()));
        fs::write(&dictionary_path, "ぷろとたいぷ\tプロトタイプ\t40\t0\n")
            .expect("write external dictionary");

        let mut engine = KanaKanjiEngine::new();
        engine
            .load_dictionary(&dictionary_path)
            .expect("load external dictionary");
        assert_eq!(engine.convert("ぷろとたいぷ", 1)[0].text, "プロトタイプ");
        engine
            .save_user_dictionary(&user_path)
            .expect("save user dictionary");
        assert_eq!(
            fs::read_to_string(&user_path).expect("read user dictionary"),
            "# aiEngine user dictionary v1\n"
        );

        fs::remove_file(dictionary_path).expect("remove external dictionary");
        fs::remove_file(user_path).expect("remove user dictionary");
    }

    #[test]
    fn loads_static_dictionary_without_persisting_it_as_user_data() {
        let prefix = format!("ai-engine-test-{}-static", std::process::id());
        let source_path = std::env::temp_dir().join(format!("{prefix}.tsv"));
        let static_path = std::env::temp_dir().join(format!("{prefix}.aidict"));
        let user_path = std::env::temp_dir().join(format!("{prefix}-user.tsv"));
        fs::write(&source_path, "ぷろとたいぷ\tプロトタイプ\t40\t0\n")
            .expect("write static source");
        StaticDictionary::compile_tsv(&source_path, &static_path).expect("compile static source");

        let mut engine = KanaKanjiEngine::new();
        engine
            .load_static_dictionary(&static_path)
            .expect("load static dictionary");
        assert_eq!(engine.convert("ぷろとたいぷ", 1)[0].text, "プロトタイプ");
        engine
            .save_user_dictionary(&user_path)
            .expect("save user dictionary");
        assert_eq!(
            fs::read_to_string(&user_path).expect("read user dictionary"),
            "# aiEngine user dictionary v1\n"
        );

        fs::remove_file(source_path).expect("remove static source");
        fs::remove_file(static_path).expect("remove static dictionary");
        fs::remove_file(user_path).expect("remove user dictionary");
    }

    #[test]
    fn evaluator_score_can_change_candidate_order() {
        let engine = KanaKanjiEngine::new();
        let candidates = engine.convert("とうきょう", 2);
        assert!(!candidates.is_empty());
        let mut alternatives = candidates;
        alternatives.push(Candidate {
            text: "東京駅".to_owned(),
            reading: "とうきょう".to_owned(),
            score: -20,
            segments: Vec::new(),
        });
        let ranked = rerank_candidates(&alternatives, &[0.0, 100.0], 1.0);
        assert_eq!(ranked[0].candidate.text, "東京駅");
    }
}
