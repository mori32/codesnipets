//! Generate the initial CRF seed model from the canonical word-ID n-grams.
#![allow(
    clippy::cast_precision_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ai_engine_crf::{Feature, Model, Result, output};
use serde_json::json;

const DEFAULT_STATISTICS: &str = r"E:\lmwork-target\wiki-ja-wordid-ngrams";
const DEFAULT_SYSTEM_DICTIONARY: &str = r"C:\Program Files\MeCab\dic\ipadic";
const DEFAULT_ADDITIONS: &str = r"D:\Git\codesnipets\aiEngine\dictionary\additions";
const DEFAULT_TARGET: &str = r"E:\lmwork-target\crf-initial-model";
const DEFAULT_WORK: &str = r"E:\lmwork-int\crf-initial-model";
const BIGRAM_LIMIT: usize = 1_500_000;
const TRIGRAM_LIMIT: usize = 3_000_000;
const CROSS_BIGRAM_LIMIT: usize = 500_000;
const AFFIX_POS_LIMIT: usize = 100;

struct Options {
    statistics: PathBuf,
    system_dictionary: PathBuf,
    additions: PathBuf,
    output: PathBuf,
    work: PathBuf,
}

fn options() -> Result<Options> {
    let mut options = Options {
        statistics: DEFAULT_STATISTICS.into(),
        system_dictionary: DEFAULT_SYSTEM_DICTIONARY.into(),
        additions: DEFAULT_ADDITIONS.into(),
        output: DEFAULT_TARGET.into(),
        work: DEFAULT_WORK.into(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "Usage: build-initial-model [--statistics DIR] [--system-dictionary DIR]\n\
                 [--additions DIR] [--output NEW_DIR] [--work NEW_DIR]\n\
                 Uses top 1,500,000 word bigrams and top 3,000,000 word trigrams,\n\
                 top 500,000 POS-word and word-POS bigrams, and 100 refined affix POS values.\n\
                 Refuses to overwrite either output directory."
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--statistics" => options.statistics = value.into(),
            "--system-dictionary" => options.system_dictionary = value.into(),
            "--additions" => options.additions = value.into(),
            "--output" => options.output = value.into(),
            "--work" => options.work = value.into(),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    Ok(options)
}

fn write_new(path: &Path, content: &str) -> Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(content.as_bytes())?;
    writer.flush()?;
    Ok(())
}

fn readme(status: &str, statistics: &Path, system: &Path, additions: &Path) -> String {
    format!(
        "# CRF初期モデル生成\n\n\
         - **目的・状態:** POS/単語/読み長の初期スコアを作成する。{status}\n\
         - **ソース:** `{}`、`{}`、`{}`。単語/POSは MeCab標準辞書と additions を統合し、\
         語彙表とID順1/2/3-gramは頻度統計ディレクトリから読む。\n\
         - **初期化:** word-bigram top {BIGRAM_LIMIT}、word-trigram top {TRIGRAM_LIMIT}。\
         接頭語・接尾語候補からraw unigram top {AFFIX_POS_LIMIT}語を語ごとの細分化POSにする。\
         `号・時・日・数・級` と助数詞等は除外し、既存POSは統合しない。\
         POS-WORD/WORD-POSは各top {CROSS_BIGRAM_LIMIT} 組、POS-bigramをbaseに線形補間する。\
         POS-trigramは使わない。読み長bi/triは9×9/9×9×9を全件出力する。\
         重みはword-bi baseline 0.9、word-tri経験側0.1、cross-bi経験側0.1、読み長bi/tri 0.15。\
         未観測イベントは経験側確率をbaseline確率に置き、\
         補正0を得る。\n\
         - **生成物:** `model.json` は初期CRFモデル、`vocabulary.jsonl`/`pos.tsv` はローカルID表、\
         `pos-token.tsv` はPOS用合成単語ID、`affix-pos-selection.tsv` は選択100語。\
         `word-cost.tsv`/`pos-bigram.tsv`/`word-bigram.tsv`/`word-trigram.tsv`/\
         `pos-word-bigram.tsv`/`word-pos-bigram.tsv`/`reading-length-bigram.tsv`/\
         `reading-length-trigram.tsv` は初期スコア表。POS-trigram表は作らない。\
         表記種別n-gramは出力しない。\n\
         - **次の利用先:** `ai_engine_crf --model-in <このフォルダ>\\model.json`。\n\
         - **削除可否:** 再生成可能。再実行は既存出力を上書きしないため、\
         削除前に利用側の依存がないことを確認する。\n",
        statistics.display(),
        system.display(),
        additions.display()
    )
}

#[allow(clippy::too_many_lines)]
fn run() -> Result<()> {
    let options = options()?;
    for path in [
        &options.statistics,
        &options.system_dictionary,
        &options.additions,
    ] {
        if !path.is_dir() {
            return Err(format!("source directory does not exist: {}", path.display()).into());
        }
    }
    if options.output.exists() || options.work.exists() {
        return Err(format!(
            "refusing to overwrite output/work directory: target={} work={}",
            options.output.display(),
            options.work.display()
        )
        .into());
    }

    // Create both directories only after all preflight checks and never reuse a run.
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = options.work.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&options.output)?;
    fs::create_dir(&options.work)?;
    write_new(
        &options.output.join("README.md"),
        &readme(
            "作業中（生成完了前のファイルは未検証）。",
            &options.statistics,
            &options.system_dictionary,
            &options.additions,
        ),
    )?;
    write_new(
        &options.work.join("README.md"),
        &format!(
            "# CRF初期モデル作業領域\n\n\
             - **目的・状態:** `{}` の生成中間情報・実行記録。再生成可能。\n\
             - **ソース:** `{}`、`{}`、`{}`。\n\
             - **生成物:** 大規模な中間データは作らず、進捗・検証ログのみ置く。\n\
             - **次の利用先:** 検証済み初期モデルは `{}`。\n\
             - **削除可否:** 実行終了後、target成果物の検証を確認して個別に削除可能。\n",
            options.work.display(),
            options.statistics.display(),
            options.system_dictionary.display(),
            options.additions.display(),
            options.output.display()
        ),
    )?;

    eprintln!(
        "initial-model-build-start statistics={} output={} work={} bigram_limit={} trigram_limit={}",
        options.statistics.display(),
        options.output.display(),
        options.work.display(),
        BIGRAM_LIMIT,
        TRIGRAM_LIMIT
    );
    let started = Instant::now();
    let training = ai_engine_dict::training_seeds_with_crf_features(
        &options.statistics,
        &options.system_dictionary,
        &options.additions,
        BIGRAM_LIMIT,
        TRIGRAM_LIMIT,
    )?;
    let pos_word_seed_count = training.pos_word_bigrams.len();
    let word_pos_seed_count = training.word_pos_bigrams.len();
    let affix_pos_selection_count = training.affix_pos_selections.len();
    eprintln!(
        "seed-scan-complete words={} pos={} word_bigrams={} word_trigrams={} pos_word_bigrams={} word_pos_bigrams={} affix_pos_selections={} pos_trigrams=disabled elapsed_seconds={:.1}",
        training.seeds.words.len(),
        training.seeds.pos_names.len(),
        training.bigrams.len(),
        training.trigrams.len(),
        pos_word_seed_count,
        word_pos_seed_count,
        affix_pos_selection_count,
        started.elapsed().as_secs_f64()
    );
    let model = Model::from_crf_ngrams(training, 1e-8, 0.9, 0.1, 0.3, 0.15)?;
    let before = output::fingerprint(&model);
    output::export(&model, &options.output)?;
    let reloaded = output::load(&options.output.join("model.json"))?;
    let after = output::fingerprint(&reloaded);
    if before != after {
        return Err("saved initial model differs from the in-memory model".into());
    }

    let names = [
        "model.json",
        "vocabulary.jsonl",
        "pos.tsv",
        "word-cost.tsv",
        "pos-bigram.tsv",
        "word-bigram.tsv",
        "word-trigram.tsv",
        "pos-token.tsv",
        "affix-pos-selection.tsv",
        "pos-word-bigram.tsv",
        "word-pos-bigram.tsv",
        "reading-length-bigram.tsv",
        "reading-length-trigram.tsv",
    ];
    let files = names
        .iter()
        .map(|name| {
            let path = options.output.join(name);
            let metadata = fs::metadata(&path)?;
            Ok(json!({"name": name, "bytes": metadata.len()}))
        })
        .collect::<std::io::Result<Vec<_>>>()?;
    let manifest = json!({
        "status": "complete",
        "statistics": options.statistics,
        "system_dictionary": options.system_dictionary,
        "user_additions": options.additions,
        "target": options.output,
        "bigram_limit": BIGRAM_LIMIT,
        "trigram_limit": TRIGRAM_LIMIT,
        "pos_word_bigram_limit": CROSS_BIGRAM_LIMIT,
        "word_pos_bigram_limit": CROSS_BIGRAM_LIMIT,
        "affix_pos_limit": AFFIX_POS_LIMIT,
        "excluded_affix_surfaces": ["号", "時", "日", "数", "級"],
        "word_bigram_baseline_weight": 0.9,
        "word_trigram_empirical_weight": 0.1,
        "cross_bigram_empirical_weight": 0.1,
        "pos_trigrams_enabled": false,
        "reading_length_empirical_weight": 0.15,
        "absent_event_rule": "set empirical probability equal to baseline, yielding correction zero",
        "word_count": model.words.len(),
        "pos_count": model.pos_names.len() - 1,
        "word_bigram_count": model.initial_bigrams.iter().filter(|(key, _)| matches!(key, Feature::Bigram(..))).count(),
        "pos_word_bigram_count": model.initial_bigrams.iter().filter(|(key, _)| matches!(key, Feature::PosWord(..))).count(),
        "word_pos_bigram_count": model.initial_bigrams.iter().filter(|(key, _)| matches!(key, Feature::WordPos(..))).count(),
        "word_trigram_count": model.initial_trigrams.len(),
        "affix_pos_selection_count": model.affix_pos_selections.len(),
        "observed_pos_trigram_count": 0,
        "reading_length_bigram_cells": 81,
        "reading_length_trigram_cells": 729,
        "synthetic_pos_token_count": model.pos_token_ids.len(),
        "checkpoint_roundtrip_fingerprint": after,
        "files": files,
        "elapsed_seconds": started.elapsed().as_secs_f64()
    });
    let manifest_path = options.output.join("manifest.json");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&manifest_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &manifest)?;
    writeln!(writer)?;
    writer.flush()?;
    fs::write(
        options.output.join("README.md"),
        readme(
            "完了・model.jsonの保存/再読込fingerprint検証済み。",
            &options.statistics,
            &options.system_dictionary,
            &options.additions,
        ),
    )?;
    fs::write(
        options.work.join("README.md"),
        format!(
            "# CRF初期モデル作業領域\n\n\
             - **目的・状態:** `{}` の初期データ生成完了。\n\
             - **ソース:** `{}`、`{}`、`{}`。\n\
             - **生成物:** 大規模な中間データはなし。target側manifestに件数・サイズ・\
             checkpoint保存/再読込fingerprint結果を記録。\n\
             - **次の利用先:** `{}`。\n\
             - **削除可否:** target成果物の検証と利用側切替を確認後、作業領域は削除可能。\n",
            options.work.display(),
            options.statistics.display(),
            options.system_dictionary.display(),
            options.additions.display(),
            options.output.display()
        ),
    )?;
    eprintln!(
        "initial-model-build-complete output={} words={} pos={} word_bigrams={} word_trigrams={} pos_word_bigrams={} word_pos_bigrams={} affix_pos_selections={} pos_trigrams=disabled readlen_bi=81 readlen_tri=729 elapsed_seconds={:.1}",
        options.output.display(),
        model.words.len(),
        model.pos_names.len() - 1,
        model
            .initial_bigrams
            .iter()
            .filter(|(key, _)| matches!(key, Feature::Bigram(..)))
            .count(),
        model.initial_trigrams.len(),
        model
            .initial_bigrams
            .iter()
            .filter(|(key, _)| matches!(key, Feature::PosWord(..)))
            .count(),
        model
            .initial_bigrams
            .iter()
            .filter(|(key, _)| matches!(key, Feature::WordPos(..)))
            .count(),
        model.affix_pos_selections.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("initial-model-build-error: {error}");
        std::process::exit(1);
    }
}
