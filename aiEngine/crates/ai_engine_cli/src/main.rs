//! Command-line experiment harness for aiEngine.

use std::env;
use std::error::Error;
use std::path::PathBuf;

use ai_engine_core::KanaKanjiEngine;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut input = None;
    let mut limit = 5_usize;
    let mut selected = None;
    let mut dictionary_path = PathBuf::from("user_dictionary.tsv");
    let mut external_dictionaries = Vec::new();
    let mut static_dictionaries = Vec::new();
    let mut arguments = env::args().skip(1);

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--candidates" => {
                limit = arguments
                    .next()
                    .ok_or("--candidates requires a positive integer")?
                    .parse()?;
                if limit == 0 {
                    return Err("--candidates must be greater than zero".into());
                }
            }
            "--select" => {
                selected = Some(
                    arguments
                        .next()
                        .ok_or("--select requires a candidate number")?
                        .parse::<usize>()?,
                );
            }
            "--user-dict" => {
                dictionary_path = arguments
                    .next()
                    .ok_or("--user-dict requires a file path")?
                    .into();
            }
            "--dictionary" => {
                external_dictionaries.push(PathBuf::from(
                    arguments
                        .next()
                        .ok_or("--dictionary requires a file path")?,
                ));
            }
            "--static-dictionary" => {
                static_dictionaries.push(PathBuf::from(
                    arguments
                        .next()
                        .ok_or("--static-dictionary requires a file path")?,
                ));
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option: {value}").into());
            }
            value if input.is_none() => input = Some(value.to_owned()),
            _ => return Err("only one input reading is accepted".into()),
        }
    }

    let input = input.ok_or("missing input; try --help")?;
    let mut engine = KanaKanjiEngine::new();
    for dictionary in external_dictionaries {
        engine.load_dictionary(dictionary)?;
    }
    for dictionary in static_dictionaries {
        engine.load_static_dictionary(dictionary)?;
    }
    if dictionary_path.exists() {
        engine.load_user_dictionary(&dictionary_path)?;
    }
    let candidates = engine.convert(&input, limit);
    for (index, candidate) in candidates.iter().enumerate() {
        let segments = candidate
            .segments
            .iter()
            .map(|word| word.surface.as_str())
            .collect::<Vec<_>>()
            .join(" / ");
        println!(
            "{index}: {}  score={}  segments=[{segments}]",
            candidate.text, candidate.score
        );
    }

    if let Some(index) = selected {
        let candidate = candidates
            .get(index)
            .ok_or("--select is outside the displayed candidate range")?;
        engine.select(candidate);
        engine.save_user_dictionary(&dictionary_path)?;
        println!("selected: {}", candidate.text);
    }
    Ok(())
}

fn print_help() {
    println!(
        "ai-engine <romaji-or-kana> [--candidates N] [--select N] [--dictionary PATH] [--static-dictionary PATH] [--user-dict PATH]"
    );
    println!("examples:");
    println!("  ai-engine nihongohenkan");
    println!("  ai-engine にほんごへんかん --candidates 5");
    println!("  ai-engine toukyou --select 0");
}
