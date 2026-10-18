//! Offline compiler for the provisional P5 static dictionary format.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ai_engine_core::StaticDictionary;

use ai_engine_dict as sdict;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut arguments = env::args().skip(1);
    match arguments.next().as_deref() {
        Some("compile") => {
            let source = PathBuf::from(
                arguments
                    .next()
                    .ok_or("compile requires a source TSV path")?,
            );
            let destination = PathBuf::from(
                arguments
                    .next()
                    .ok_or("compile requires a destination .aidict path")?,
            );
            if arguments.next().is_some() {
                return Err("compile accepts exactly a source and destination path".into());
            }
            StaticDictionary::compile_tsv(&source, &destination)?;
            println!(
                "compiled: {} -> {}",
                source.display(),
                destination.display()
            );
            Ok(())
        }
        Some("compile-mozc") => {
            let source_directory = PathBuf::from(
                arguments
                    .next()
                    .ok_or("compile-mozc requires a dictionary_oss directory")?,
            );
            let destination = PathBuf::from(
                arguments
                    .next()
                    .ok_or("compile-mozc requires a destination .aidict path")?,
            );
            if arguments.next().is_some() {
                return Err(
                    "compile-mozc accepts exactly a source directory and destination path".into(),
                );
            }
            let temporary_connection = prepare_mozc_connection(&source_directory)?;
            let result = if let Some(connection_path) = temporary_connection.as_ref() {
                StaticDictionary::compile_mozc_directory_with_connection(
                    &source_directory,
                    connection_path,
                    &destination,
                )
            } else {
                StaticDictionary::compile_mozc_directory(&source_directory, &destination)
            };
            if let Some(connection_path) = temporary_connection {
                let _ = fs::remove_file(connection_path);
            }
            let stats = result?;
            println!(
                "compiled Mozc: files={} rows={} unique={} matrix={} -> {}",
                stats.files,
                stats.rows,
                stats.unique_entries,
                stats.matrix_dimension,
                destination.display()
            );
            Ok(())
        }
        Some(command @ ("compile-sdict" | "dump-sdict")) => {
            run_sdict_command(command, &mut arguments)
        }
        Some("--help" | "-h") | None => {
            print_help();
            Ok(())
        }
        Some(command) => Err(format!("unknown command: {command}").into()),
    }
}

fn run_sdict_command(
    command: &str,
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn Error>> {
    match command {
        "compile-sdict" => {
            let input_dir = PathBuf::from(
                arguments
                    .next()
                    .ok_or("compile-sdict requires an input directory")?,
            );
            let destination = PathBuf::from(
                arguments
                    .next()
                    .ok_or("compile-sdict requires a destination .sdict path")?,
            );
            let mut system_dictionary = None;
            let mut user_additions = None;
            let mut ngram_cutoff = sdict::DEFAULT_NGRAM_CUTOFF;
            while let Some(option) = arguments.next() {
                match option.as_str() {
                    "--system-dictionary" => {
                        system_dictionary = Some(PathBuf::from(
                            arguments
                                .next()
                                .ok_or("--system-dictionary requires a path")?,
                        ));
                    }
                    "--user-additions" => {
                        user_additions = Some(PathBuf::from(
                            arguments.next().ok_or("--user-additions requires a path")?,
                        ));
                    }
                    "--ngram-cutoff" => {
                        ngram_cutoff = arguments
                            .next()
                            .ok_or("--ngram-cutoff requires a count")?
                            .parse()?;
                    }
                    _ => return Err(format!("unknown compile-sdict option: {option}").into()),
                }
            }
            let system_dictionary = system_dictionary
                .ok_or("compile-sdict requires --system-dictionary <MeCab dicdir>")?;
            let user_additions = user_additions
                .ok_or("compile-sdict requires --user-additions <addition source directory>")?;
            sdict::compile(
                &input_dir,
                &destination,
                &system_dictionary,
                &user_additions,
                ngram_cutoff,
            )?;
            println!("compiled aiEngine sdict: {}", destination.display());
        }
        "dump-sdict" => {
            let source = PathBuf::from(
                arguments
                    .next()
                    .ok_or("dump-sdict requires a source .sdict path")?,
            );
            let output = PathBuf::from(
                arguments
                    .next()
                    .ok_or("dump-sdict requires an output directory")?,
            );
            if arguments.next().is_some() {
                return Err("dump-sdict accepts exactly two paths".into());
            }
            sdict::dump(&source, &output)?;
            println!("dumped aiEngine sdict: {}", output.display());
        }
        _ => return Err("unsupported sdict command".into()),
    }
    Ok(())
}

/// Returns a temporary plain-text connection matrix when Mozc only provides
/// `connection.deflate`. The official Mozc source tree ships
/// `build_tools/zlib_util.py`, so the importer can use an already-installed
/// Python without adding a Rust compression dependency to the provisional
/// compiler.
fn prepare_mozc_connection(source_directory: &Path) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let text_path = source_directory.join("connection_single_column.txt");
    if text_path.is_file() {
        return Ok(None);
    }

    let deflate_path = source_directory.join("connection.deflate");
    if !deflate_path.is_file() {
        return Ok(None);
    }

    let source_root = source_directory
        .parent()
        .and_then(Path::parent)
        .ok_or("cannot locate the Mozc source root from dictionary_oss")?;
    let utility = source_root.join("build_tools").join("zlib_util.py");
    if !utility.is_file() {
        return Err(format!(
            "{} exists, but Mozc build_tools/zlib_util.py was not found at {}",
            deflate_path.display(),
            utility.display()
        )
        .into());
    }

    let temporary_path = std::env::temp_dir().join(format!(
        "ai-engine-mozc-connection-{}.txt",
        std::process::id()
    ));
    let launchers = [
        ("python", Vec::<&str>::new()),
        ("py", vec!["-3"]),
        ("python3", Vec::<&str>::new()),
    ];
    let mut last_error = None;
    for (executable, prefix) in launchers {
        let status = Command::new(executable)
            .args(prefix)
            .arg(&utility)
            .arg("decompress")
            .arg(&deflate_path)
            .arg(&temporary_path)
            .status();
        match status {
            Ok(status) if status.success() && temporary_path.is_file() => {
                return Ok(Some(temporary_path));
            }
            Ok(status) => {
                last_error = Some(format!("{executable} exited with {status}"));
            }
            Err(error) => {
                last_error = Some(format!("{executable}: {error}"));
            }
        }
    }

    let _ = fs::remove_file(&temporary_path);
    Err(format!(
        "could not decompress {}; install Python or provide connection_single_column.txt ({})",
        deflate_path.display(),
        last_error.unwrap_or_else(|| "no Python launcher was available".to_owned())
    )
    .into())
}

fn print_help() {
    println!("ai-engine-dict compile SOURCE.tsv DESTINATION.aidict");
    println!("ai-engine-dict compile-mozc DICTIONARY_OSS_DIR DESTINATION.aidict");
    println!(
        "ai-engine-dict compile-sdict INPUT_DIR DESTINATION.sdict --system-dictionary MECAB_DICDIR --user-additions ADDITION_DIR [--ngram-cutoff COUNT]"
    );
    println!("ai-engine-dict dump-sdict SOURCE.sdict OUTPUT_DIR");
    println!("  compile-mozc auto-expands connection.deflate with Mozc's Python helper");
    println!("example:");
    println!(
        "  cargo run -p ai_engine_dict -- compile dictionaries\\phase2.tsv dictionaries\\phase5.aidict"
    );
}
