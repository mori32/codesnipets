//! Dependency-free Phase 2 benchmark harness.

use std::time::Instant;

use ai_engine_core::KanaKanjiEngine;

fn main() {
    let engine = KanaKanjiEngine::new();
    let inputs = [
        "にほんごへんかん",
        "かなかんじへんかん",
        "konnichihasekai",
        "とうきょうだいがく",
    ];
    let iterations = 10_000;
    let mut samples = Vec::with_capacity(iterations);
    let mut result_count = 0_usize;

    for iteration in 0..iterations {
        let input = inputs[iteration % inputs.len()];
        let start = Instant::now();
        result_count += engine.convert(input, 5).len();
        samples.push(start.elapsed());
    }

    samples.sort_unstable();
    let total = samples.iter().sum::<std::time::Duration>();
    let p50 = samples[iterations / 2];
    let p95 = samples[iterations * 95 / 100];
    let average = total / 10_000;
    println!("iterations: {iterations}");
    println!("results: {result_count}");
    println!("average: {average:?}");
    println!("p50: {p50:?}");
    println!("p95: {p95:?}");
}
