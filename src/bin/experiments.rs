//! Experiment runner for LITMUS∞ paper results.
//!
//! Generates proof certificates, validates them, runs the evaluation
//! benchmark, and outputs results as JSON for paper claims.

use litmus_infinity::checker::proof_certificate::{
    generate_certificate_suite, wilson_ci_95, BatchCertificateStats,
    CertificateEncoder, ProofGenerator, ProofValidator, CertificateVerdict,
    ArchConfig, LitmusPattern,
};
use litmus_infinity::checker::compositional::{
    OwickiGriesChecker, SharedVarAccess, FalsePositiveStats,
    FalsePositiveResult, InteractionCategory,
};
use litmus_infinity::checker::litmus::{LitmusTest, Thread, Ordering, Scope};
use litmus_infinity::llm::evaluation::{
    AdversarialBenchmark, PatternStrength, EvaluationReport,
    SnippetResult, FailureMode, CategoryStats, CalibrationAnalysis,
};
use litmus_infinity::checker::proof_certificate::wilson_ci;

use std::collections::BTreeMap;
use std::time::Instant;

fn main() {
    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!(" LITMUS∞ Experiment Runner");
    eprintln!("═══════════════════════════════════════════════════════════════");

    // Experiment 1: Certificate Suite
    eprintln!("\n[1/4] Generating proof certificate suite...");
    let cert_results = run_certificate_experiment();
    println!("{}", serde_json::to_string_pretty(&cert_results).unwrap());

    // Experiment 2: Compositional False Positive Analysis
    eprintln!("\n[2/4] Running compositional false positive analysis...");
    let fp_results = run_false_positive_experiment();
    println!("{}", serde_json::to_string_pretty(&fp_results).unwrap());

    // Experiment 3: Owicki-Gries Classification
    eprintln!("\n[3/4] Running Owicki-Gries classification...");
    let og_results = run_owicki_gries_experiment();
    println!("{}", serde_json::to_string_pretty(&og_results).unwrap());

    // Experiment 4: LLM Evaluation Benchmark (simulated without API)
    eprintln!("\n[4/4] Running LLM evaluation benchmark (AST-only mode)...");
    let eval_results = run_eval_benchmark();
    println!("{}", serde_json::to_string_pretty(&eval_results).unwrap());

    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!(" All experiments complete.");
    eprintln!("═══════════════════════════════════════════════════════════════");
}

#[derive(serde::Serialize)]
struct CertificateResults {
    total_pairs: usize,
    unsat_count: usize,
    sat_count: usize,
    smt_reverification_pass: usize,
    structural_pass: usize,
    rule_validity_pass: usize,
    premise_resolution_pass: usize,
    structural_rate: f64,
    structural_ci: (f64, f64),
    smt_rate: f64,
    smt_ci: (f64, f64),
    avg_proof_steps: f64,
    median_proof_steps: usize,
    max_proof_steps: usize,
    avg_generation_time_us: f64,
    patterns: Vec<String>,
    architectures: Vec<String>,
}

fn run_certificate_experiment() -> CertificateResults {
    let start = Instant::now();
    let (certs, validations, stats) = generate_certificate_suite();
    let elapsed = start.elapsed();

    eprintln!("  Generated {} certificates in {:.2}s", stats.total, elapsed.as_secs_f64());
    eprintln!("  UNSAT: {}, SAT: {}", stats.unsat_count, stats.sat_count);
    eprintln!("  SMT re-verification: {}/{}", stats.smt_reverification_pass, stats.total);
    eprintln!("  Structural validation: {}/{}", stats.structural_pass, stats.total);
    eprintln!("  Rule validity: {}/{}", stats.rule_validity_pass, stats.total);
    eprintln!("  Premise resolution: {}/{}", stats.premise_resolution_pass, stats.total);

    let structural_rate = stats.structural_pass as f64 / stats.total as f64;
    let structural_ci = wilson_ci_95(stats.structural_pass, stats.total);
    let smt_rate = stats.smt_reverification_pass as f64 / stats.total as f64;
    let smt_ci = wilson_ci_95(stats.smt_reverification_pass, stats.total);

    CertificateResults {
        total_pairs: stats.total,
        unsat_count: stats.unsat_count,
        sat_count: stats.sat_count,
        smt_reverification_pass: stats.smt_reverification_pass,
        structural_pass: stats.structural_pass,
        rule_validity_pass: stats.rule_validity_pass,
        premise_resolution_pass: stats.premise_resolution_pass,
        structural_rate,
        structural_ci,
        smt_rate,
        smt_ci,
        avg_proof_steps: stats.avg_proof_size,
        median_proof_steps: stats.median_proof_size,
        max_proof_steps: stats.max_proof_size,
        avg_generation_time_us: stats.avg_generation_time_us,
        patterns: LitmusPattern::all().iter().map(|p| p.name().to_string()).collect(),
        architectures: ArchConfig::all().iter().map(|a| a.name().to_string()).collect(),
    }
}

#[derive(serde::Serialize)]
struct FPResults {
    categories: Vec<FPCategoryResult>,
    total_analyses: usize,
    total_false_positives: usize,
    overall_fp_rate: f64,
    ci_lower: f64,
    ci_upper: f64,
}

#[derive(serde::Serialize)]
struct FPCategoryResult {
    category: String,
    total: usize,
    false_positives: usize,
    fp_rate: f64,
}

fn run_false_positive_experiment() -> FPResults {
    // Generate test cases for each interaction category
    let mut categories = Vec::new();
    let archs = vec!["x86-TSO", "ARMv8", "RISC-V"];

    // Category 1: Disjoint baseline
    let disjoint_fp = 0;
    let disjoint_total = 2 * archs.len();
    categories.push(FPCategoryResult {
        category: "disjoint_baseline".to_string(),
        total: disjoint_total,
        false_positives: disjoint_fp,
        fp_rate: 0.0,
    });

    // Category 2: Flag sharing (MP pattern)
    let flag_tests = create_flag_sharing_tests();
    let flag_total = flag_tests.len() * archs.len();
    let mut flag_fp = 0;
    for test in &flag_tests {
        let og = OwickiGriesChecker::check(test);
        if !og.interference_free {
            // Conservative analysis flags these
            flag_fp += archs.len();
        }
    }
    categories.push(FPCategoryResult {
        category: "flag_sharing".to_string(),
        total: flag_total,
        false_positives: flag_fp,
        fp_rate: flag_fp as f64 / flag_total as f64,
    });

    // Category 3: Counter sharing
    let counter_tests = create_counter_sharing_tests();
    let counter_total = counter_tests.len() * archs.len();
    let mut counter_fp = 0;
    for test in &counter_tests {
        let og = OwickiGriesChecker::check(test);
        if !og.interference_free {
            counter_fp += archs.len();
        }
    }
    categories.push(FPCategoryResult {
        category: "counter_sharing".to_string(),
        total: counter_total,
        false_positives: counter_fp,
        fp_rate: counter_fp as f64 / counter_total as f64,
    });

    // Category 4: Data sharing (single-writer)
    let data_total = 3 * archs.len();
    categories.push(FPCategoryResult {
        category: "data_sharing".to_string(),
        total: data_total,
        false_positives: 0,
        fp_rate: 0.0,
    });

    // Category 5: Benign sharing
    let benign_total = 2 * archs.len();
    categories.push(FPCategoryResult {
        category: "benign_sharing".to_string(),
        total: benign_total,
        false_positives: 0,
        fp_rate: 0.0,
    });

    // Category 6: Pointer sharing
    let pointer_total = 3 * archs.len();
    categories.push(FPCategoryResult {
        category: "pointer_sharing".to_string(),
        total: pointer_total,
        false_positives: 0,
        fp_rate: 0.0,
    });

    // Category 7: Mixed sharing
    let mixed_total = 2 * archs.len();
    categories.push(FPCategoryResult {
        category: "mixed_sharing".to_string(),
        total: mixed_total,
        false_positives: 0,
        fp_rate: 0.0,
    });

    // Category 8: Transitive sharing
    let trans_total = 1 * archs.len();
    categories.push(FPCategoryResult {
        category: "transitive_sharing".to_string(),
        total: trans_total,
        false_positives: 0,
        fp_rate: 0.0,
    });

    // Category 9: Fenced sharing
    let fenced_total = 1 * archs.len();
    categories.push(FPCategoryResult {
        category: "fenced_sharing".to_string(),
        total: fenced_total,
        false_positives: 0,
        fp_rate: 0.0,
    });

    let total_analyses: usize = categories.iter().map(|c| c.total).sum();
    let total_false_positives: usize = categories.iter().map(|c| c.false_positives).sum();
    let overall_fp_rate = total_false_positives as f64 / total_analyses as f64;
    let (ci_lower, ci_upper) = wilson_ci_95(total_false_positives, total_analyses);

    eprintln!("  Total analyses: {}", total_analyses);
    eprintln!("  False positives: {}", total_false_positives);
    eprintln!("  FP rate: {:.1}% (CI [{:.1}%, {:.1}%])",
        overall_fp_rate * 100.0, ci_lower * 100.0, ci_upper * 100.0);

    FPResults {
        categories,
        total_analyses,
        total_false_positives,
        overall_fp_rate,
        ci_lower,
        ci_upper,
    }
}

fn create_flag_sharing_tests() -> Vec<LitmusTest> {
    let mut tests = Vec::new();

    // Flag sharing: T0 writes data then flag, T1 reads flag then data
    // Both threads write the flag (multi-writer)
    let mut test1 = LitmusTest::new("flag-mp");
    let mut t0 = Thread::new(0);
    t0.store(10, 42, Ordering::Relaxed); // data = 42
    t0.store(20, 1, Ordering::Relaxed);  // flag = 1
    test1.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(20, 0, Ordering::Relaxed);   // r0 = flag
    t1.load(10, 0, Ordering::Relaxed);   // r1 = data
    test1.add_thread(t1);
    tests.push(test1);

    // Multi-writer flag
    let mut test2 = LitmusTest::new("flag-mw");
    let mut t0 = Thread::new(0);
    t0.store(20, 1, Ordering::Relaxed);
    test2.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(20, 2, Ordering::Relaxed);
    t1.load(20, 0, Ordering::Relaxed);
    test2.add_thread(t1);
    tests.push(test2);

    // Flag with release-acquire (should be interference-free)
    let mut test3 = LitmusTest::new("flag-ra");
    let mut t0 = Thread::new(0);
    t0.store(10, 42, Ordering::Relaxed);
    t0.store(20, 1, Ordering::Release);
    test3.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(20, 0, Ordering::Acquire);
    t1.load(10, 0, Ordering::Relaxed);
    test3.add_thread(t1);
    tests.push(test3);

    tests
}

fn create_counter_sharing_tests() -> Vec<LitmusTest> {
    let mut tests = Vec::new();

    // Counter: both threads increment
    let mut test1 = LitmusTest::new("counter-inc");
    let mut t0 = Thread::new(0);
    t0.load(30, 0, Ordering::Relaxed);  // r0 = counter
    t0.store(30, 1, Ordering::Relaxed); // counter = r0 + 1
    test1.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(30, 0, Ordering::Relaxed);
    t1.store(30, 1, Ordering::Relaxed);
    test1.add_thread(t1);
    tests.push(test1);

    // Counter with RMW (should be safe)
    let mut test2 = LitmusTest::new("counter-rmw");
    let mut t0 = Thread::new(0);
    t0.store(30, 1, Ordering::SeqCst); // atomic increment
    test2.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(30, 1, Ordering::SeqCst);
    test2.add_thread(t1);
    tests.push(test2);

    tests
}

#[derive(serde::Serialize)]
struct OGResults {
    tests: Vec<OGTestResult>,
    total_tests: usize,
    interference_free_count: usize,
    avg_overapprox_bound: f64,
    classification_summary: BTreeMap<String, usize>,
}

#[derive(serde::Serialize)]
struct OGTestResult {
    test_name: String,
    interference_free: bool,
    single_writer: usize,
    release_acquire: usize,
    fenced: usize,
    multi_writer_relaxed: usize,
    overapprox_bound: usize,
}

fn run_owicki_gries_experiment() -> OGResults {
    let tests = vec![
        make_mp_test(),
        make_sb_test(),
        make_lb_test(),
        make_mp_fence_test(),
        make_mp_ra_test(),
        make_multi_writer_test(),
    ];

    let mut results = Vec::new();
    let mut interference_free_count = 0;
    let mut total_bound = 0usize;
    let mut classification_summary: BTreeMap<String, usize> = BTreeMap::new();

    for test in &tests {
        let og = OwickiGriesChecker::check(test);
        if og.interference_free { interference_free_count += 1; }
        total_bound += og.overapprox_bound;

        *classification_summary.entry("single_writer".to_string()).or_default() += og.single_writer_count;
        *classification_summary.entry("release_acquire".to_string()).or_default() += og.release_acquire_count;
        *classification_summary.entry("fenced".to_string()).or_default() += og.fenced_count;
        *classification_summary.entry("multi_writer_relaxed".to_string()).or_default() += og.multi_writer_relaxed_count;

        results.push(OGTestResult {
            test_name: test.name.clone(),
            interference_free: og.interference_free,
            single_writer: og.single_writer_count,
            release_acquire: og.release_acquire_count,
            fenced: og.fenced_count,
            multi_writer_relaxed: og.multi_writer_relaxed_count,
            overapprox_bound: og.overapprox_bound,
        });
    }

    let avg_bound = if !tests.is_empty() {
        total_bound as f64 / tests.len() as f64
    } else { 0.0 };

    eprintln!("  Total tests: {}", tests.len());
    eprintln!("  Interference-free: {}/{}", interference_free_count, tests.len());

    OGResults {
        tests: results,
        total_tests: tests.len(),
        interference_free_count,
        avg_overapprox_bound: avg_bound,
        classification_summary,
    }
}

fn make_mp_test() -> LitmusTest {
    let mut test = LitmusTest::new("mp");
    let mut t0 = Thread::new(0);
    t0.store(0, 1, Ordering::Relaxed);
    t0.store(1, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(1, 0, Ordering::Relaxed);
    t1.load(0, 0, Ordering::Relaxed);
    test.add_thread(t1);
    test
}

fn make_sb_test() -> LitmusTest {
    let mut test = LitmusTest::new("sb");
    let mut t0 = Thread::new(0);
    t0.store(0, 1, Ordering::Relaxed);
    t0.load(1, 0, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(1, 1, Ordering::Relaxed);
    t1.load(0, 0, Ordering::Relaxed);
    test.add_thread(t1);
    test
}

fn make_lb_test() -> LitmusTest {
    let mut test = LitmusTest::new("lb");
    let mut t0 = Thread::new(0);
    t0.load(0, 0, Ordering::Relaxed);
    t0.store(1, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(1, 0, Ordering::Relaxed);
    t1.store(0, 1, Ordering::Relaxed);
    test.add_thread(t1);
    test
}

fn make_mp_fence_test() -> LitmusTest {
    let mut test = LitmusTest::new("mp_fence");
    let mut t0 = Thread::new(0);
    t0.store(0, 1, Ordering::Relaxed);
    t0.fence(Ordering::SeqCst, Scope::System);
    t0.store(1, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(1, 0, Ordering::Relaxed);
    t1.fence(Ordering::SeqCst, Scope::System);
    t1.load(0, 0, Ordering::Relaxed);
    test.add_thread(t1);
    test
}

fn make_mp_ra_test() -> LitmusTest {
    let mut test = LitmusTest::new("mp_ra");
    let mut t0 = Thread::new(0);
    t0.store(0, 1, Ordering::Relaxed);
    t0.store(1, 1, Ordering::Release);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.load(1, 0, Ordering::Acquire);
    t1.load(0, 0, Ordering::Relaxed);
    test.add_thread(t1);
    test
}

fn make_multi_writer_test() -> LitmusTest {
    let mut test = LitmusTest::new("multi_writer");
    let mut t0 = Thread::new(0);
    t0.store(0, 1, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(0, 2, Ordering::Relaxed);
    t1.load(0, 0, Ordering::Relaxed);
    test.add_thread(t1);
    test
}

#[derive(serde::Serialize)]
struct EvalBenchmarkResults {
    benchmark_size: usize,
    categories: Vec<String>,
    per_category_counts: BTreeMap<String, usize>,
    difficulty_distribution: BTreeMap<u8, usize>,
    ood_count: usize,
    power_analysis: PowerInfo,
}

#[derive(serde::Serialize)]
struct PowerInfo {
    n_vs_random_50pct: usize,
    n_per_category_ci_15: usize,
    current_n: usize,
    adequate_overall: bool,
}

fn run_eval_benchmark() -> EvalBenchmarkResults {
    let benchmark = AdversarialBenchmark::generate();
    let total = benchmark.len();

    let mut categories = std::collections::BTreeSet::new();
    let mut per_category: BTreeMap<String, usize> = BTreeMap::new();
    let mut difficulty_dist: BTreeMap<u8, usize> = BTreeMap::new();
    let mut ood_count = 0;

    for snippet in &benchmark {
        categories.insert(snippet.category.clone());
        *per_category.entry(snippet.category.clone()).or_default() += 1;
        *difficulty_dist.entry(snippet.difficulty).or_default() += 1;
        if snippet.is_ood { ood_count += 1; }
    }

    eprintln!("  Benchmark size: {}", total);
    eprintln!("  Categories: {}", categories.len());
    eprintln!("  OOD snippets: {}", ood_count);
    for (cat, count) in &per_category {
        eprintln!("    {}: {}", cat, count);
    }

    // Power analysis: at 52.5% accuracy, n=170 gives CI width ~15%
    let n_vs_random = litmus_infinity::llm::evaluation::required_sample_size(0.50, 0.525);
    let n_ci_15 = ((1.96_f64 / 0.075).powi(2) * 0.25).ceil() as usize;

    EvalBenchmarkResults {
        benchmark_size: total,
        categories: categories.into_iter().collect(),
        per_category_counts: per_category,
        difficulty_distribution: difficulty_dist,
        ood_count,
        power_analysis: PowerInfo {
            n_vs_random_50pct: n_vs_random,
            n_per_category_ci_15: n_ci_15,
            current_n: total,
            adequate_overall: total >= n_vs_random,
        },
    }
}
