# LITMUS∞ — API Reference

All features listed below are implemented in Rust. Build with `cargo build`.

## CLI Commands

### Verify a litmus test against a memory model
```bash
cargo run --bin litmus-cli -- verify --file TEST.toml --model SC    # From file
cargo run --bin litmus-cli -- verify -t sb --model TSO              # Built-in test
```
Supported models: `SC`, `TSO`, `PSO`, `ARM`, `RISC-V`

### Check (parse-only validation)
```bash
cargo run --bin litmus-cli -- check TEST.toml
```

### Fence recommendations across all architectures
```bash
cargo run --bin litmus-cli -- fence-advise --file TEST.toml
```

### Symmetry compression analysis
```bash
cargo run --bin litmus-cli -- compress --file TEST.toml
```

### Generate a starter template
```bash
cargo run --bin litmus-cli -- init-test -o my_test.toml [--template sb|mp|lb|iriw|fence]
```

### Other commands
```bash
cargo run --bin litmus-cli -- models              # List supported memory models
cargo run --bin litmus-cli -- diff SC TSO          # Diff two models
cargo run --bin litmus-cli -- list-patterns        # List built-in patterns
cargo run --bin litmus-cli -- portability-check -p spinlock  # Check pattern portability
cargo run --bin litmus-cli -- benchmark -o results/         # Full benchmark suite
```

## TOML Test File Format

```toml
name = "MyTest"

[locations]
x = 0
y = 0

[[threads]]
ops = ["W(x, 1)", "R(y) r0"]

[[threads]]
ops = ["W(y, 1)", "R(x) r1"]

[forbidden]
x = 0
y = 0
```

**Operations:** `W(var, value)` — write, `R(var) reg` — read, `fence` — memory barrier.

JSON format is also supported (same structure with JSON syntax). The parser also accepts herd7, LISA, and PTX formats.

### Programmatic file loading
```rust
use litmus_infinity::frontend::parser::LitmusParser;
let parser = LitmusParser::new();
let test = parser.parse(&std::fs::read_to_string("my_test.toml").unwrap()).unwrap();
// Also: parser.parse_toml(), parser.parse_json(), parser.parse_herd(), parser.parse_simple()
```

---

## `litmus-experiments` — Experiment Runner

Generates proof certificates, runs false positive analysis, Owicki-Gries classification, and benchmark statistics. Results saved to `data/experiment_results.json`.

```bash
cargo run --bin litmus-experiments
```

---

## Proof Certificate System (`src/checker/proof_certificate.rs`)

### Types

- `ProofStep` — Individual Alethe-format proof step (Assume, Resolution, Rewrite, Let, Subproof)
- `ProofCertificate` — Complete certificate: steps, verdict (SAT/UNSAT), generation time, validation results
- `ProofGenerator` — DPLL-based proof generator for SAT encoding of memory models
- `ProofValidator` — 4-level validator (structural, rule validity, premise resolution, SMT re-verification)
- `CertificateEncoder` — Encodes litmus patterns (MP, SB, LB, IRIW, 2+2W, WRC) as Boolean formulas for 10 architectures
- `BatchCertificateGenerator` — Generates certificate suites across all pattern-architecture pairs

### Usage

```rust
use litmus_inf::checker::proof_certificate::*;

// Generate a single certificate
let encoder = CertificateEncoder::new();
let (vars, clauses) = encoder.encode_pattern("mp", "ARMv8");
let generator = ProofGenerator::new();
let cert = generator.generate(vars, clauses);

// Validate at all 4 levels
let validator = ProofValidator::new();
let validation = validator.validate(&cert);
assert!(validation.structural_valid);
assert!(validation.rule_valid);
assert!(validation.premises_valid);
assert!(validation.smt_valid);

// Batch generation: 14 patterns × 10 architectures = 140 certificates
let batch = BatchCertificateGenerator::new();
let suite = batch.generate_certificate_suite();
assert_eq!(suite.len(), 140);
```

### Key Functions

- `wilson_ci_95(successes, total)` — Wilson score 95% confidence interval
- `generate_certificate_suite()` — Generates all 140 pattern-model certificates

---

## Compositional Verifier (`src/checker/compositional.rs`)

### Types

- `CompositeVerifier` — Disjoint-variable compositional checker
- `OwickiGriesChecker` — Interference freedom checker for shared-variable programs
- `SharedVarAccess` — Classification: SingleWriter, ReleaseAcquire, Fenced, ReadOnly, MultiWriterRelaxed
- `OwickiGriesResult` — Result including interference_free flag, variable classifications, overapproximation bound
- `FalsePositiveStats` — Empirical FP analysis with Wilson CI
- `InteractionCategory` — Categories: DisjointBaseline, FlagSharing, CounterSharing, DataSharing, etc.

### Usage

```rust
use litmus_inf::checker::compositional::*;
use litmus_inf::checker::litmus::*;

// Create a litmus test
let mut test = LitmusTest::new("mp");
// ... add threads, set initial values ...

// Owicki-Gries interference freedom check
let og_checker = OwickiGriesChecker::new();
let result = og_checker.check_interference_freedom(&test);
println!("Interference-free: {}", result.interference_free);
println!("Overapprox bound: {}", result.overapproximation_bound);

// False positive analysis
let fp_stats = og_checker.analyze_false_positives();
println!("FP rate: {:.1}% (CI [{:.1}%, {:.1}%])",
    fp_stats.overall_rate * 100.0,
    fp_stats.ci_lower * 100.0,
    fp_stats.ci_upper * 100.0);
```

---

## LLM Evaluation Framework (`src/llm/evaluation.rs`)

### Types

- `AdversarialBenchmark` — 205-snippet benchmark across 8 categories
- `EvalSnippet` — Individual snippet with code, expected pattern, difficulty, OOD flag
- `FailureMode` — Correct, Conservative, Dangerous, NoMatch, RelatedMatch
- `PatternStrength` — Safety ordering for failure classification
- `ConfusionMatrix` — Multi-class confusion matrix
- `CalibrationAnalysis` — ECE, MCE, Brier score, reliability diagram bins
- `EvaluationReport` — Full report with per-category stats, power analysis, effect sizes

### Usage

```rust
use litmus_inf::llm::evaluation::*;

// Generate the benchmark
let benchmark = AdversarialBenchmark::generate();
assert_eq!(benchmark.snippets.len(), 205);

// Categories
let categories = benchmark.category_counts();
// application: 25, coherence: 25, dependency_patterns: 25, gpu: 25,
// kernel_patterns: 25, lock_free: 30, message_passing: 25, store_buffering: 25

// Difficulty distribution
let difficulty = benchmark.difficulty_distribution();
// 1: 2, 2: 20, 3: 74, 4: 76, 5: 33

// OOD count
let ood = benchmark.ood_count(); // 109
```

---

## Core Types (`src/checker/mod.rs`)

### `LitmusTest`

```rust
let mut test = LitmusTest::new("my_test");
test.set_initial(Address(0), Value(0));
let mut t0 = Thread::new(0);
t0.store(Address(0), Value(1), Ordering::Release);
t0.fence(Ordering::SeqCst, Scope::System);
test.add_thread(t0);
```

### `Thread`

- `store(addr, value, ordering)` — Memory store
- `load(addr, expected_value, ordering)` — Memory load
- `fence(ordering, scope)` — Memory fence

### Ordering: `Relaxed`, `Acquire`, `Release`, `AcqRel`, `SeqCst`
### Scope: `CTA`, `GPU`, `System`, `None`

---

## Architectures

Built-in (10): `SC`, `x86-TSO`, `SPARC-PSO`, `ARMv8`, `RISC-V`, `PTX-CTA`, `PTX-GPU`, `OpenCL-WG`, `OpenCL-Dev`, `Vulkan-WG`

---

## Experiment Results

After running `cargo run --bin litmus-experiments`, results are saved to `data/experiment_results.json`:

| Experiment | Key Result |
|-----------|------------|
| Proof certificates | 140/140 pass all 4 validation levels |
| False positive analysis | 10.5% FP rate (CI [4.9%, 21.1%]) |
| Owicki-Gries | 5/6 interference-free |
| Eval benchmark | 205 snippets, 8 categories, 109 OOD |
