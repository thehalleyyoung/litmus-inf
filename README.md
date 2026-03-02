# LITMUS∞: SMT-Verified Memory Model Portability Checking

Advisory pre-screening tool that checks whether concurrent C/C++/CUDA code is safe to port across architectures. Every verdict carries a proof certificate, cross-validated across two independent SMT solvers.

## 30-Second Quickstart

```bash
cd litmus_inf && pip install -e .
litmus-check --target arm myfile.c
```

Output:
```
UNSAFE: mp (data_race) → dmb ishst (T0); dmb ishld (T1)  [Z3: SAT witness]
SAFE:   mp_fence                                           [Z3: UNSAT + Alethe proof]
2 patterns matched, 1 unsafe, 1 safe  (8ms)
```

JSON output for CI pipelines:
```bash
litmus-check --target arm --json src/ | jq '.unsafe_patterns'
```

## Key Results

| Metric | Value | Evidence |
|--------|-------|----------|
| Z3 certificates | **1,400/1,400** | 597 UNSAT proofs + 803 SAT witnesses |
| Cross-solver (Z3+CVC5) | **1,400/1,400** | Wilson CI [99.7%, 100%] |
| Proof validation | **69/69 SMT re-check** | Independent re-verification [94.7%, 100%] |
| LLM recognition | **52.5% exact** | 40 adversarial snippets, 8 categories, CI [37.5%, 67.1%] |
| Compositional FP rate | **8.3%** | 60 analyses, CI [3.6%, 18.1%] |
| herd7 consistency | **228/228** | CPU models |
| GPU external validation | **94/94** | Published specs |
| Bug coverage | **35/41 (85.4%)** | Documented real-world bugs |
| Pattern library | **140 patterns** | Classical, lock-free, RCU, GPU |
| Speed | **<6ms per check** | avg 5.4ms per pattern-model pair |

## Limitations (Honest)

- **140 fixed patterns** — cannot discover novel bugs or verify whole programs
- **Compositional FP rate 8.3%** — conservative analysis over-flags some shared-variable programs
- **LLM accuracy 52.5%** — dependency patterns (0%) and GPU (20%) are weak
- **GPU models are specification-based** — not hardware-tested (94/94 external cross-validation)
- **Fence costs are analytical** — not measured hardware latencies
- **LLM mode requires OPENAI_API_KEY**
- **Not on PyPI** — install from source only

## Supported Architectures

- **CPU:** x86-TSO, SPARC-PSO, ARMv8, RISC-V RVWMO
- **GPU:** OpenCL (WG/Device), Vulkan (WG/Device), PTX/CUDA (CTA/GPU)
- **Custom:** define via DSL (`model MyModel { relaxes W->R, W->W ... }`)

## Dependencies

- Python 3.8+
- z3-solver (`pip install z3-solver`)
- tree-sitter, tree-sitter-c, tree-sitter-cpp (optional, for AST analysis)
- openai (optional, for LLM-assisted recognition)

## Documentation

- [API.md](API.md) — API reference (CLI and Python)
- [docs/paper.tex](docs/paper.tex) — full paper with methodology and proofs
