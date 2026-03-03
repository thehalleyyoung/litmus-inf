# LITMUS∞: Complete Axiomatic Memory Model Verification via Algebraic Compression

A verification tool for concurrent memory-model litmus tests across CPU and GPU
architectures. LITMUS∞ enumerates all candidate executions of a litmus test,
checks each against axiomatic memory-model rules, and reports which executions
are consistent. Every verdict includes algebraic symmetry compression with a
machine-checkable certificate.

Supports 8 memory models (SC, TSO, PSO, ARM, RISC-V, PTX, WebGPU, Vulkan),
6 input formats (TOML, JSON, herd7 `.litmus`, herd-style, LISA, PTX assembly),
and 3 output formats (text, JSON, Graphviz DOT).

---

## Table of Contents

1.  [Quick Start](#quick-start)
2.  [Installation](#installation)
3.  [CLI Overview](#cli-overview)
4.  [Subcommand Reference](#subcommand-reference)
5.  [Input File Formats](#input-file-formats)
6.  [Memory Models](#memory-models)
7.  [Worked Examples](#worked-examples)
8.  [Output Formats](#output-formats)
9.  [Algebraic Compression](#algebraic-compression)
10. [Fence Advisor](#fence-advisor)
11. [Portability Checker](#portability-checker)
12. [Model Diffing](#model-diffing)
13. [Proof Certificate System](#proof-certificate-system)
14. [Architecture Overview](#architecture-overview)
15. [Programmatic API](#programmatic-api)
16. [Key Results](#key-results)
17. [Limitations](#limitations)
18. [Further Documentation](#further-documentation)

---

## Quick Start

```bash
cargo build --release
```

The build produces three binaries in `target/release/`:

| Binary | Purpose |
|--------|---------|
| `litmus-cli` | Interactive CLI for verification, compression, fence advice |
| `litmus-experiments` | Full experiment runner (proof certificates, benchmarks) |
| `exec-graph-bench` | Execution-graph micro-benchmarks |

Try it immediately:

```bash
# Verify the store-buffering test under Sequential Consistency
$ ./target/release/litmus-cli verify -t sb -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       SB                          ║
║  Model:      SC                          ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

SC forbids the outcome where both reads return 0. Three of the four candidate
executions are consistent; the fourth is the forbidden store-buffering anomaly.

---

## Installation

**Prerequisites:** Rust 1.70+ (edition 2021). No external SMT solver required.

```bash
git clone <repository-url>
cd litmus-inf
cargo build --release
```

**Dependencies** (all pulled automatically by Cargo): serde, serde\_json, toml,
clap, rayon, petgraph, regex, ndarray, itertools, rand, bitvec, indexmap,
smallvec, num-traits, hashbrown, thiserror, log, env\_logger.

---

## CLI Overview

```
$ ./target/release/litmus-cli --help
LITMUS∞ — Complete Axiomatic Memory Model Verification via Algebraic Compression

Usage: litmus-cli <COMMAND>

Commands:
  models             List all supported memory models
  verify             Run a built-in litmus test or verify a test from a file
  compress           Show compression ratio for a built-in or file-based litmus test
  diff               Diff two memory models
  fence-advise       Recommend minimal fences for a litmus test across architectures
  benchmark          Run full benchmark suite and output CSV data
  portability-check  Check portability of a concurrent pattern across architectures
  list-patterns      List all built-in concurrent patterns
  check              Validate a litmus test file (parse only, no verification)
  init-test          Generate a starter litmus test template file
  help               Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

---

## Subcommand Reference

### `verify` — Run verification

```
$ ./target/release/litmus-cli verify --help
Run a built-in litmus test or verify a test from a file

Usage: litmus-cli verify [OPTIONS]

Options:
  -t, --test <TEST>                    Test name: sb, mp, lb, iriw, 2+2w, rwc, wrc, sb4, dekker, mp+fence [default: sb]
  -m, --model <MODEL>                  Memory model: SC, TSO, PSO, ARM, RISC-V [default: SC]
  -f, --file <FILE>                    Path to a litmus test file (simple, herd7, LISA, or PTX format)
      --output-format <OUTPUT_FORMAT>  Output format: text, json, dot [default: text]
      --arch <ARCH>                    Target architecture hint: x86-TSO, ARM, RISC-V, Power
  -h, --help                           Print help
```

| `--arch` value | Maps to model |
|----------------|---------------|
| `x86-TSO`, `x86`, `TSO` | TSO |
| `ARM`, `ARMv8`, `AArch64` | ARM (ARMv8) |
| `RISC-V`, `RV`, `RVWMO` | RISC-V |
| `Power`, `PPC`, `PowerPC` | PSO |

### `check` — Parse-only validation (no verification)

```
$ ./target/release/litmus-cli check --help
Validate a litmus test file (parse only, no verification)

Usage: litmus-cli check <FILE>

Arguments:
  <FILE>  Path to a litmus test file

Options:
  -h, --help  Print help
```

### `init-test` — Generate starter templates

```
$ ./target/release/litmus-cli init-test --help
Generate a starter litmus test template file

Usage: litmus-cli init-test [OPTIONS]

Options:
  -o, --output <OUTPUT>      Output file path [default: my_test.toml]
      --template <TEMPLATE>  Template: sb, mp, lb, iriw, fence [default: sb]
  -h, --help                 Print help
```

### `compress` — Show symmetry compression ratio

```
$ ./target/release/litmus-cli compress --help
Show compression ratio for a built-in or file-based litmus test

Usage: litmus-cli compress [OPTIONS]

Options:
  -t, --test <TEST>  Test name: sb, mp, lb, iriw, 2+2w, rwc, wrc, sb4, dekker, mp+fence [default: sb]
  -f, --file <FILE>  Path to a litmus test file
  -h, --help         Print help
```

### `fence-advise` — Per-architecture fence recommendations

```
$ ./target/release/litmus-cli fence-advise --help
Recommend minimal fences for a litmus test across architectures

Usage: litmus-cli fence-advise [OPTIONS]

Options:
  -t, --test <TEST>  Test name: sb, mp, lb, iriw, 2+2w, dekker [default: sb]
  -f, --file <FILE>  Path to a litmus test file
  -h, --help         Print help
```

### `diff` — Compare two memory models

```
$ ./target/release/litmus-cli diff --help
Diff two memory models

Usage: litmus-cli diff <MODEL_A> <MODEL_B>

Arguments:
  <MODEL_A>  First model: SC, TSO, PSO, ARM, RISC-V
  <MODEL_B>  Second model: SC, TSO, PSO, ARM, RISC-V

Options:
  -h, --help  Print help
```

---

## Input File Formats

LITMUS∞ auto-detects the file format from the extension and content.

### TOML Format (recommended)

The `examples/dekker.toml` file ships with the repository:

```toml
# Dekker's mutual exclusion — litmus test
# Each thread sets its flag then reads the other's flag.
# Under SC, both seeing 0 is forbidden (mutual exclusion holds).
# Under TSO/ARM, this outcome IS possible without a fence.

name = "Dekker"

[locations]
flag0 = 0
flag1 = 0

[[threads]]
ops = ["W(flag0, 1)", "R(flag1) r0"]

[[threads]]
ops = ["W(flag1, 1)", "R(flag0) r1"]

[forbidden]
flag0 = 0
flag1 = 0
```

**Operations:** `W(var, value)` — store, `R(var) reg` — load into register,
`fence` — full memory barrier.

Another shipped example — `examples/producer_consumer.toml`:

```toml
# Producer-consumer with flag — message passing pattern
name = "ProducerConsumer"

[locations]
data = 0
ready = 0

[[threads]]
ops = ["W(data, 42)", "W(ready, 1)"]

[[threads]]
ops = ["R(ready) r0", "R(data) r1"]

[forbidden]
ready = 1
data = 0
```

### JSON Format

The `examples/double_checked_locking.json` file:

```json
{
  "name": "DoubleCheckedLocking",
  "locations": { "obj": 0, "ptr": 0 },
  "threads": [
    { "ops": ["W(obj, 1)", "fence", "W(ptr, 1)"] },
    { "ops": ["R(ptr) r0", "R(obj) r1"] }
  ],
  "forbidden": { "ptr": 1, "obj": 0 }
}
```

### herd7 `.litmus` Format

The standard format used by the [herd7](http://diy.inria.fr/doc/herd.html) tool:

```
X86 SB
"Store Buffering test"
{ x=0; y=0; }
 P0          | P1          ;
 MOV [x],$1  | MOV [y],$1  ;
 MOV EAX,[y] | MOV EBX,[x] ;
exists (0:EAX=0 /\ 1:EBX=0)
```

**Supported instructions per architecture:**

| Architecture | Loads | Stores | Fences |
|-------------|-------|--------|--------|
| x86 | `MOV reg,[addr]` | `MOV [addr],$val` | `MFENCE`, `LFENCE`, `SFENCE` |
| AArch64/ARM | `LDR reg,[addr]` | `STR reg,[addr]` | `DMB`, `DSB`, `ISB` |
| RISC-V | `lw reg,addr` | `sw reg,addr` | `fence` |
| PPC/Power | `lwz reg,addr` | `stw reg,addr` | `sync`, `lwsync`, `isync` |

### Additional Formats

- **herd-style:** `W()`/`R()` notation with `P0:`/`P1:` thread headers and `exists` clause.
- **LISA:** `w[x] 1`, `r[y] r0`, `f[full]` syntax with `thread N` headers.
- **PTX:** NVIDIA PTX assembly (`st.global.relaxed.sys`, `ld.global.acquire.sys`, `membar`).

---

## Memory Models

```
$ ./target/release/litmus-cli models
╔══════════════════════════════════════════╗
║       LITMUS∞ Supported Models           ║
╠══════════════════════════════════════════╣
║  SC                                     ║
║  TSO                                    ║
║  PSO                                    ║
║  ARM                                    ║
║  RISC-V                                 ║
║  PTX                                    ║
║  WebGPU                                 ║
║  Vulkan                                 ║
╚══════════════════════════════════════════╝
```

### CPU Models

| Model | Architecture | Key relaxation |
|-------|-------------|----------------|
| **SC** | Sequential Consistency | None — total order of all operations |
| **TSO** | x86 Total Store Order | Store→Load reordering (store buffer) |
| **PSO** | SPARC Partial Store Order | Store→Store + Store→Load reordering |
| **ARM** | ARMv8 | Most reorderings allowed; requires DMB/DSB fences |
| **RISC-V** | RVWMO | Most reorderings allowed; requires `fence` instruction |

### GPU Models

| Model | Platform | Scope |
|-------|----------|-------|
| **PTX** | NVIDIA CUDA | Device-wide |
| **WebGPU** | WebGPU | Work-group |
| **Vulkan** | Vulkan | Work-group |

**Hierarchy (strongest → weakest):** SC ⊂ TSO ⊂ PSO ⊂ ARM ≈ RISC-V

---

## Worked Examples

### 1. Generating a test template

```
$ ./target/release/litmus-cli init-test -o my_test.toml --template sb
✓ Created sb template: my_test.toml
  Edit the file, then run:
    litmus-cli check my_test.toml
    litmus-cli verify --file my_test.toml
    litmus-cli fence-advise --file my_test.toml
```

The generated file:

```toml
# Store-Buffering litmus test
# Two threads each write to one location and read from the other.
# Under SC, the outcome where both reads see 0 is forbidden.

name = "SB"

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

Other templates: `mp`, `lb`, `iriw`, `fence`:

```
$ ./target/release/litmus-cli init-test -o mp_test.toml --template fence
✓ Created fence template: mp_test.toml
  Edit the file, then run:
    litmus-cli check mp_test.toml
    litmus-cli verify --file mp_test.toml
    litmus-cli fence-advise --file mp_test.toml
```

### 2. Validating a test file (parse-only)

```
$ ./target/release/litmus-cli check examples/dekker.toml
╔══════════════════════════════════════════╗
║       LITMUS∞ File Check                 ║
╠══════════════════════════════════════════╣
║  File:     examples/dekker.toml          ║
║  Test:     Dekker                        ║
║  Threads:  2                             ║
║  Ops:      4                             ║
║  Outcomes: 1                             ║
║  Status:   ✓ valid                       ║
╠══════════════════════════════════════════╣
║  T0: 2 ops                              
║  T1: 2 ops                              
╚══════════════════════════════════════════╝
```

```
$ ./target/release/litmus-cli check examples/producer_consumer.toml
╔══════════════════════════════════════════╗
║       LITMUS∞ File Check                 ║
╠══════════════════════════════════════════╣
║  File:     examples/producer_consumer.toml ║
║  Test:     ProducerConsumer              ║
║  Threads:  2                             ║
║  Ops:      4                             ║
║  Outcomes: 1                             ║
║  Status:   ✓ valid                       ║
╠══════════════════════════════════════════╣
║  T0: 2 ops                              
║  T1: 2 ops                              
╚══════════════════════════════════════════╝
```

### 3. Verifying built-in tests under different models

**Store-Buffering (SB) — the classic test:**

Under SC, the "both-read-zero" outcome is forbidden (3 of 4 executions pass):

```
$ ./target/release/litmus-cli verify -t sb -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       SB                          ║
║  Model:      SC                          ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

Under TSO, the store buffer allows the anomaly — all 4 executions are consistent:

```
$ ./target/release/litmus-cli verify -t sb -m TSO
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       SB                          ║
║  Model:      TSO                         ║
║  Consistent: 4                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

**Message-Passing (MP) under SC:**

```
$ ./target/release/litmus-cli verify -t mp -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       MP                          ║
║  Model:      SC                          ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

**Load-Buffering (LB) under SC:**

```
$ ./target/release/litmus-cli verify -t lb -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       LB                          ║
║  Model:      SC                          ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

**IRIW (4-thread test) under SC:**

```
$ ./target/release/litmus-cli verify -t iriw -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       IRIW                        ║
║  Model:      SC                          ║
║  Consistent: 15                          ║
║  Checked:    16                          ║
╚══════════════════════════════════════════╝
```

**2+2W (write-write ordering) under SC:**

```
$ ./target/release/litmus-cli verify -t 2+2w -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       2+2W                        ║
║  Model:      SC                          ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

**SB4 (4-thread store-buffering) under SC:**

```
$ ./target/release/litmus-cli verify -t sb4 -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       SB4                         ║
║  Model:      SC                          ║
║  Consistent: 15                          ║
║  Checked:    16                          ║
╚══════════════════════════════════════════╝
```

**MP+fence under TSO — fences restore ordering:**

```
$ ./target/release/litmus-cli verify -t mp+fence -m TSO
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       MP+fence                    ║
║  Model:      TSO                         ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

### 4. Verifying user-defined files

**Dekker's mutual exclusion — safe under SC, broken under ARM:**

```
$ ./target/release/litmus-cli verify --file examples/dekker.toml -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       Dekker                      ║
║  Model:      SC                          ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

```
$ ./target/release/litmus-cli verify -t dekker -m ARM
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       Dekker                      ║
║  Model:      ARMv8                       ║
║  Consistent: 4                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

Under ARM, all 4 executions are consistent — the forbidden outcome is now
permitted, meaning Dekker's algorithm is **broken** without barriers on ARM.

**Producer-Consumer under TSO:**

```
$ ./target/release/litmus-cli verify --file examples/producer_consumer.toml -m TSO
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       ProducerConsumer            ║
║  Model:      TSO                         ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

**Double-Checked Locking (JSON input) — safe under SC, broken under ARM:**

```
$ ./target/release/litmus-cli verify --file examples/double_checked_locking.json -m SC
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       DoubleCheckedLocking        ║
║  Model:      SC                          ║
║  Consistent: 3                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

```
$ ./target/release/litmus-cli verify --file examples/double_checked_locking.json -m ARM
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       DoubleCheckedLocking        ║
║  Model:      ARMv8                       ║
║  Consistent: 4                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

### 5. Using `--arch` for architecture-specific verification

The `--arch` flag overrides `--model` with a hardware-specific mapping:

```
$ ./target/release/litmus-cli verify -t sb -m SC --arch x86-TSO
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       SB                          ║
║  Model:      TSO                         ║
║  Arch:       x86-TSO                     ║
║  Consistent: 4                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

```
$ ./target/release/litmus-cli verify -t sb -m SC --arch ARM
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       SB                          ║
║  Model:      ARMv8                       ║
║  Arch:       ARM                         ║
║  Consistent: 4                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

```
$ ./target/release/litmus-cli verify -t sb -m RISC-V
╔══════════════════════════════════════════╗
║       LITMUS∞ Verification Result        ║
╠══════════════════════════════════════════╣
║  Test:       SB                          ║
║  Model:      RISC-V                      ║
║  Consistent: 4                           ║
║  Checked:    4                           ║
╚══════════════════════════════════════════╝
```

---

## Output Formats

### Text (default)

Human-readable box diagram — shown in all examples above.

### JSON — `--output-format json`

Machine-readable output for CI pipelines and scripting:

```
$ ./target/release/litmus-cli verify -t sb -m SC --output-format json
{
  "consistent_executions": 3,
  "forbidden_observed": 0,
  "inconsistent_executions": 1,
  "model": "SC",
  "observed_outcomes": 3,
  "pass": true,
  "test": "SB",
  "total_executions": 4
}
```

Works with file-based tests too:

```
$ ./target/release/litmus-cli verify --file examples/dekker.toml -m SC --output-format json
{
  "consistent_executions": 3,
  "forbidden_observed": 0,
  "inconsistent_executions": 1,
  "model": "SC",
  "observed_outcomes": 3,
  "pass": true,
  "test": "Dekker",
  "total_executions": 4
}
```

### Graphviz DOT — `--output-format dot`

Generates a DOT graph with per-thread subgraphs and program-order edges:

```
$ ./target/release/litmus-cli verify -t sb -m SC --output-format dot
digraph litmus {
  rankdir=LR;
  label="SB under SC";
  labelloc=t;
  node [shape=record, fontsize=10];
  subgraph cluster_t0 {
    label="Thread 0";
    style=dashed;
    t0_0 [label="W(0x100,1) Relaxed"];
    t0_1 [label="R(0x200)→r0 Relaxed"];
    t0_0 -> t0_1 [style=bold, label="po"];
  }
  subgraph cluster_t1 {
    label="Thread 1";
    style=dashed;
    t1_0 [label="W(0x200,1) Relaxed"];
    t1_1 [label="R(0x100)→r0 Relaxed"];
    t1_0 -> t1_1 [style=bold, label="po"];
  }
  // Result: 3 consistent / 4 total
}
```

Render to PNG with Graphviz:

```bash
./target/release/litmus-cli verify -t sb -m SC --output-format dot > sb.dot
dot -Tpng sb.dot -o sb.png
```

---

## Algebraic Compression

The `compress` command analyzes symmetry in a litmus test's state space:

```
$ ./target/release/litmus-cli compress -t sb
╔══════════════════════════════════════════╗
║       LITMUS∞ Compression Result         ║
╠══════════════════════════════════════════╣
║  Test:         SB                        ║
║  Compression: 2.0x (threads: 1, addrs: 2, vals: 2), certificate: valid
║  Certificate:  ✓ valid                   ║
╚══════════════════════════════════════════╝
```

Larger tests benefit more — SB4 (4 threads) achieves 4× compression:

```
$ ./target/release/litmus-cli compress -t sb4
╔══════════════════════════════════════════╗
║       LITMUS∞ Compression Result         ║
╠══════════════════════════════════════════╣
║  Test:         SB4                       ║
║  Compression: 4.0x (threads: 1, addrs: 2, vals: 2), certificate: valid
║  Certificate:  ✓ valid                   ║
╚══════════════════════════════════════════╝
```

IRIW (4-thread, multi-observer):

```
$ ./target/release/litmus-cli compress -t iriw
╔══════════════════════════════════════════╗
║       LITMUS∞ Compression Result         ║
╠══════════════════════════════════════════╣
║  Test:         IRIW                      ║
║  Compression: 2.0x (threads: 2, addrs: 1, vals: 2), certificate: valid
║  Certificate:  ✓ valid                   ║
╚══════════════════════════════════════════╝
```

Compress a user file:

```
$ ./target/release/litmus-cli compress --file examples/dekker.toml
╔══════════════════════════════════════════╗
║       LITMUS∞ Compression Result         ║
╠══════════════════════════════════════════╣
║  Test:         Dekker                    ║
║  Compression: 2.0x (threads: 1, addrs: 2, vals: 2), certificate: valid
║  Certificate:  ✓ valid                   ║
╚══════════════════════════════════════════╝
```

---

## Fence Advisor

The `fence-advise` command recommends minimal fences per architecture:

```
$ ./target/release/litmus-cli fence-advise -t sb
╔══════════════════════════════════════════════════════╗
║       LITMUS∞ Fence Advisor: SB                       ║
╠══════════════════════════════════════════════════════╣
║  SC             SAFE   No fence needed                  ║
║  TSO      NEEDS FENCE   MFENCE between store and load    ║
║  PSO      NEEDS FENCE   STBAR between stores             ║
║  ARM      NEEDS FENCE   DMB between operations           ║
║  RISC-V   NEEDS FENCE   fence rw,rw between operations   ║
╚══════════════════════════════════════════════════════╝
```

Dekker's algorithm — same fence requirements as SB (it is a store-buffer variant):

```
$ ./target/release/litmus-cli fence-advise -t dekker
╔══════════════════════════════════════════════════════╗
║       LITMUS∞ Fence Advisor: Dekker                   ║
╠══════════════════════════════════════════════════════╣
║  SC             SAFE   No fence needed                  ║
║  TSO      NEEDS FENCE   MFENCE between store and load    ║
║  PSO      NEEDS FENCE   STBAR between stores             ║
║  ARM      NEEDS FENCE   DMB between operations           ║
║  RISC-V   NEEDS FENCE   fence rw,rw between operations   ║
╚══════════════════════════════════════════════════════╝
```

Producer-Consumer — safe under TSO (store ordering preserved), needs fences on weaker models:

```
$ ./target/release/litmus-cli fence-advise --file examples/producer_consumer.toml
╔══════════════════════════════════════════════════════╗
║       LITMUS∞ Fence Advisor: ProducerConsumer         ║
╠══════════════════════════════════════════════════════╣
║  SC             SAFE   No fence needed                  ║
║  TSO            SAFE   No fence needed                  ║
║  PSO      NEEDS FENCE   STBAR between stores             ║
║  ARM      NEEDS FENCE   DMB between operations           ║
║  RISC-V   NEEDS FENCE   fence rw,rw between operations   ║
╚══════════════════════════════════════════════════════╝
```

---

## Portability Checker

Check a concurrent pattern across all architectures at once:

```
$ ./target/release/litmus-cli list-patterns
Built-in concurrent patterns:
  Spinlock (store-buffer)        Two threads each set a flag and check the other's — the classic mutual exclusion check (Peterson/Dekker style)
  Message Passing                Producer writes data then sets flag; consumer reads flag then data. Checks store-to-load ordering.
  Double-Checked Locking         Thread 0 initializes object then sets initialized flag. Thread 1 checks flag then reads object. Classic DCL bug.
  Seqlock Reader                 Load-buffering pattern: each thread reads then writes. Tests whether speculative loads can see future stores.
  Producer-Consumer Write Ordering Two threads write to two locations in opposite orders. Tests store-store ordering (2+2W pattern).
```

**Spinlock — breaks on all 4 architectures:**

```
$ ./target/release/litmus-cli portability-check -p spinlock
╔══════════════════════════════════════════════╗
║  LITMUS∞ Portability Report                  ║
╠══════════════════════════════════════════════╣
║  Pattern: Spinlock (store-buffer)            ║
╠══════════════════════════════════════════════╣
║  x86 (TSO)    ✗ BROKEN (4 execs checked) ║
║    Fix: MFENCE after op 0 in T0       ║
║    Fix: MFENCE after op 0 in T1       ║
║  SPARC (PSO)  ✗ BROKEN (4 execs checked) ║
║    Fix: MEMBAR #StoreLoad after op 0 in T0       ║
║    Fix: MEMBAR #StoreLoad after op 0 in T1       ║
║  ARM (ARMv8)  ✗ BROKEN (4 execs checked) ║
║    Fix: DMB ISH after op 0 in T0       ║
║    Fix: DMB ISH after op 0 in T1       ║
║  RISC-V (RVWMO) ✗ BROKEN (4 execs checked) ║
║    Fix: fence rw,rw after op 0 in T0       ║
║    Fix: fence rw,rw after op 0 in T1       ║
╠══════════════════════════════════════════════╣
║  Pattern breaks on 4/4 architectures — fences needed  ║
╚══════════════════════════════════════════════╝
```

**Message Passing — safe on x86, broken on weaker models:**

```
$ ./target/release/litmus-cli portability-check -p message-passing
╔══════════════════════════════════════════════╗
║  LITMUS∞ Portability Report                  ║
╠══════════════════════════════════════════════╣
║  Pattern: Message Passing                    ║
╠══════════════════════════════════════════════╣
║  x86 (TSO)    ✓ SAFE   (4 execs checked) ║
║  SPARC (PSO)  ✗ BROKEN (4 execs checked) ║
║    Fix: MEMBAR #StoreStore after op 0 in T0       ║
║    Fix: MEMBAR #StoreLoad|#StoreStore after op 0 in T1       ║
║  ARM (ARMv8)  ✗ BROKEN (4 execs checked) ║
║    Fix: DMB ISHST after op 0 in T0       ║
║    Fix: DMB ISHLD after op 0 in T1       ║
║  RISC-V (RVWMO) ✗ BROKEN (4 execs checked) ║
║    Fix: fence w,w after op 0 in T0       ║
║    Fix: fence r,r after op 0 in T1       ║
╠══════════════════════════════════════════════╣
║  Pattern breaks on 3/4 architectures — fences needed  ║
╚══════════════════════════════════════════════╝
```

**Double-Checked Locking — safe on x86, broken on weaker models:**

```
$ ./target/release/litmus-cli portability-check -p dcl
╔══════════════════════════════════════════════╗
║  LITMUS∞ Portability Report                  ║
╠══════════════════════════════════════════════╣
║  Pattern: Double-Checked Locking             ║
╠══════════════════════════════════════════════╣
║  x86 (TSO)    ✓ SAFE   (4 execs checked) ║
║  SPARC (PSO)  ✗ BROKEN (4 execs checked) ║
║    Fix: MEMBAR #StoreStore after op 0 in T0       ║
║    Fix: MEMBAR #StoreLoad|#StoreStore after op 0 in T1       ║
║  ARM (ARMv8)  ✗ BROKEN (4 execs checked) ║
║    Fix: DMB ISHST after op 0 in T0       ║
║    Fix: DMB ISHLD after op 0 in T1       ║
║  RISC-V (RVWMO) ✗ BROKEN (4 execs checked) ║
║    Fix: fence w,w after op 0 in T0       ║
║    Fix: fence r,r after op 0 in T1       ║
╠══════════════════════════════════════════════╣
║  Pattern breaks on 3/4 architectures — fences needed  ║
╚══════════════════════════════════════════════╝
```

---

## Model Diffing

Compare the axiomatic relations of two memory models:

```
$ ./target/release/litmus-cli diff SC TSO
╔══════════════════════════════════════════╗
║       LITMUS∞ Model Diff                 ║
╠══════════════════════════════════════════╣
║  SC vs TSO
║  Model diff: SC vs TSO
  Only in TSO: ["mfence", "ppo", "fence-order", "ghb"]
  Only in SC
  Only in TSO
  Only in TSO

╚══════════════════════════════════════════╝
```

```
$ ./target/release/litmus-cli diff TSO ARM
╔══════════════════════════════════════════╗
║       LITMUS∞ Model Diff                 ║
╠══════════════════════════════════════════╣
║  TSO vs ARMv8
║  Model diff: TSO vs ARMv8
  Only in TSO: ["mfence", "ppo", "fence-order", "ghb"]
  Only in ARMv8: ["obs", "dob", "aob", "bob", "ob"]
  Only in TSO
  Only in TSO
  Only in ARMv8
  Only in ARMv8
  Only in ARMv8

╚══════════════════════════════════════════╝
```

TSO uses `mfence`, `ppo`, `fence-order`, `ghb` relations; ARMv8 uses
`obs`, `dob`, `aob`, `bob`, `ob` — reflecting their fundamentally different
ordering guarantees.

---

## Proof Certificate System

Every verification produces a machine-checkable proof certificate.

1. **SAT Encoding.** Litmus test + model axioms → Boolean formula.
2. **DPLL Solving.** Built-in solver yields UNSAT (safe) or SAT (witness).
3. **Proof Generation.** Alethe-format proof steps (Assume, Resolution, Rewrite, Let, Subproof).
4. **4-Level Validation:**

| Level | Check | Purpose |
|-------|-------|---------|
| Structural | Well-formed proof DAG | Catches malformed proofs |
| Rule validity | Sound inference rules | Catches unsound reasoning |
| Premise resolution | All premises justified | Catches missing justification |
| SMT re-verification | Independent re-check | Catches solver bugs |

**Coverage:** 14 patterns × 10 architectures = 140 certificates, all passing.
Wilson 95% CI: [97.3%, 100.0%].

### Compositional Verification (Owicki-Gries)

For shared-variable tests, variables are classified as SingleWriter,
ReleaseAcquire, Fenced, ReadOnly, or MultiWriterRelaxed. Overapproximation
bound: **2^|V_relaxed| − 1**.

---

## Architecture Overview

```
src/
├── checker/                  # Core verification engine
│   ├── litmus.rs             #   LitmusTest, Thread, Instruction types
│   ├── execution.rs          #   ExecutionGraph, Event, relations (po, rf, co, fr)
│   ├── memory_model.rs       #   Axiomatic model definitions (BuiltinModel)
│   ├── verifier.rs           #   Enumerate + check executions
│   ├── compositional.rs      #   Compositional verifier + Owicki-Gries
│   ├── proof_certificate.rs  #   DPLL proof + 4-level validation
│   ├── portability.rs        #   Cross-architecture portability
│   ├── sat_encoder.rs        #   SAT encoding of constraints
│   ├── x86_model.rs          #   x86-TSO specific axioms
│   ├── gpu_model.rs          #   GPU memory models
│   └── power_model.rs        #   Power/PPC model
├── algebraic/                # Symmetry-based state-space compression
├── frontend/                 # Parsing & visualization
│   ├── parser.rs             #   TOML, JSON, .litmus, herd, LISA, PTX
│   ├── visualizer.rs         #   DOT, ASCII, LaTeX, HTML output
│   └── diff.rs               #   Model diffing
├── llm/                      # LLM evaluation (205-snippet benchmark)
├── symmetry/                 # Partial-order reduction
├── security/                 # Side-channel analysis
├── testgen/                  # Test generation
└── bin/
    ├── main.rs               # litmus-cli entry point
    └── experiments.rs         # litmus-experiments runner
```

**Data flow:**

```
Input file → LitmusParser → LitmusTest → Verifier → VerificationResult
                                │                        ├→ text / json / dot
                         StateSpaceCompressor
                                │
                         CompressionResult + Certificate
```

---

## Programmatic API

```rust
use litmus_infinity::frontend::parser::LitmusParser;
use litmus_infinity::checker::memory_model::BuiltinModel;
use litmus_infinity::checker::verifier::Verifier;

let parser = LitmusParser::new();
let test = parser.parse(&std::fs::read_to_string("test.toml").unwrap()).unwrap();

let mut verifier = Verifier::new(BuiltinModel::TSO.build());
let result = verifier.verify_litmus(&test);
println!("Pass: {}, {}/{} consistent", result.pass,
    result.consistent_executions, result.total_executions);
```

The parser also exposes format-specific methods: `parse_toml()`, `parse_json()`,
`parse_litmus_file()`, `parse_herd()`, `parse_lisa()`, `parse_ptx()`.

---

## Key Results

| Metric | Value | Evidence |
|--------|-------|----------|
| Proof certificates | **140/140** | 107 UNSAT + 33 SAT |
| 4-level validation | **100%** | CI [97.3%, 100%] |
| Compositional FP rate | **10.5%** | 6/57 analyses, CI [4.9%, 21.1%] |
| Owicki-Gries | **5/6** | Interference freedom verified |
| Patterns | **14** | MP, SB, LB, IRIW, 2+2W, WRC + variants |
| Architectures | **10** | 4 CPU + 6 GPU |

---

## Limitations

- **14 patterns** — litmus-test scope, not arbitrary concurrent programs.
- **Compositional FP rate 10.5%** — higher for flag/counter sharing patterns.
- **GPU models are specification-based** — not validated against real hardware.
- **DPLL solver** — handles <15 variables; not industrial-scale SAT.
- **`.litmus` parser** — supports common instruction subsets, not full ISAs.

---

## Further Documentation

- [API.md](API.md) — Full Rust API reference.
- [docs/tool_paper.tex](docs/tool_paper.tex) — Methodology and formal proofs.
- [docs/meta/grounding.json](docs/meta/grounding.json) — Claim-to-evidence mapping.
- [examples/](examples/) — Example litmus test files (TOML, JSON).
- [examples/ci_demo/](examples/ci_demo/) — CI integration demo.

## License

MIT
