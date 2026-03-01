; ═══════════════════════════════════════════════════════════════════
; LITMUS∞ SMT-LIB2 Encoding — Axiomatic Memory Model Verification
; ═══════════════════════════════════════════════════════════════════
;
; Theory: Axiomatic weak memory model verification (Alglave et al. 2014)
;
; Sorts:
;   Int — event timestamps (topological ordering for acyclicity)
;   Int — read-from values (index into per-address store list)
;   Bool — coherence order relation (total order on stores per address)
;
; Relations encoded:
;   po  — program order (model-dependent preservation)
;   rf  — read-from (which store each load reads)
;   co  — coherence order (total order on stores per address)
;   fr  — from-reads (derived: load reads from s_i, s_j co-after s_i)
;
; Safety property:
;   UNSAT ⟹ forbidden outcome unreachable under model (SAFE)
;   SAT   ⟹ forbidden outcome reachable (UNSAFE, witness provided)
;
; Memory model: ARM
; Pattern: publish_array_fence
; Forbidden outcome: {"r0": 1, "r1": 0}
; ═══════════════════════════════════════════════════════════════════

; ── Theory Declarations ─────────────────────────────────────────────
; Axiomatic memory model framework (Alglave, Maranget, Tautschnig 2014)
;
; An execution X = (E, po, rf, co) where:
;   E   = set of memory events (reads/writes with addresses and values)
;   po  = program order (per-thread total order on events)
;   rf  = read-from function mapping each read to the write it reads
;   co  = coherence order (per-address total order on writes)
;   fr  = from-reads: derived as rf^{-1}; co
;
; A memory model M defines which executions are consistent via
; acyclicity constraints on combinations of these relations.
;
; Models encoded:
;   TSO   — acyclic(po ∪ rf ∪ co ∪ fr), po relaxes W→R only
;   PSO   — additionally relaxes W→W
;   ARM   — relaxes all cross-address po except deps and fences
;   RISC-V — like ARM with asymmetric fence.{pred}.{succ}
;   GPU    — like ARM with scoped fences (workgroup/device/system)
;
; Encoding strategy:
;   Acyclicity is encoded via integer timestamps:
;   For each event e, ts(e) ∈ ℤ with ts(a) < ts(b) for each preserved edge a→b.
;   A cycle exists iff the conjunction of all ts constraints is unsatisfiable.
;   We conjoin the forbidden outcome constraint and check satisfiability.

; ── Pattern: publish_array_fence ─────────────────────────
; Threads: 2
; Addresses: x, y, z
; Memory model: ARM
; Forbidden outcome: {'r0': 1, 'r1': 0}
;
; Operations:
;   e0: T0 Store x = 1
;   e1: T0 Store y = 1
;   e2: T0 Fence
;   e3: T0 Store z = 1
;   e4: T1 Load z → r0
;   e5: T1 Fence
;   e6: T1 Load x → r1
;   e7: T1 Load y → r2

(set-logic QF_LIA)

(declare-fun rf_4 () Int)
(declare-fun rf_6 () Int)
(declare-fun rf_7 () Int)
(declare-fun ts_0 () Int)
(declare-fun ts_1 () Int)
(declare-fun ts_2 () Int)
(declare-fun ts_3 () Int)
(declare-fun ts_4 () Int)
(declare-fun ts_5 () Int)
(declare-fun ts_6 () Int)
(declare-fun ts_7 () Int)
(declare-fun ts_init_x () Int)
(declare-fun ts_init_y () Int)
(declare-fun ts_init_z () Int)
(assert (>= rf_4 0))
(assert (< rf_4 2))
(assert (>= rf_6 0))
(assert (< rf_6 2))
(assert (>= rf_7 0))
(assert (< rf_7 2))
(assert (>= ts_0 0))
(assert (< ts_0 110))
(assert (>= ts_1 0))
(assert (< ts_1 110))
(assert (>= ts_2 0))
(assert (< ts_2 110))
(assert (>= ts_3 0))
(assert (< ts_3 110))
(assert (>= ts_4 0))
(assert (< ts_4 110))
(assert (>= ts_5 0))
(assert (< ts_5 110))
(assert (>= ts_6 0))
(assert (< ts_6 110))
(assert (>= ts_7 0))
(assert (< ts_7 110))
(assert (>= ts_init_x 0))
(assert (>= ts_init_y 0))
(assert (>= ts_init_z 0))
(assert (< ts_0 ts_3))
(assert (< ts_1 ts_3))
(assert (< ts_4 ts_6))
(assert (< ts_4 ts_7))
(assert (=> (= rf_4 0) (< ts_init_z ts_4)))
(assert (=> (= rf_4 1) (< ts_3 ts_4)))
(assert (=> (= rf_6 0) (< ts_init_x ts_6)))
(assert (=> (= rf_6 1) (< ts_0 ts_6)))
(assert (=> (= rf_7 0) (< ts_init_y ts_7)))
(assert (=> (= rf_7 1) (< ts_1 ts_7)))
(assert (< ts_init_x ts_0))
(assert (< ts_init_y ts_1))
(assert (< ts_init_z ts_3))
(assert (=> (= rf_4 0) (< ts_4 ts_3)))
(assert (=> (= rf_6 0) (< ts_6 ts_0)))
(assert (=> (= rf_7 0) (< ts_7 ts_1)))
(assert (and (or (= rf_4 1)) (or (= rf_6 0))))

(check-sat)

; ── UNSAT Proof Certificate ──
; Status: UNSAT — forbidden outcome UNREACHABLE under ARM
; Unsat core size: 5 assertions (out of 42)
; Core assertions: track_39, track_30, track_forbidden, track_25, track_27
(get-unsat-core)

(exit)