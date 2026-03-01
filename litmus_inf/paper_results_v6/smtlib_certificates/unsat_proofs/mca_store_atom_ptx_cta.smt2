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
; Memory model: PTX-CTA
; Pattern: mca_store_atom
; Forbidden outcome: {"r0": 1, "r1": 2, "r2": 2, "r3": 1}
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

; ── Pattern: mca_store_atom ─────────────────────────
; Threads: 4
; Addresses: x
; Memory model: PTX-CTA
; Forbidden outcome: {'r0': 1, 'r1': 2, 'r2': 2, 'r3': 1}
;
; Operations:
;   e0: T0 Store x = 1
;   e1: T1 Store x = 2
;   e2: T2 Load x → r0
;   e3: T2 Load x → r1
;   e4: T3 Load x → r2
;   e5: T3 Load x → r3

(set-logic QF_LIA)

(declare-fun rf_2 () Int)
(declare-fun rf_3 () Int)
(declare-fun rf_4 () Int)
(declare-fun rf_5 () Int)
(declare-fun co_x_2_1 () Bool)
(declare-fun co_x_1_2 () Bool)
(declare-fun ts_0 () Int)
(declare-fun ts_1 () Int)
(declare-fun ts_2 () Int)
(declare-fun ts_3 () Int)
(declare-fun ts_4 () Int)
(declare-fun ts_5 () Int)
(declare-fun ts_init_x () Int)
(assert (>= rf_2 0))
(assert (< rf_2 3))
(assert (>= rf_3 0))
(assert (< rf_3 3))
(assert (>= rf_4 0))
(assert (< rf_4 3))
(assert (>= rf_5 0))
(assert (< rf_5 3))
(assert (or co_x_1_2 co_x_2_1))
(assert (not (and co_x_1_2 co_x_2_1)))
(assert (>= ts_0 0))
(assert (< ts_0 70))
(assert (>= ts_1 0))
(assert (< ts_1 70))
(assert (>= ts_2 0))
(assert (< ts_2 70))
(assert (>= ts_3 0))
(assert (< ts_3 70))
(assert (>= ts_4 0))
(assert (< ts_4 70))
(assert (>= ts_5 0))
(assert (< ts_5 70))
(assert (>= ts_init_x 0))
(assert (< ts_2 ts_3))
(assert (< ts_4 ts_5))
(assert (=> (= rf_2 0) (< ts_init_x ts_2)))
(assert (=> (= rf_2 1) (< ts_0 ts_2)))
(assert (=> (= rf_2 2) (< ts_1 ts_2)))
(assert (=> (= rf_3 0) (< ts_init_x ts_3)))
(assert (=> (= rf_3 1) (< ts_0 ts_3)))
(assert (=> (= rf_3 2) (< ts_1 ts_3)))
(assert (=> (= rf_4 0) (< ts_init_x ts_4)))
(assert (=> (= rf_4 1) (< ts_0 ts_4)))
(assert (=> (= rf_4 2) (< ts_1 ts_4)))
(assert (=> (= rf_5 0) (< ts_init_x ts_5)))
(assert (=> (= rf_5 1) (< ts_0 ts_5)))
(assert (=> (= rf_5 2) (< ts_1 ts_5)))
(assert (< ts_init_x ts_0))
(assert (< ts_init_x ts_1))
(assert (=> co_x_1_2 (< ts_0 ts_1)))
(assert (=> co_x_2_1 (< ts_1 ts_0)))
(assert (=> (= rf_2 0) (< ts_2 ts_0)))
(assert (=> (= rf_2 0) (< ts_2 ts_1)))
(assert (=> (and (= rf_2 1) co_x_1_2) (< ts_2 ts_1)))
(assert (=> (and (= rf_2 2) co_x_2_1) (< ts_2 ts_0)))
(assert (=> (= rf_3 0) (< ts_3 ts_0)))
(assert (=> (= rf_3 0) (< ts_3 ts_1)))
(assert (=> (and (= rf_3 1) co_x_1_2) (< ts_3 ts_1)))
(assert (=> (and (= rf_3 2) co_x_2_1) (< ts_3 ts_0)))
(assert (=> (= rf_4 0) (< ts_4 ts_0)))
(assert (=> (= rf_4 0) (< ts_4 ts_1)))
(assert (=> (and (= rf_4 1) co_x_1_2) (< ts_4 ts_1)))
(assert (=> (and (= rf_4 2) co_x_2_1) (< ts_4 ts_0)))
(assert (=> (= rf_5 0) (< ts_5 ts_0)))
(assert (=> (= rf_5 0) (< ts_5 ts_1)))
(assert (=> (and (= rf_5 1) co_x_1_2) (< ts_5 ts_1)))
(assert (=> (and (= rf_5 2) co_x_2_1) (< ts_5 ts_0)))
(assert (and (or (= rf_2 1)) (or (= rf_3 2)) (or (= rf_4 2)) (or (= rf_5 1))))

(check-sat)

; ── UNSAT Proof Certificate ──
; Status: UNSAT — forbidden outcome UNREACHABLE under PTX-CTA
; Unsat core size: 8 assertions (out of 58)
; Core assertions: track_26, track_48, track_55, track_24, track_forbidden, track_33, track_23, track_8
(get-unsat-core)

(exit)