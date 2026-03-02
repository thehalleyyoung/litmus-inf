#!/usr/bin/env python3
"""
Empirical False Positive Rate Analysis for Compositional Reasoning.

Addresses the critique: "Conservative shared-variable composition renders
compositionality theorem inapplicable to most real programs."

This module:
1. Generates a large set of multi-pattern programs with shared variables
2. Computes SMT ground truth (joint analysis) for each
3. Compares conservative compositional analysis against ground truth
4. Reports empirical false positive rates with Wilson confidence intervals

The key insight: conservative composition may report "potentially unsafe"
for programs that are actually safe under joint SMT analysis. We quantify
exactly how often this happens and characterize which interaction patterns
cause false positives.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import (
    PATTERNS, ARCHITECTURES, verify_test, LitmusTest, MemOp,
    recommend_fence, get_stores_to_addr,
)
from statistical_analysis import wilson_ci

try:
    from z3 import Solver, Bool, Int, And, Or, Not, Implies, sat, unsat, BoolVal
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

from smt_validation import encode_litmus_test_smt, validate_pattern_smt


@dataclass
class FPAnalysisResult:
    """Result of false positive analysis for one program."""
    program_name: str
    arch: str
    n_patterns: int
    has_shared_vars: bool
    shared_var_count: int
    conservative_safe: bool
    joint_smt_safe: bool
    is_false_positive: bool  # conservative says unsafe, joint says safe
    is_true_positive: bool   # both say unsafe
    is_true_negative: bool   # both say safe
    interaction_category: str
    time_ms: float


def _generate_shared_variable_programs():
    """Generate diverse multi-pattern programs with shared variables.
    
    Categories:
    1. flag_sharing: Two patterns share a flag/signal variable
    2. data_sharing: Two patterns share a data variable
    3. counter_sharing: Patterns share a counter/index
    4. pointer_sharing: Patterns share a pointer/reference
    5. mixed_sharing: Complex multi-variable sharing
    6. benign_sharing: Read-only sharing (should not cause false positives)
    7. transitive_sharing: A shares with B, B shares with C
    """
    programs = []
    
    # Category 1: Flag sharing (MP + SB sharing flag)
    programs.append({
        'name': 'flag_share_mp_sb',
        'category': 'flag_sharing',
        'description': 'MP and SB share flag variable',
        'ops': [
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=0, optype='store', addr='flag', value=1),
            MemOp(thread=1, optype='load', addr='flag', reg='r0'),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=1, optype='store', addr='done', value=1),
            MemOp(thread=1, optype='load', addr='flag', reg='r2'),
            MemOp(thread=2, optype='store', addr='flag', value=1),
            MemOp(thread=2, optype='load', addr='done', reg='r3'),
        ],
    })
    
    # Category 1b: Flag sharing with fences (should be safe)
    programs.append({
        'name': 'flag_share_mp_sb_fenced',
        'category': 'flag_sharing',
        'description': 'MP and SB share flag, both fenced',
        'ops': [
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=0, optype='fence', addr=''),
            MemOp(thread=0, optype='store', addr='flag', value=1),
            MemOp(thread=1, optype='load', addr='flag', reg='r0'),
            MemOp(thread=1, optype='fence', addr=''),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=1, optype='store', addr='done', value=1),
            MemOp(thread=1, optype='fence', addr=''),
            MemOp(thread=1, optype='load', addr='flag', reg='r2'),
            MemOp(thread=2, optype='store', addr='flag', value=1),
            MemOp(thread=2, optype='fence', addr=''),
            MemOp(thread=2, optype='load', addr='done', reg='r3'),
        ],
    })
    
    # Category 2: Data sharing
    programs.append({
        'name': 'data_share_mp_wrc',
        'category': 'data_sharing',
        'description': 'MP and WRC share data variable',
        'ops': [
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=0, optype='store', addr='flag_a', value=1),
            MemOp(thread=1, optype='load', addr='flag_a', reg='r0'),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=1, optype='store', addr='flag_b', value=1),
            MemOp(thread=2, optype='load', addr='flag_b', reg='r2'),
            MemOp(thread=2, optype='load', addr='data', reg='r3'),
        ],
    })
    
    # Category 2b: Data sharing with different access patterns
    programs.append({
        'name': 'data_share_sb_corr',
        'category': 'data_sharing',
        'description': 'SB and coherence share data variable',
        'ops': [
            MemOp(thread=0, optype='store', addr='x', value=1),
            MemOp(thread=0, optype='load', addr='y', reg='r0'),
            MemOp(thread=1, optype='store', addr='y', value=1),
            MemOp(thread=1, optype='load', addr='x', reg='r1'),
            MemOp(thread=2, optype='load', addr='x', reg='r2'),
            MemOp(thread=2, optype='load', addr='x', reg='r3'),
        ],
    })
    
    # Category 3: Counter sharing
    programs.append({
        'name': 'counter_share_mp_rmw',
        'category': 'counter_sharing',
        'description': 'MP and RMW both access shared counter',
        'ops': [
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=0, optype='store', addr='counter', value=1),
            MemOp(thread=1, optype='load', addr='counter', reg='r0'),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=2, optype='store', addr='counter', value=2),
            MemOp(thread=2, optype='load', addr='counter', reg='r2'),
        ],
    })
    
    # Category 4: Pointer sharing (publication pattern)
    programs.append({
        'name': 'ptr_share_publish_read',
        'category': 'pointer_sharing',
        'description': 'Publish data through shared pointer, multiple readers',
        'ops': [
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=0, optype='store', addr='ptr', value=1),
            MemOp(thread=1, optype='load', addr='ptr', reg='r0'),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=2, optype='load', addr='ptr', reg='r2'),
            MemOp(thread=2, optype='load', addr='data', reg='r3'),
        ],
    })
    
    # Category 5: Mixed sharing
    programs.append({
        'name': 'mixed_share_3way',
        'category': 'mixed_sharing',
        'description': 'Three patterns sharing two variables',
        'ops': [
            MemOp(thread=0, optype='store', addr='x', value=1),
            MemOp(thread=0, optype='store', addr='y', value=1),
            MemOp(thread=1, optype='load', addr='y', reg='r0'),
            MemOp(thread=1, optype='load', addr='x', reg='r1'),
            MemOp(thread=1, optype='store', addr='z', value=1),
            MemOp(thread=2, optype='load', addr='z', reg='r2'),
            MemOp(thread=2, optype='load', addr='y', reg='r3'),
        ],
    })
    
    # Category 6: Benign sharing (read-only from one side)
    programs.append({
        'name': 'benign_readonly_share',
        'category': 'benign_sharing',
        'description': 'One pattern only reads shared variable',
        'ops': [
            MemOp(thread=0, optype='store', addr='x', value=1),
            MemOp(thread=0, optype='store', addr='y', value=1),
            MemOp(thread=1, optype='load', addr='y', reg='r0'),
            MemOp(thread=1, optype='load', addr='x', reg='r1'),
            MemOp(thread=2, optype='load', addr='x', reg='r2'),
            MemOp(thread=2, optype='store', addr='z', value=1),
        ],
    })
    
    # Category 6b: Both sides read-only on shared var
    programs.append({
        'name': 'benign_both_readonly',
        'category': 'benign_sharing',
        'description': 'Both patterns only read shared variable',
        'ops': [
            MemOp(thread=0, optype='store', addr='a', value=1),
            MemOp(thread=0, optype='load', addr='shared', reg='r0'),
            MemOp(thread=1, optype='store', addr='b', value=1),
            MemOp(thread=1, optype='load', addr='shared', reg='r1'),
            MemOp(thread=2, optype='store', addr='shared', value=1),
        ],
    })
    
    # Category 7: Transitive sharing
    programs.append({
        'name': 'transitive_abc_chain',
        'category': 'transitive_sharing',
        'description': 'A shares x with B, B shares y with C',
        'ops': [
            MemOp(thread=0, optype='store', addr='data_a', value=1),
            MemOp(thread=0, optype='store', addr='x', value=1),
            MemOp(thread=1, optype='load', addr='x', reg='r0'),
            MemOp(thread=1, optype='load', addr='data_a', reg='r1'),
            MemOp(thread=1, optype='store', addr='y', value=1),
            MemOp(thread=2, optype='load', addr='y', reg='r2'),
            MemOp(thread=2, optype='store', addr='data_c', value=1),
        ],
    })
    
    # Additional programs for statistical significance
    # Disjoint baselines (should never have false positives)
    programs.append({
        'name': 'disjoint_mp_sb',
        'category': 'disjoint_baseline',
        'description': 'Disjoint MP + SB (baseline: no sharing)',
        'ops': [
            MemOp(thread=0, optype='store', addr='a', value=1),
            MemOp(thread=0, optype='store', addr='b', value=1),
            MemOp(thread=1, optype='load', addr='b', reg='r0'),
            MemOp(thread=1, optype='load', addr='a', reg='r1'),
            MemOp(thread=2, optype='store', addr='c', value=1),
            MemOp(thread=2, optype='load', addr='d', reg='r2'),
            MemOp(thread=3, optype='store', addr='d', value=1),
            MemOp(thread=3, optype='load', addr='c', reg='r3'),
        ],
    })
    
    programs.append({
        'name': 'disjoint_mp_mp',
        'category': 'disjoint_baseline',
        'description': 'Two independent MPs (baseline)',
        'ops': [
            MemOp(thread=0, optype='store', addr='data1', value=1),
            MemOp(thread=0, optype='store', addr='flag1', value=1),
            MemOp(thread=1, optype='load', addr='flag1', reg='r0'),
            MemOp(thread=1, optype='load', addr='data1', reg='r1'),
            MemOp(thread=2, optype='store', addr='data2', value=1),
            MemOp(thread=2, optype='store', addr='flag2', value=1),
            MemOp(thread=3, optype='load', addr='flag2', reg='r2'),
            MemOp(thread=3, optype='load', addr='data2', reg='r3'),
        ],
    })
    
    # Fenced shared programs (should be safe)
    programs.append({
        'name': 'fenced_shared_mp_mp',
        'category': 'fenced_sharing',
        'description': 'Two MPs sharing data, both fully fenced',
        'ops': [
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=0, optype='fence', addr=''),
            MemOp(thread=0, optype='store', addr='flag1', value=1),
            MemOp(thread=1, optype='load', addr='flag1', reg='r0'),
            MemOp(thread=1, optype='fence', addr=''),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=2, optype='store', addr='data', value=2),
            MemOp(thread=2, optype='fence', addr=''),
            MemOp(thread=2, optype='store', addr='flag2', value=1),
            MemOp(thread=3, optype='load', addr='flag2', reg='r2'),
            MemOp(thread=3, optype='fence', addr=''),
            MemOp(thread=3, optype='load', addr='data', reg='r3'),
        ],
    })
    
    # Store buffering with shared variable
    programs.append({
        'name': 'sb_shared_with_observer',
        'category': 'data_sharing',
        'description': 'SB with third thread observing shared variable',
        'ops': [
            MemOp(thread=0, optype='store', addr='x', value=1),
            MemOp(thread=0, optype='load', addr='y', reg='r0'),
            MemOp(thread=1, optype='store', addr='y', value=1),
            MemOp(thread=1, optype='load', addr='x', reg='r1'),
            MemOp(thread=2, optype='load', addr='x', reg='r2'),
            MemOp(thread=2, optype='load', addr='y', reg='r3'),
        ],
    })
    
    # IRIW-like sharing
    programs.append({
        'name': 'iriw_shared_reads',
        'category': 'mixed_sharing',
        'description': 'IRIW-like with shared read addresses',
        'ops': [
            MemOp(thread=0, optype='store', addr='x', value=1),
            MemOp(thread=1, optype='store', addr='y', value=1),
            MemOp(thread=2, optype='load', addr='x', reg='r0'),
            MemOp(thread=2, optype='load', addr='y', reg='r1'),
            MemOp(thread=3, optype='load', addr='y', reg='r2'),
            MemOp(thread=3, optype='load', addr='x', reg='r3'),
        ],
    })
    
    # Dekker-like mutual exclusion sharing
    programs.append({
        'name': 'dekker_shared_flag',
        'category': 'flag_sharing',
        'description': 'Dekker with third thread also using flag',
        'ops': [
            MemOp(thread=0, optype='store', addr='flag0', value=1),
            MemOp(thread=0, optype='load', addr='flag1', reg='r0'),
            MemOp(thread=1, optype='store', addr='flag1', value=1),
            MemOp(thread=1, optype='load', addr='flag0', reg='r1'),
            MemOp(thread=2, optype='load', addr='flag0', reg='r2'),
            MemOp(thread=2, optype='load', addr='flag1', reg='r3'),
        ],
    })
    
    # Work-stealing deque sharing
    programs.append({
        'name': 'work_steal_shared_buffer',
        'category': 'pointer_sharing',
        'description': 'Work stealing with shared buffer access',
        'ops': [
            MemOp(thread=0, optype='store', addr='buffer', value=1),
            MemOp(thread=0, optype='store', addr='bottom', value=1),
            MemOp(thread=1, optype='load', addr='bottom', reg='r0'),
            MemOp(thread=1, optype='load', addr='buffer', reg='r1'),
            MemOp(thread=2, optype='load', addr='bottom', reg='r2'),
            MemOp(thread=2, optype='load', addr='buffer', reg='r3'),
        ],
    })
    
    # SeqLock sharing
    programs.append({
        'name': 'seqlock_multi_reader',
        'category': 'counter_sharing',
        'description': 'SeqLock with multiple readers sharing seq counter',
        'ops': [
            MemOp(thread=0, optype='store', addr='seq', value=1),
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=0, optype='store', addr='seq', value=2),
            MemOp(thread=1, optype='load', addr='seq', reg='r0'),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=2, optype='load', addr='seq', reg='r2'),
            MemOp(thread=2, optype='load', addr='data', reg='r3'),
        ],
    })
    
    # Reference counting with reclamation
    programs.append({
        'name': 'refcount_reclaim',
        'category': 'counter_sharing',
        'description': 'Reference count decrement + data access',
        'ops': [
            MemOp(thread=0, optype='store', addr='refcount', value=1),
            MemOp(thread=0, optype='store', addr='data', value=1),
            MemOp(thread=1, optype='load', addr='refcount', reg='r0'),
            MemOp(thread=1, optype='load', addr='data', reg='r1'),
            MemOp(thread=2, optype='store', addr='refcount', value=0),
            MemOp(thread=2, optype='load', addr='data', reg='r2'),
        ],
    })
    
    # RCU publication with multiple readers
    programs.append({
        'name': 'rcu_multi_reader',
        'category': 'pointer_sharing',
        'description': 'RCU publish with two readers',
        'ops': [
            MemOp(thread=0, optype='store', addr='new_data', value=1),
            MemOp(thread=0, optype='store', addr='gptr', value=1),
            MemOp(thread=1, optype='load', addr='gptr', reg='r0'),
            MemOp(thread=1, optype='load', addr='new_data', reg='r1'),
            MemOp(thread=2, optype='load', addr='gptr', reg='r2'),
            MemOp(thread=2, optype='load', addr='new_data', reg='r3'),
        ],
    })
    
    return programs


def _compute_joint_smt_result(ops, arch):
    """Compute SMT result for the joint program (all ops together).
    
    This is the ground truth: encode ALL operations into a single SMT
    formula and check if the forbidden outcome is reachable.
    """
    if not Z3_AVAILABLE:
        return None, 0.0
    
    model_map = {'x86': 'TSO', 'arm': 'ARM', 'riscv': 'RISC-V', 'sparc': 'PSO'}
    model_name = model_map.get(arch)
    if not model_name:
        return None, 0.0
    
    # Build a joint litmus test from all ops
    addrs = sorted(set(op.addr for op in ops if op.addr and op.optype != 'fence'))
    n_threads = max((op.thread for op in ops), default=0) + 1
    
    # Forbidden: all loads see 0 (the weakest assertion)
    forbidden = {}
    for op in ops:
        if op.optype == 'load' and op.reg:
            forbidden[op.reg] = 0
    
    if not forbidden:
        return 'trivial', 0.0
    
    test = LitmusTest(
        name='joint_test', n_threads=n_threads,
        addresses=addrs, ops=ops, forbidden=forbidden,
    )
    
    start = time.time()
    try:
        solver, rf_val, co_vars, forbidden_conj = encode_litmus_test_smt(
            test, model_name)
        solver.add(forbidden_conj)
        result = solver.check()
        elapsed = (time.time() - start) * 1000
        
        if result == sat:
            return 'unsafe', elapsed
        elif result == unsat:
            return 'safe', elapsed
        return 'timeout', elapsed
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return f'error:{str(e)[:50]}', elapsed


def _compute_conservative_result(ops, arch):
    """Compute conservative compositional result.
    
    Identifies patterns, checks each individually, and composes
    conservatively (any unsafe pattern with shared vars → unsafe).
    """
    from compositional_reasoning import (
        identify_patterns_in_program, check_disjoint_composition,
    )
    
    instances = identify_patterns_in_program(ops)
    if not instances:
        return 'unrecognized', False, 0
    
    is_disjoint, shared_vars, interactions = check_disjoint_composition(instances)
    
    # Check each pattern individually
    any_unsafe = False
    for inst in instances:
        if inst.pattern_name in PATTERNS:
            result = validate_pattern_smt(
                inst.pattern_name,
                {'x86': 'TSO', 'arm': 'ARM', 'riscv': 'RISC-V'}.get(arch, 'TSO')
            )
            if result.get('smt_result') == 'sat':
                any_unsafe = True
    
    if is_disjoint:
        return 'safe' if not any_unsafe else 'unsafe', False, len(shared_vars)
    else:
        # Conservative: any unsafe pattern with shared vars → unsafe
        if any_unsafe:
            return 'unsafe', True, len(shared_vars)
        else:
            return 'safe', True, len(shared_vars)


def run_false_positive_analysis(output_dir='paper_results_v13'):
    """Run comprehensive false positive rate analysis.
    
    For each program × architecture:
    1. Compute conservative compositional verdict
    2. Compute joint SMT ground truth
    3. Compare: false positive = conservative says unsafe, joint says safe
    """
    os.makedirs(output_dir, exist_ok=True)
    
    programs = _generate_shared_variable_programs()
    archs = ['x86', 'arm', 'riscv']
    
    print("\n" + "=" * 70)
    print("LITMUS∞ Compositional False Positive Rate Analysis")
    print("=" * 70)
    print(f"Programs: {len(programs)}")
    print(f"Architectures: {archs}")
    print(f"Total analyses: {len(programs) * len(archs)}")
    
    results = []
    category_stats = defaultdict(lambda: {
        'total': 0, 'fp': 0, 'tp': 0, 'tn': 0, 'fn': 0
    })
    
    for prog in programs:
        for arch in archs:
            start = time.time()
            
            # Conservative analysis
            cons_verdict, is_conservative, n_shared = _compute_conservative_result(
                prog['ops'], arch)
            
            # Joint SMT ground truth
            joint_verdict, joint_time = _compute_joint_smt_result(prog['ops'], arch)
            
            elapsed = (time.time() - start) * 1000
            
            # Classify
            cons_safe = cons_verdict == 'safe'
            cons_unknown = cons_verdict == 'unrecognized'
            joint_safe = joint_verdict in ('safe', 'trivial')
            
            # Exclude unrecognized from FP/FN analysis
            is_fp = not cons_safe and not cons_unknown and joint_safe
            is_tp = not cons_safe and not cons_unknown and not joint_safe
            is_tn = cons_safe and joint_safe
            is_fn = cons_safe and not joint_safe
            
            category = prog.get('category', 'unknown')
            category_stats[category]['total'] += 1
            if is_fp:
                category_stats[category]['fp'] += 1
            elif is_tp:
                category_stats[category]['tp'] += 1
            elif is_tn:
                category_stats[category]['tn'] += 1
            elif is_fn:
                category_stats[category]['fn'] += 1
            
            result = FPAnalysisResult(
                program_name=prog['name'],
                arch=arch,
                n_patterns=0,
                has_shared_vars=n_shared > 0,
                shared_var_count=n_shared,
                conservative_safe=cons_safe,
                joint_smt_safe=joint_safe,
                is_false_positive=is_fp,
                is_true_positive=is_tp,
                is_true_negative=is_tn,
                interaction_category=category,
                time_ms=elapsed,
            )
            results.append(result)
            
            status_char = '✓' if not is_fp else '⚠FP'
            print(f"  {prog['name']:30s} {arch:6s}: "
                  f"cons={cons_verdict:8s} joint={str(joint_verdict):8s} "
                  f"{status_char}")
    
    # Compute overall statistics
    total = len(results)
    n_fp = sum(1 for r in results if r.is_false_positive)
    n_tp = sum(1 for r in results if r.is_true_positive)
    n_tn = sum(1 for r in results if r.is_true_negative)
    n_fn = sum(1 for r in results
               if r.conservative_safe and not r.joint_smt_safe)
    
    fp_rate, fp_ci_low, fp_ci_high = wilson_ci(n_fp, total)
    
    # Shared-variable specific rates
    shared_results = [r for r in results if r.has_shared_vars]
    n_shared_total = len(shared_results)
    n_shared_fp = sum(1 for r in shared_results if r.is_false_positive)
    shared_fp_rate, shared_fp_ci_low, shared_fp_ci_high = wilson_ci(
        n_shared_fp, n_shared_total) if n_shared_total > 0 else (0, 0, 0)
    
    # Per-category breakdown
    category_breakdown = {}
    for cat, stats in sorted(category_stats.items()):
        cat_fp_rate, cat_ci_low, cat_ci_high = wilson_ci(
            stats['fp'], stats['total'])
        category_breakdown[cat] = {
            'total': stats['total'],
            'false_positives': stats['fp'],
            'true_positives': stats['tp'],
            'true_negatives': stats['tn'],
            'fp_rate': round(cat_fp_rate, 4),
            'fp_ci_95': [round(cat_ci_low, 4), round(cat_ci_high, 4)],
        }
    
    report = {
        'experiment': 'Compositional false positive rate analysis',
        'description': (
            'Compares conservative compositional analysis against joint SMT '
            'ground truth to quantify false positive rates. A false positive '
            'occurs when conservative analysis flags a program as unsafe but '
            'joint SMT analysis proves it safe.'
        ),
        'n_programs': len(programs),
        'n_architectures': len(archs),
        'total_analyses': total,
        'overall': {
            'false_positives': n_fp,
            'true_positives': n_tp,
            'true_negatives': n_tn,
            'fp_rate': round(fp_rate, 4),
            'fp_ci_95': [round(fp_ci_low, 4), round(fp_ci_high, 4)],
            'precision': round(n_tp / max(n_tp + n_fp, 1), 4),
            'soundness': 'verified' if n_fn == 0 else 'VIOLATED',
        },
        'shared_variable_analyses': {
            'total': n_shared_total,
            'false_positives': n_shared_fp,
            'fp_rate': round(shared_fp_rate, 4),
            'fp_ci_95': [round(shared_fp_ci_low, 4), round(shared_fp_ci_high, 4)],
        },
        'per_category': category_breakdown,
        'details': [
            {
                'program': r.program_name,
                'arch': r.arch,
                'category': r.interaction_category,
                'has_shared_vars': r.has_shared_vars,
                'conservative_safe': r.conservative_safe,
                'joint_smt_safe': r.joint_smt_safe,
                'is_false_positive': r.is_false_positive,
                'time_ms': round(r.time_ms, 2),
            }
            for r in results
        ],
    }
    
    with open(f'{output_dir}/false_positive_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"FALSE POSITIVE RATE ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total analyses:        {total}")
    print(f"False positives:       {n_fp}")
    print(f"True positives:        {n_tp}")
    print(f"True negatives:        {n_tn}")
    print(f"Overall FP rate:       {fp_rate:.1%} (95% CI: [{fp_ci_low:.1%}, {fp_ci_high:.1%}])")
    print(f"Shared-var FP rate:    {shared_fp_rate:.1%} (95% CI: [{shared_fp_ci_low:.1%}, {shared_fp_ci_high:.1%}])")
    print(f"Soundness:             {'✓ VERIFIED (0 false negatives)' if n_fn == 0 else '✗ VIOLATED'}")
    print(f"\nPer-category:")
    for cat, stats in sorted(category_breakdown.items()):
        print(f"  {cat:25s}: FP={stats['false_positives']}/{stats['total']} "
              f"({stats['fp_rate']:.1%})")
    
    return report


if __name__ == '__main__':
    report = run_false_positive_analysis()
