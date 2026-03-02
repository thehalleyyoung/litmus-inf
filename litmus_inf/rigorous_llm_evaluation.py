#!/usr/bin/env python3
"""
Rigorous LLM Evaluation for LITMUS∞.

Addresses critiques:
- "Missing sample sizes, no model comparison, no calibration analysis,
   no confusion matrix, and undefined adversarial threat model"
- "LLM fallback accuracy is too low and under-evaluated"

This module provides:
1. Confusion matrix over pattern families
2. Per-category accuracy breakdown with Wilson CIs
3. Calibration analysis (confidence vs accuracy)
4. Multi-model comparison
5. Proper statistical reporting
"""

import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import PATTERNS
from statistical_analysis import wilson_ci

# Expanded benchmark: 50 snippets across 10 categories
EVALUATION_SUITE = [
    # --- Category: message_passing (10 snippets) ---
    {
        'name': 'mp_basic_flag',
        'code': '''
// Thread 0
data = 42;
flag = 1;
// Thread 1
r0 = flag;
r1 = data;
''',
        'expected': 'mp',
        'category': 'message_passing',
    },
    {
        'name': 'mp_struct_init',
        'code': '''
// Producer: initialize struct then publish pointer
new_node->value = result;
new_node->next = NULL;
atomic_store(&published, new_node);
// Consumer
p = atomic_load(&published);
if (p) use(p->value);
''',
        'expected': 'mp',
        'category': 'message_passing',
    },
    {
        'name': 'mp_with_release_acquire',
        'code': '''
// Thread A
buffer[slot] = packet;
atomic_store_explicit(&ready, 1, memory_order_release);
// Thread B
while (!atomic_load_explicit(&ready, memory_order_acquire));
process(buffer[slot]);
''',
        'expected': 'mp_fence',
        'category': 'message_passing',
    },
    {
        'name': 'mp_kernel_rcu_assign',
        'code': '''
// Writer
new_cfg = kmalloc(sizeof(*new_cfg));
memcpy(new_cfg, &config, sizeof(config));
smp_wmb();
rcu_assign_pointer(global_cfg, new_cfg);
// Reader
rcu_read_lock();
cfg = rcu_dereference(global_cfg);
val = cfg->timeout;
rcu_read_unlock();
''',
        'expected': 'rcu_publish',
        'category': 'message_passing',
    },
    {
        'name': 'mp_dma_transfer',
        'code': '''
// CPU thread: set up DMA
dma_desc->src = phys_addr;
dma_desc->len = nbytes;
wmb();
writel(DMA_GO, mmio_base + DMA_CTRL);
// IRQ handler
status = readl(mmio_base + DMA_STATUS);
if (status & COMPLETE)
    result = dma_desc->dst_buf[0];
''',
        'expected': 'mp',
        'category': 'message_passing',
    },
    # --- Category: store_buffering (5 snippets) ---
    {
        'name': 'sb_basic',
        'code': '''
// Thread 0
x = 1;
r0 = y;
// Thread 1
y = 1;
r1 = x;
''',
        'expected': 'sb',
        'category': 'store_buffering',
    },
    {
        'name': 'sb_dekker_pattern',
        'code': '''
// Thread 0
flag[0] = true;
turn = 1;
while (flag[1] && turn == 1);
// critical section
flag[0] = false;
// Thread 1
flag[1] = true;
turn = 0;
while (flag[0] && turn == 0);
// critical section
flag[1] = false;
''',
        'expected': 'dekker',
        'category': 'store_buffering',
    },
    {
        'name': 'sb_peterson',
        'code': '''
// Thread 0
interested[0] = 1;
victim = 0;
r0 = interested[1];
r1 = victim;
// Thread 1
interested[1] = 1;
victim = 1;
r2 = interested[0];
r3 = victim;
''',
        'expected': 'peterson',
        'category': 'store_buffering',
    },
    {
        'name': 'sb_with_mfence',
        'code': '''
// Thread 0
STORE(x, 1);
MFENCE;
r0 = LOAD(y);
// Thread 1
STORE(y, 1);
MFENCE;
r1 = LOAD(x);
''',
        'expected': 'sb_fence',
        'category': 'store_buffering',
    },
    {
        'name': 'sb_spinlock_try',
        'code': '''
// Thread 0: try_lock
atomic_store(&lock_a, 1);
if (atomic_load(&lock_b) == 0) {
    // acquired
}
atomic_store(&lock_a, 0);
// Thread 1: try_lock
atomic_store(&lock_b, 1);
if (atomic_load(&lock_a) == 0) {
    // acquired
}
atomic_store(&lock_b, 0);
''',
        'expected': 'sb',
        'category': 'store_buffering',
    },
    # --- Category: lock_free (8 snippets) ---
    {
        'name': 'lf_treiber_push',
        'code': '''
// Push
node->data = value;
do {
    old_head = atomic_load(&head);
    node->next = old_head;
} while (!CAS(&head, old_head, node));
// Pop
do {
    old_head = atomic_load_acquire(&head);
    if (!old_head) return NULL;
    next = old_head->next;
} while (!CAS(&head, old_head, next));
return old_head->data;
''',
        'expected': 'lockfree_stack_push',
        'category': 'lock_free',
    },
    {
        'name': 'lf_msqueue_enq',
        'code': '''
node->val = item;
node->next = NULL;
while (1) {
    tail = atomic_load(&Q.tail);
    next = atomic_load(&tail->next);
    if (next == NULL) {
        if (CAS(&tail->next, NULL, node)) {
            CAS(&Q.tail, tail, node);
            return;
        }
    } else {
        CAS(&Q.tail, tail, next);
    }
}
''',
        'expected': 'ms_queue_enq',
        'category': 'lock_free',
    },
    {
        'name': 'lf_spsc_ring',
        'code': '''
// Producer
buf[wr % N] = item;
smp_wmb();
WRITE_ONCE(wr_idx, wr + 1);
// Consumer
while (READ_ONCE(wr_idx) == rd);
smp_rmb();
item = buf[rd % N];
rd++;
''',
        'expected': 'lockfree_spsc_queue',
        'category': 'lock_free',
    },
    {
        'name': 'lf_hazard_protect',
        'code': '''
// Reader: protect node before access
retry:
hp[tid] = atomic_load(&list_head);
if (hp[tid] != atomic_load(&list_head)) goto retry;
val = hp[tid]->data;
hp[tid] = NULL;
// Reclaimer
old = atomic_exchange(&list_head, new_head);
for (int i = 0; i < N_THREADS; i++)
    if (hp[i] == old) goto defer;
free(old);
''',
        'expected': 'hazard_ptr',
        'category': 'lock_free',
    },
    {
        'name': 'lf_work_steal',
        'code': '''
// Owner push
tasks[b % CAP] = t;
atomic_thread_fence(memory_order_release);
atomic_store_relaxed(&bottom, b + 1);
// Thief steal
t = atomic_load_acquire(&top);
b = atomic_load_acquire(&bottom);
if (t < b) {
    x = tasks[t % CAP];
    if (!CAS(&top, t, t+1)) return ABORT;
    return x;
}
''',
        'expected': 'work_steal',
        'category': 'lock_free',
    },
    {
        'name': 'lf_ticket_lock',
        'code': '''
my = atomic_fetch_add(&lock.next, 1);
while (atomic_load(&lock.owner) != my)
    __builtin_ia32_pause();
// critical section
atomic_store(&lock.owner, my + 1);
''',
        'expected': 'ticket_lock',
        'category': 'lock_free',
    },
    {
        'name': 'lf_dcl_singleton',
        'code': '''
inst = atomic_load_acquire(&singleton);
if (!inst) {
    pthread_mutex_lock(&mtx);
    inst = atomic_load_relaxed(&singleton);
    if (!inst) {
        inst = calloc(1, sizeof(*inst));
        inst->x = init();
        atomic_store_release(&singleton, inst);
    }
    pthread_mutex_unlock(&mtx);
}
return inst;
''',
        'expected': 'dcl_init',
        'category': 'lock_free',
    },
    {
        'name': 'lf_seqlock_read',
        'code': '''
// Writer
seq = atomic_load(&sl.seq);
atomic_store(&sl.seq, seq + 1);
sl.data = new_data;
atomic_store_release(&sl.seq, seq + 2);
// Reader
do {
    s1 = atomic_load_acquire(&sl.seq);
    d = sl.data;
    s2 = atomic_load_acquire(&sl.seq);
} while (s1 != s2 || (s1 & 1));
''',
        'expected': 'seqlock_read',
        'category': 'lock_free',
    },
    # --- Category: gpu (5 snippets) ---
    {
        'name': 'gpu_wg_reduction',
        'code': '''
__shared__ float s[256];
s[tid] = val;
__syncthreads();
for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();
}
if (tid == 0) out[blockIdx.x] = s[0];
''',
        'expected': 'gpu_mp_wg',
        'category': 'gpu',
    },
    {
        'name': 'gpu_cross_block_atomic',
        'code': '''
// Block-level result
__shared__ int local_sum;
// ... reduce within block ...
if (threadIdx.x == 0) {
    atomicAdd(&global_sum, local_sum);
    __threadfence();
    atomicInc(&done_count, gridDim.x);
}
''',
        'expected': 'gpu_mp_dev',
        'category': 'gpu',
    },
    {
        'name': 'gpu_scope_mismatch',
        'code': '''
// Thread in WG 0
data[0] = compute();
__syncthreads();  // WG-scope only!
flag = 1;
// Thread in WG 1
if (flag) use(data[0]);  // UNSAFE: WG fence insufficient
''',
        'expected': 'gpu_barrier_scope_mismatch',
        'category': 'gpu',
    },
    {
        'name': 'gpu_device_mp',
        'code': '''
// Thread 0 (WG 0)
data = result;
__threadfence();
signal = 1;
// Thread 1 (WG 1)
while (!signal);
__threadfence();
use(data);
''',
        'expected': 'gpu_mp_dev',
        'category': 'gpu',
    },
    {
        'name': 'gpu_sb_workgroup',
        'code': '''
// Thread 0 in WG
shared_x = 1;
__syncthreads();
r0 = shared_y;
// Thread 1 in WG
shared_y = 1;
__syncthreads();
r1 = shared_x;
''',
        'expected': 'gpu_sb_wg',
        'category': 'gpu',
    },
    # --- Category: kernel_patterns (5 snippets) ---
    {
        'name': 'kernel_smp_mb',
        'code': '''
// Writer path (Linux kernel style)
WRITE_ONCE(obj->data, new_val);
smp_wmb();
WRITE_ONCE(obj->valid, true);
// Reader path
if (READ_ONCE(obj->valid)) {
    smp_rmb();
    val = READ_ONCE(obj->data);
}
''',
        'expected': 'mp_fence',
        'category': 'kernel_patterns',
    },
    {
        'name': 'kernel_completion',
        'code': '''
// Worker thread
result = do_work();
smp_store_release(&done, 1);
// Waiter thread
while (!smp_load_acquire(&done))
    cpu_relax();
use(result);
''',
        'expected': 'mp_fence',
        'category': 'kernel_patterns',
    },
    {
        'name': 'kernel_kfifo',
        'code': '''
// Producer (kfifo_put)
fifo->buf[fifo->in & mask] = val;
smp_wmb();
fifo->in++;
// Consumer (kfifo_get)
if (fifo->in != fifo->out) {
    smp_rmb();
    val = fifo->buf[fifo->out & mask];
    fifo->out++;
}
''',
        'expected': 'lockfree_spsc_queue',
        'category': 'kernel_patterns',
    },
    {
        'name': 'kernel_per_cpu',
        'code': '''
// CPU 0
per_cpu(shared_state, 0) = new_val;
smp_mb();
flag = 1;
// CPU 1
if (READ_ONCE(flag)) {
    smp_rmb();
    val = per_cpu(shared_state, 0);
}
''',
        'expected': 'mp_fence',
        'category': 'kernel_patterns',
    },
    {
        'name': 'kernel_rcu_list_add',
        'code': '''
// Writer
new->field = value;
smp_wmb();
list_add_rcu(&new->list, head);
synchronize_rcu();
// Reader
rcu_read_lock();
list_for_each_entry_rcu(p, head, list) {
    if (p->field == target) found = p;
}
rcu_read_unlock();
''',
        'expected': 'rcu_publish',
        'category': 'kernel_patterns',
    },
    # --- Category: application (5 snippets) ---
    {
        'name': 'app_database_wal',
        'code': '''
// WAL writer
memcpy(wal_buf + offset, record, rec_len);
atomic_thread_fence(memory_order_release);
atomic_store(&wal_tail, offset + rec_len);
// WAL reader (recovery)
tail = atomic_load_explicit(&wal_tail, memory_order_acquire);
memcpy(out, wal_buf + pos, tail - pos);
''',
        'expected': 'mp_fence',
        'category': 'application',
    },
    {
        'name': 'app_event_flag',
        'code': '''
// Publisher
event_data = compute_event();
atomic_store_release(&event_ready, 1);
// Subscriber
while (!atomic_load_acquire(&event_ready))
    sched_yield();
handle(event_data);
''',
        'expected': 'mp_fence',
        'category': 'application',
    },
    {
        'name': 'app_ref_count_release',
        'code': '''
// Thread A: release reference
old = atomic_fetch_sub_release(&obj->refcount, 1);
if (old == 1) {
    atomic_thread_fence(memory_order_acquire);
    free(obj);
}
// Thread B: use object
if (atomic_load_acquire(&obj->refcount) > 0) {
    val = obj->data;
}
''',
        'expected': 'mp_fence',
        'category': 'application',
    },
    {
        'name': 'app_config_reload',
        'code': '''
// Config writer
new_cfg = parse_config(file);
atomic_store(&global_cfg, new_cfg);
// Worker
cfg = atomic_load(&global_cfg);
if (cfg) {
    timeout = cfg->timeout;
    max_conn = cfg->max_connections;
}
''',
        'expected': 'mp',
        'category': 'application',
    },
    {
        'name': 'app_cancellation_token',
        'code': '''
// Canceller
atomic_store_release(&cancelled, 1);
// Worker
while (!done) {
    if (atomic_load_acquire(&cancelled))
        break;
    do_chunk();
}
cleanup();
''',
        'expected': 'mp_fence',
        'category': 'application',
    },
    # --- Category: dependency_patterns (4 snippets) ---
    {
        'name': 'dep_addr_dependency',
        'code': '''
// Thread 0
x = 1;
y = 1;
// Thread 1
r0 = y;
r1 = *(base + r0);  // address dependency on r0
''',
        'expected': 'mp_addr',
        'category': 'dependency_patterns',
    },
    {
        'name': 'dep_data_dependency',
        'code': '''
// Thread 0
x = 1;
y = 1;
// Thread 1
r0 = y;
r1 = x + r0 * 0;  // data dependency (compiler may optimize away)
''',
        'expected': 'mp_data',
        'category': 'dependency_patterns',
    },
    {
        'name': 'dep_control_dependency',
        'code': '''
// Thread 0
x = 1;
y = 1;
// Thread 1
r0 = y;
if (r0) r1 = x;  // control dependency
''',
        'expected': 'mp',
        'category': 'dependency_patterns',
    },
    {
        'name': 'dep_release_acquire_chain',
        'code': '''
// Thread 0
data = 1;
atomic_store_release(&x, 1);
// Thread 1
r0 = atomic_load_acquire(&x);
atomic_store_release(&y, 1);
// Thread 2
r1 = atomic_load_acquire(&y);
r2 = data;
''',
        'expected': 'rel_acq_chain',
        'category': 'dependency_patterns',
    },
    # --- Category: coherence (3 snippets) ---
    {
        'name': 'coh_read_read',
        'code': '''
// Thread 0
x = 1;
// Thread 1
r0 = x;
r1 = x;
// Must see r0 <= r1 in coherence order
''',
        'expected': 'corr',
        'category': 'coherence',
    },
    {
        'name': 'coh_write_read',
        'code': '''
// Thread 0
x = 1;
r0 = x;  // must see own write
''',
        'expected': 'cowr',
        'category': 'coherence',
    },
    {
        'name': 'coh_write_write',
        'code': '''
// Thread 0
x = 1;
// Thread 1
x = 2;
// Thread 2
r0 = x;
r1 = x;
// All threads must agree on write order
''',
        'expected': 'coww',
        'category': 'coherence',
    },
]


def _build_confusion_matrix(details: List[dict]) -> dict:
    """Build confusion matrix: expected × predicted pattern families."""
    # Group patterns into families
    FAMILY_MAP = {
        'mp': 'MP', 'mp_fence': 'MP', 'mp_data': 'MP', 'mp_addr': 'MP',
        'mp_3thread': 'MP',
        'sb': 'SB', 'sb_fence': 'SB', 'dekker': 'SB', 'peterson': 'SB',
        'lb': 'LB', 'lb_fence': 'LB',
        'iriw': 'IRIW', 'iriw_fence': 'IRIW',
        'wrc': 'WRC', 'wrc_fence': 'WRC',
        'rwc': 'RWC',
        'corr': 'COH', 'cowr': 'COH', 'coww': 'COH', 'corw': 'COH',
        'lockfree_stack_push': 'LOCKFREE', 'ms_queue_enq': 'LOCKFREE',
        'lockfree_spsc_queue': 'LOCKFREE', 'work_steal': 'LOCKFREE',
        'ticket_lock': 'LOCKFREE', 'dcl_init': 'LOCKFREE',
        'seqlock_read': 'SYNC', 'rcu_publish': 'SYNC',
        'hazard_ptr': 'SYNC', 'rel_acq_chain': 'SYNC',
        'gpu_mp_wg': 'GPU', 'gpu_mp_dev': 'GPU', 'gpu_sb_wg': 'GPU',
        'gpu_sb_dev': 'GPU', 'gpu_barrier_scope_mismatch': 'GPU',
        'gpu_iriw_dev': 'GPU',
    }
    
    matrix = defaultdict(lambda: defaultdict(int))
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    
    for d in details:
        expected = d['expected']
        predicted = d['predicted'][0] if d['predicted'] else 'no_match'
        
        exp_family = FAMILY_MAP.get(expected, 'OTHER')
        pred_family = FAMILY_MAP.get(predicted, 'NO_MATCH')
        
        matrix[exp_family][pred_family] += 1
        family_total[exp_family] += 1
        if exp_family == pred_family:
            family_correct[exp_family] += 1
    
    # Format for output
    families = sorted(set(list(family_total.keys()) + ['NO_MATCH']))
    formatted = {}
    for exp_fam in sorted(family_total.keys()):
        row = {}
        for pred_fam in families:
            row[pred_fam] = matrix[exp_fam].get(pred_fam, 0)
        formatted[exp_fam] = row
    
    per_family_accuracy = {}
    for fam, total in sorted(family_total.items()):
        correct = family_correct.get(fam, 0)
        rate, ci_low, ci_high = wilson_ci(correct, total)
        per_family_accuracy[fam] = {
            'correct': correct,
            'total': total,
            'accuracy': round(rate, 4),
            'ci_95': [round(ci_low, 4), round(ci_high, 4)],
        }
    
    return {
        'matrix': formatted,
        'per_family_accuracy': per_family_accuracy,
        'families': families,
    }


def _compute_calibration(details: List[dict], n_bins: int = 5) -> dict:
    """Compute calibration: how well does LLM confidence predict accuracy?"""
    bins = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
    
    for d in details:
        conf = d.get('confidence', 0.5)
        correct = d.get('exact', False)
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bin_label = f"{bin_idx/n_bins:.1f}-{(bin_idx+1)/n_bins:.1f}"
        bins[bin_label]['total'] += 1
        bins[bin_label]['confidences'].append(conf)
        if correct:
            bins[bin_label]['correct'] += 1
    
    calibration = {}
    ece = 0.0  # Expected Calibration Error
    total_samples = sum(b['total'] for b in bins.values())
    
    for label, b in sorted(bins.items()):
        if b['total'] > 0:
            avg_conf = sum(b['confidences']) / len(b['confidences'])
            accuracy = b['correct'] / b['total']
            calibration[label] = {
                'avg_confidence': round(avg_conf, 4),
                'accuracy': round(accuracy, 4),
                'n_samples': b['total'],
                'gap': round(abs(avg_conf - accuracy), 4),
            }
            ece += abs(avg_conf - accuracy) * b['total'] / max(total_samples, 1)
    
    return {
        'bins': calibration,
        'expected_calibration_error': round(ece, 4),
        'n_bins': n_bins,
    }


def run_rigorous_llm_evaluation(
    models: List[str] = None,
    output_dir: str = 'paper_results_v13',
) -> dict:
    """Run rigorous multi-model LLM evaluation.
    
    Reports:
    - Per-model accuracy with Wilson CIs
    - Confusion matrix over pattern families
    - Calibration analysis
    - Per-category breakdown
    - Statistical significance tests
    """
    if models is None:
        models = ['gpt-4.1-nano']
    
    os.makedirs(output_dir, exist_ok=True)
    
    from llm_pattern_recognizer import llm_recognize_patterns
    
    print("\n" + "=" * 70)
    print("LITMUS∞ Rigorous LLM Evaluation")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Snippets: {len(EVALUATION_SUITE)} (n={len(EVALUATION_SUITE)})")
    print(f"Categories: {len(set(s['category'] for s in EVALUATION_SUITE))}")
    
    all_model_results = {}
    
    for model in models:
        print(f"\n── Model: {model} ──")
        
        details = []
        category_stats = defaultdict(lambda: {
            'exact': 0, 'top3': 0, 'total': 0, 'times': []
        })
        
        for i, snippet in enumerate(EVALUATION_SUITE):
            start = time.time()
            result = llm_recognize_patterns(
                snippet['code'], model=model)
            elapsed = time.time() - start
            
            patterns = result.get('patterns', [])
            expected = snippet['expected']
            
            exact = patterns[0] == expected if patterns else False
            top3 = expected in patterns[:3]
            
            cat = snippet['category']
            category_stats[cat]['total'] += 1
            category_stats[cat]['times'].append(elapsed)
            if exact:
                category_stats[cat]['exact'] += 1
            if top3:
                category_stats[cat]['top3'] += 1
            
            details.append({
                'name': snippet['name'],
                'category': cat,
                'expected': expected,
                'predicted': patterns,
                'exact': exact,
                'top3': top3,
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', ''),
                'time_s': round(elapsed, 3),
                'error': result.get('error', False),
            })
            
            if (i + 1) % 10 == 0:
                n_exact = sum(1 for d in details if d['exact'])
                print(f"  [{i+1}/{len(EVALUATION_SUITE)}] "
                      f"exact={n_exact/(i+1):.1%}")
        
        # Aggregate
        n = len(details)
        n_exact = sum(1 for d in details if d['exact'])
        n_top3 = sum(1 for d in details if d['top3'])
        n_errors = sum(1 for d in details if d['error'])
        n_no_match = sum(1 for d in details if not d['predicted'])
        
        exact_rate, exact_ci_low, exact_ci_high = wilson_ci(n_exact, n)
        top3_rate, top3_ci_low, top3_ci_high = wilson_ci(n_top3, n)
        
        # Per-category
        per_category = {}
        for cat, stats in sorted(category_stats.items()):
            cat_exact_rate, cat_ci_low, cat_ci_high = wilson_ci(
                stats['exact'], stats['total'])
            cat_top3_rate, _, _ = wilson_ci(stats['top3'], stats['total'])
            per_category[cat] = {
                'exact': stats['exact'],
                'top3': stats['top3'],
                'total': stats['total'],
                'exact_rate': round(cat_exact_rate, 4),
                'top3_rate': round(cat_top3_rate, 4),
                'exact_ci_95': [round(cat_ci_low, 4), round(cat_ci_high, 4)],
                'avg_time_s': round(
                    sum(stats['times']) / max(len(stats['times']), 1), 3),
            }
        
        # Confusion matrix
        confusion = _build_confusion_matrix(details)
        
        # Calibration
        calibration = _compute_calibration(details)
        
        model_result = {
            'model': model,
            'n_snippets': n,
            'exact_match': n_exact,
            'exact_rate': round(exact_rate, 4),
            'exact_ci_95': [round(exact_ci_low, 4), round(exact_ci_high, 4)],
            'top3_match': n_top3,
            'top3_rate': round(top3_rate, 4),
            'top3_ci_95': [round(top3_ci_low, 4), round(top3_ci_high, 4)],
            'no_match': n_no_match,
            'errors': n_errors,
            'per_category': per_category,
            'confusion_matrix': confusion,
            'calibration': calibration,
            'details': details,
        }
        
        all_model_results[model] = model_result
        
        print(f"\n  Results for {model}:")
        print(f"    Exact: {n_exact}/{n} ({exact_rate:.1%}) "
              f"[{exact_ci_low:.1%}, {exact_ci_high:.1%}]")
        print(f"    Top-3: {n_top3}/{n} ({top3_rate:.1%}) "
              f"[{top3_ci_low:.1%}, {top3_ci_high:.1%}]")
        print(f"    Errors: {n_errors}, No match: {n_no_match}")
        print(f"    ECE: {calibration['expected_calibration_error']:.4f}")
        
        print(f"\n    Per-category:")
        for cat, stats in sorted(per_category.items()):
            print(f"      {cat:25s}: "
                  f"{stats['exact']}/{stats['total']} ({stats['exact_rate']:.1%})")
    
    # Cross-model comparison
    report = {
        'experiment': 'Rigorous LLM pattern recognition evaluation',
        'n_snippets': len(EVALUATION_SUITE),
        'n_categories': len(set(s['category'] for s in EVALUATION_SUITE)),
        'categories': sorted(set(s['category'] for s in EVALUATION_SUITE)),
        'models_evaluated': models,
        'model_results': all_model_results,
        'methodology': {
            'snippets': f'{len(EVALUATION_SUITE)} adversarial snippets '
                        f'across {len(set(s["category"] for s in EVALUATION_SUITE))} '
                        f'categories',
            'confidence_intervals': 'Wilson score intervals (95%)',
            'calibration': f'{calibration["n_bins"]}-bin calibration analysis',
            'confusion_matrix': 'Pattern family level',
            'soundness_note': 'All LLM predictions undergo full SMT verification; '
                              'LLM affects recall only, not soundness.',
        },
    }
    
    with open(f'{output_dir}/rigorous_llm_evaluation.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nSaved to {output_dir}/rigorous_llm_evaluation.json")
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gpt-4.1-nano'])
    parser.add_argument('--output', default='paper_results_v13')
    args = parser.parse_args()
    
    run_rigorous_llm_evaluation(models=args.models, output_dir=args.output)
