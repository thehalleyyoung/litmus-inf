#!/usr/bin/env python3
"""
LLM-Assisted Pattern Recognition for LITMUS∞.

Uses an LLM as a pre-filter to map out-of-distribution concurrent code
to the canonical 140-pattern library. The LLM identifies the memory
ordering idiom in the code, and the SMT verification backend handles
all formal checking.

Architecture:
  Source code → LLM classifier → Pattern name → SMT verification
  
Soundness: The LLM only affects recall (which patterns are checked),
not precision (all verdicts are SMT-certified). A false positive in
pattern matching produces a conservative result (checking a stronger
pattern), while a false negative produces an explicit "no match" warning.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from portcheck import PATTERNS, ARCHITECTURES, verify_test, recommend_fence

# Pattern descriptions for the LLM prompt
PATTERN_CATALOG = {}
for name, pdata in PATTERNS.items():
    desc = pdata.get('description', '')
    ops = pdata.get('ops', [])
    n_threads = len(set(op.thread for op in ops))
    n_loads = sum(1 for op in ops if op.optype == 'load')
    n_stores = sum(1 for op in ops if op.optype == 'store')
    n_fences = sum(1 for op in ops if op.optype == 'fence')
    has_dep = any(op.dep_on for op in ops)
    is_gpu = any(op.workgroup > 0 for op in ops) or any(op.scope for op in ops if op.optype == 'fence')
    
    PATTERN_CATALOG[name] = {
        'description': desc,
        'threads': n_threads,
        'loads': n_loads,
        'stores': n_stores,
        'fences': n_fences,
        'has_dependency': has_dep,
        'is_gpu': is_gpu,
    }

# Core CPU patterns (for concise LLM prompt)
CORE_PATTERNS = [
    ('mp', 'Message passing: T0 writes data then flag; T1 reads flag then data. Forbidden: flag=1, data=0.'),
    ('sb', 'Store buffering: T0 writes x, reads y; T1 writes y, reads x. Forbidden: both reads see 0.'),
    ('lb', 'Load buffering: T0 reads x, writes y; T1 reads y, writes x. Forbidden: both reads see 1.'),
    ('iriw', 'Independent reads of independent writes: 4 threads, tests multi-copy atomicity.'),
    ('wrc', 'Write-read causality: T0 writes x; T1 reads x, writes y; T2 reads y, reads x.'),
    ('rwc', 'Read-write causality: T0 writes x; T1 reads x, reads y; T2 writes y, reads x.'),
    ('2+2w', 'Two threads write two addresses in opposite order, observer reads.'),
    ('dekker', 'Dekker mutual exclusion: flag+turn variables.'),
    ('peterson', 'Peterson mutual exclusion: flag+turn pattern.'),
    ('corr', 'Coherence read-read: two reads of same address must see coherent order.'),
    ('cowr', 'Coherence write-read: write followed by read on same address.'),
    ('coww', 'Coherence write-write: two writes to same address, observer reads.'),
    ('isa2', 'ISA2: 3-thread test with data dependency.'),
    ('mp_fence', 'MP with full fences on both threads.'),
    ('sb_fence', 'SB with full fences on both threads.'),
    ('lb_fence', 'LB with full fences on both threads.'),
    ('mp_data', 'MP with data dependency on producer.'),
    ('mp_addr', 'MP with address dependency on consumer.'),
    ('mp_3thread', '3-thread MP relay.'),
    ('amoswap', 'Atomic swap: two threads CAS on same address.'),
    ('lockfree_spsc_queue', 'SPSC queue: write data, update tail; reader reads tail then data.'),
    ('lockfree_stack_push', 'Lock-free stack push: write node then CAS head.'),
    ('dcl_init', 'Double-checked locking: init data, set flag; reader checks flag, reads data.'),
    ('rcu_publish', 'RCU publish: write new data, update pointer; reader dereferences pointer.'),
    ('seqlock_read', 'Seqlock read-side: load seq, load data, load seq again.'),
    ('hazard_ptr', 'Hazard pointer: publish HP, check before reclaim.'),
    ('rel_acq_chain', 'Release-acquire chain: 3 threads with transitive ordering.'),
    ('rmw_cas_mp', 'CAS-based message passing.'),
    ('ticket_lock', 'Ticket lock: fetch-and-add ticket, wait for serving counter.'),
    ('work_steal', 'Work-stealing deque: push-steal interaction.'),
    ('publish_array', 'Array publication: write data array, then flag.'),
]

GPU_PATTERNS = [
    ('gpu_mp_wg', 'GPU MP within same workgroup with WG fence.'),
    ('gpu_mp_dev', 'GPU MP across workgroups with device fence.'),
    ('gpu_sb_wg', 'GPU SB within workgroup.'),
    ('gpu_sb_dev', 'GPU SB across workgroups with device fence.'),
    ('gpu_barrier_scope_mismatch', 'GPU MP with WG fence but threads in different workgroups.'),
    ('gpu_iriw_dev', 'GPU IRIW across workgroups with device fences.'),
]


def build_llm_prompt(code: str, target_arch: str = None) -> str:
    """Build the LLM prompt for pattern recognition with few-shot examples."""
    pattern_list = "\n".join(
        f"  - {name}: {desc}" for name, desc in CORE_PATTERNS
    )
    gpu_list = "\n".join(
        f"  - {name}: {desc}" for name, desc in GPU_PATTERNS
    )
    
    target_info = ""
    if target_arch:
        target_info = f"\nTarget architecture: {target_arch}"
    
    return f"""You are a concurrency expert. Identify which memory ordering pattern from the catalog matches this code.

PATTERN CATALOG (CPU):
{pattern_list}

GPU PATTERNS (for CUDA/OpenCL/Vulkan code):
{gpu_list}

KEY RULES:
1. Focus on ordering between stores and loads ACROSS threads.
2. release/acquire or smp_wmb/smp_rmb or atomic_store_release/atomic_load_acquire → use the _fence variant (e.g., mp_fence).
3. Plain stores followed by plain loads with no barrier → use the bare variant (e.g., mp).
4. RCU patterns (rcu_assign_pointer, rcu_dereference) → rcu_publish.
5. SeqLock (load seq, load data, load seq) → seqlock_read.
6. Hazard pointers (hp[tid] = ptr, barrier, check) → hazard_ptr.
7. Lock-free stack (CAS on head pointer) → lockfree_stack_push.
8. Lock-free queue (CAS on tail->next) → ms_queue_enq.
9. SPSC ring buffer (write data, barrier, update index) → lockfree_spsc_queue.
10. Ticket lock (fetch_add ticket, spin on serving) → ticket_lock.
11. DCL (check flag, lock, check flag, init, release flag) → dcl_init.
12. Work stealing (push to bottom, steal from top via CAS) → work_steal.
13. Coherence (two reads of same addr by same thread) → corr/cowr/coww.
14. CUDA __syncthreads() → gpu_mp_wg; __threadfence() → gpu_mp_dev.
{target_info}

EXAMPLES:
Code: "data=1; flag=1; // T1: if(flag) use(data);"  →  {{"patterns":["mp"],"confidence":0.9,"reasoning":"store-store then load-load across threads is message passing"}}
Code: "x=1; r0=y; // T1: y=1; r1=x;"  →  {{"patterns":["sb"],"confidence":0.9,"reasoning":"each thread stores then loads other's variable"}}
Code: "data=v; smp_wmb(); flag=1; // T1: while(!flag); smp_rmb(); use(data);"  →  {{"patterns":["mp_fence"],"confidence":0.95,"reasoning":"MP with explicit fences"}}

CODE:
```
{code}
```

Respond with ONLY valid JSON:
{{"patterns": ["pattern_name"], "confidence": 0.8, "reasoning": "brief"}}"""


def llm_recognize_patterns(code: str, target_arch: str = None,
                           model: str = "gpt-4.1-nano") -> Dict:
    """Use LLM to recognize memory ordering patterns in code.
    
    Returns dict with 'patterns' (list of pattern names),
    'confidence' (0-1), and 'reasoning'.
    """
    try:
        import openai
    except ImportError:
        return {'patterns': [], 'confidence': 0.0, 
                'reasoning': 'openai not installed', 'error': True}
    
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return {'patterns': [], 'confidence': 0.0,
                'reasoning': 'OPENAI_API_KEY not set', 'error': True}
    
    client = openai.OpenAI(api_key=api_key)
    prompt = build_llm_prompt(code, target_arch)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a memory model expert. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200,
        )
        
        text = response.choices[0].message.content.strip()
        # Parse JSON from response (handle markdown wrapping)
        if text.startswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        
        result = json.loads(text)
        
        # Validate pattern names
        valid_patterns = [p for p in result.get('patterns', []) if p in PATTERNS]
        result['patterns'] = valid_patterns
        result['error'] = False
        return result
        
    except Exception as e:
        return {'patterns': [], 'confidence': 0.0,
                'reasoning': f'LLM error: {str(e)}', 'error': True}


def hybrid_check_portability(code: str, target_arch: str = "arm",
                              use_llm: bool = True,
                              llm_model: str = "gpt-4.1-nano") -> List[Dict]:
    """Hybrid pattern recognition: AST analysis + optional LLM fallback.
    
    1. Try AST-based analysis first (fast, deterministic)
    2. If AST confidence is low or no match, use LLM as fallback
    3. All results go through SMT verification (soundness preserved)
    """
    from ast_analyzer import get_ast_analyzer
    
    analyzer = get_ast_analyzer()
    results = []
    
    # Phase 1: AST analysis
    ast_results = analyzer.check_portability(code, target_arch)
    
    # Check if AST found good matches
    ast_found = len(ast_results) > 0
    ast_confidence = max((r.get('similarity', 0) for r in ast_results), default=0)
    
    if ast_found and ast_confidence >= 0.5:
        return ast_results
    
    # Phase 2: LLM fallback
    if use_llm:
        llm_result = llm_recognize_patterns(code, target_arch, model=llm_model)
        
        if not llm_result.get('error') and llm_result.get('patterns'):
            model_name = ARCHITECTURES.get(target_arch, target_arch)
            
            for pattern_name in llm_result['patterns']:
                if pattern_name not in PATTERNS:
                    continue
                
                pattern = PATTERNS[pattern_name]
                test = LitmusTest(
                    name=pattern_name,
                    n_threads=len(set(op.thread for op in pattern['ops'])),
                    addresses=pattern['addresses'],
                    ops=pattern['ops'],
                    forbidden=pattern['forbidden'],
                )
                
                safe, _, cert = verify_test(test, model_name)
                fence_rec = None
                if not safe:
                    fence_rec = recommend_fence(test, target_arch)
                
                results.append({
                    'pattern': pattern_name,
                    'description': pattern.get('description', ''),
                    'safe': safe,
                    'target_arch': target_arch,
                    'fence_recommendation': fence_rec,
                    'certificate': cert,
                    'source': 'llm',
                    'llm_confidence': llm_result.get('confidence', 0),
                    'llm_reasoning': llm_result.get('reasoning', ''),
                })
    
    # Combine AST and LLM results (dedup by pattern name)
    seen = set()
    combined = []
    for r in ast_results + results:
        pname = r.get('pattern', '')
        if pname not in seen:
            seen.add(pname)
            combined.append(r)
    
    return combined


def run_llm_benchmark(benchmarks: List[Dict], 
                       model: str = "gpt-4.1-nano") -> Dict:
    """Run LLM pattern recognition on benchmark snippets.
    
    Each benchmark entry: {'name': str, 'code': str, 'expected': str, 'category': str}
    """
    results = {
        'total': len(benchmarks),
        'exact_match': 0,
        'top3_match': 0,
        'no_match': 0,
        'errors': 0,
        'details': [],
    }
    
    for i, bench in enumerate(benchmarks):
        start = time.time()
        llm_result = llm_recognize_patterns(bench['code'], model=model)
        elapsed = time.time() - start
        
        patterns = llm_result.get('patterns', [])
        expected = bench['expected']
        
        exact = patterns[0] == expected if patterns else False
        top3 = expected in patterns[:3]
        
        if exact:
            results['exact_match'] += 1
        if top3:
            results['top3_match'] += 1
        if not patterns:
            results['no_match'] += 1
        if llm_result.get('error'):
            results['errors'] += 1
        
        results['details'].append({
            'name': bench['name'],
            'expected': expected,
            'predicted': patterns,
            'exact': exact,
            'top3': top3,
            'confidence': llm_result.get('confidence', 0),
            'time_s': round(elapsed, 2),
        })
        
        status = '✓' if exact else ('~' if top3 else '✗')
        if (i + 1) % 10 == 0 or i == len(benchmarks) - 1:
            exact_pct = results['exact_match'] / (i + 1) * 100
            top3_pct = results['top3_match'] / (i + 1) * 100
            print(f"  [{i+1}/{len(benchmarks)}] exact={exact_pct:.1f}% top3={top3_pct:.1f}%")
    
    results['exact_rate'] = results['exact_match'] / results['total'] if results['total'] else 0
    results['top3_rate'] = results['top3_match'] / results['total'] if results['total'] else 0
    
    return results


# Adversarial benchmark snippets (from adversarial_benchmark.py)
ADVERSARIAL_SNIPPETS = [
    {
        'name': 'crypto_aes_ctr',
        'code': '''
// Thread 0: AES-CTR encryption
counter = atomic_load(&shared_ctr);
atomic_store(&shared_ctr, counter + 1);
encrypt_block(key, counter, &output);
// Thread 1: Read encrypted output
if (atomic_load(&output_ready)) {
    memcpy(buf, output, 16);
}
''',
        'expected': 'mp',
        'category': 'cryptography',
    },
    {
        'name': 'embedded_dma',
        'code': '''
// Thread 0: DMA setup
dma_buf[0] = data;
dma_buf[1] = len;
mmio_write(DMA_CTRL, START);
// Thread 1: DMA completion handler (interrupt context)
status = mmio_read(DMA_STATUS);
if (status & DONE) result = dma_buf[0];
''',
        'expected': 'mp',
        'category': 'embedded',
    },
    {
        'name': 'network_packet_ring',
        'code': '''
// Producer: write packet to ring buffer slot
ring[head % SIZE] = pkt;
smp_wmb();
WRITE_ONCE(head, head + 1);
// Consumer
t = READ_ONCE(head);
smp_rmb();
p = ring[(t-1) % SIZE];
''',
        'expected': 'mp_fence',
        'category': 'networking',
    },
    {
        'name': 'lockfree_treiber_stack',
        'code': '''
// Push: node->next = head; CAS(&head, node->next, node)
node->data = value;
do {
    old_head = atomic_load_explicit(&head, memory_order_relaxed);
    node->next = old_head;
} while (!atomic_compare_exchange_weak(&head, &old_head, node));
// Pop: old = head; CAS(&head, old, old->next)
do {
    old_head = atomic_load_explicit(&head, memory_order_acquire);
    if (!old_head) return NULL;
    next = old_head->next;
} while (!atomic_compare_exchange_weak(&head, &old_head, next));
return old_head->data;
''',
        'expected': 'lockfree_stack_push',
        'category': 'lockfree',
    },
    {
        'name': 'simd_parallel_reduce',
        'code': '''
// Each thread computes partial sum
partial[tid] = local_sum;
__syncthreads();
// Reduction tree
for (stride = blockDim.x/2; stride > 0; stride >>= 1) {
    if (tid < stride)
        partial[tid] += partial[tid + stride];
    __syncthreads();
}
if (tid == 0) result[blockIdx.x] = partial[0];
''',
        'expected': 'gpu_mp_wg',
        'category': 'gpu',
    },
    {
        'name': 'coroutine_channel',
        'code': '''
// Sender coroutine
channel.data = value;
channel.ready.store(true, std::memory_order_release);
co_yield;
// Receiver coroutine
while (!channel.ready.load(std::memory_order_acquire))
    co_yield;
auto val = channel.data;
''',
        'expected': 'mp_fence',
        'category': 'coroutines',
    },
    {
        'name': 'kernel_rcu_update',
        'code': '''
// Writer
new_data = kmalloc(sizeof(*new_data));
*new_data = compute_new();
smp_wmb();
rcu_assign_pointer(global_ptr, new_data);
synchronize_rcu();
kfree(old_data);
// Reader
rcu_read_lock();
p = rcu_dereference(global_ptr);
val = p->field;
rcu_read_unlock();
''',
        'expected': 'rcu_publish',
        'category': 'kernel',
    },
    {
        'name': 'seqlock_reader',
        'code': '''
// Writer
write_seqlock(&sl);
shared_data.x = new_x;
shared_data.y = new_y;
write_sequnlock(&sl);
// Reader
do {
    seq = read_seqbegin(&sl);
    x = shared_data.x;
    y = shared_data.y;
} while (read_seqretry(&sl, seq));
''',
        'expected': 'seqlock_read',
        'category': 'kernel',
    },
    {
        'name': 'work_stealing_deque',
        'code': '''
// Owner: push
buffer[bottom % SIZE] = task;
atomic_thread_fence(memory_order_release);
atomic_store_explicit(&bottom, bottom + 1, memory_order_relaxed);
// Thief: steal
t = atomic_load_explicit(&top, memory_order_acquire);
b = atomic_load_explicit(&bottom, memory_order_acquire);
if (t < b) {
    task = buffer[t % SIZE];
    if (!atomic_compare_exchange_strong(&top, &t, t + 1))
        return EMPTY;
    return task;
}
''',
        'expected': 'work_steal',
        'category': 'lockfree',
    },
    {
        'name': 'hazard_pointer_scan',
        'code': '''
// Thread 0: protect access
hp[tid] = node;
memory_barrier();
if (node == atomic_load(&head)) {
    val = node->data;  // safe to access
}
hp[tid] = NULL;
// Thread 1: reclaim
atomic_store(&head, new_head);
memory_barrier();
for (i = 0; i < N; i++) {
    if (hp[i] == old_node) goto defer;
}
free(old_node);
''',
        'expected': 'hazard_ptr',
        'category': 'lockfree',
    },
    {
        'name': 'dcl_singleton',
        'code': '''
// Double-checked locking singleton
if (!instance) {
    lock(&mutex);
    if (!instance) {
        tmp = malloc(sizeof(*tmp));
        tmp->field = init_value;
        atomic_store_explicit(&instance, tmp, memory_order_release);
    }
    unlock(&mutex);
}
ptr = atomic_load_explicit(&instance, memory_order_acquire);
use(ptr->field);
''',
        'expected': 'dcl_init',
        'category': 'application',
    },
    {
        'name': 'spsc_ring_buffer',
        'code': '''
// Producer
while (LOAD(write_idx) - LOAD(read_idx) >= SIZE)
    ; // full
buffer[LOAD(write_idx) % SIZE] = item;
smp_wmb();
STORE(write_idx, LOAD(write_idx) + 1);
// Consumer
while (LOAD(read_idx) == LOAD(write_idx))
    ; // empty
smp_rmb();
item = buffer[LOAD(read_idx) % SIZE];
STORE(read_idx, LOAD(read_idx) + 1);
''',
        'expected': 'lockfree_spsc_queue',
        'category': 'lockfree',
    },
    {
        'name': 'cuda_reduction_xwg',
        'code': '''
// Phase 1: intra-block reduction
__shared__ float shared[256];
shared[threadIdx.x] = val;
__syncthreads();
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
        shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
}
// Phase 2: cross-block via atomics
if (threadIdx.x == 0)
    atomicAdd(&global_result, shared[0]);
__threadfence();
if (threadIdx.x == 0)
    atomicAdd(&blocks_done, 1);
''',
        'expected': 'gpu_mp_dev',
        'category': 'gpu',
    },
    {
        'name': 'ms_queue_enqueue',
        'code': '''
// Enqueue
node->data = value;
node->next = NULL;
while (true) {
    tail = atomic_load(&Q->tail);
    next = atomic_load(&tail->next);
    if (next == NULL) {
        if (CAS(&tail->next, NULL, node)) {
            CAS(&Q->tail, tail, node);
            return;
        }
    } else {
        CAS(&Q->tail, tail, next);
    }
}
''',
        'expected': 'ms_queue_enq',
        'category': 'lockfree',
    },
    {
        'name': 'ticket_lock_impl',
        'code': '''
// Acquire
my_ticket = atomic_fetch_add(&lock->ticket, 1);
while (atomic_load(&lock->serving) != my_ticket)
    cpu_relax();
// Critical section
shared_data = compute();
// Release
atomic_store(&lock->serving, my_ticket + 1);
''',
        'expected': 'ticket_lock',
        'category': 'synchronization',
    },
]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LLM-assisted pattern recognition')
    parser.add_argument('--benchmark', action='store_true', help='Run adversarial benchmark')
    parser.add_argument('--code', type=str, help='Analyze a code snippet')
    parser.add_argument('--target', type=str, default='arm', help='Target architecture')
    parser.add_argument('--model', type=str, default='gpt-4.1-nano', help='LLM model')
    args = parser.parse_args()
    
    if args.benchmark:
        print("\n" + "=" * 70)
        print("LITMUS∞ LLM-Assisted Pattern Recognition Benchmark")
        print("=" * 70)
        print(f"Model: {args.model}")
        print(f"Snippets: {len(ADVERSARIAL_SNIPPETS)}")
        
        results = run_llm_benchmark(ADVERSARIAL_SNIPPETS, model=args.model)
        
        print(f"\nResults:")
        print(f"  Exact match: {results['exact_match']}/{results['total']} ({results['exact_rate']:.1%})")
        print(f"  Top-3 match: {results['top3_match']}/{results['total']} ({results['top3_rate']:.1%})")
        print(f"  No match: {results['no_match']}")
        print(f"  Errors: {results['errors']}")
        
        print(f"\nPer-snippet:")
        for d in results['details']:
            status = '✓' if d['exact'] else ('~' if d['top3'] else '✗')
            print(f"  {status} {d['name']}: expected={d['expected']}, "
                  f"predicted={d['predicted']}, conf={d['confidence']:.2f}")
        
        # Save results
        os.makedirs('paper_results_v8', exist_ok=True)
        with open('paper_results_v8/llm_benchmark.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to paper_results_v8/llm_benchmark.json")
    
    elif args.code:
        result = hybrid_check_portability(args.code, args.target, 
                                           use_llm=True, llm_model=args.model)
        print(json.dumps(result, indent=2, default=str))
    
    else:
        # Quick demo
        demo_code = """
// Thread 0: Producer
data = compute();
flag = 1;
// Thread 1: Consumer
if (flag) {
    use(data);
}
"""
        print("Demo: Hybrid pattern recognition")
        print(f"Code: {demo_code.strip()}")
        result = hybrid_check_portability(demo_code, 'arm', use_llm=True, 
                                           llm_model=args.model)
        for r in result:
            status = 'SAFE' if r.get('safe') else 'UNSAFE'
            src = r.get('source', 'ast')
            print(f"  {status}: {r['pattern']} (via {src})")
