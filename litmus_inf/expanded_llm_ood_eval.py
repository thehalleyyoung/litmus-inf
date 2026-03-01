#!/usr/bin/env python3
"""
Expanded LLM OOD Evaluation with Full Statistical Rigor.

Addresses consensus weakness #1: "LLM-assisted OOD evaluation (93.3%)
is statistically incomplete: unreported sample size, no confidence intervals."

Expands from n=15 to n=50 adversarial OOD snippets, reports Wilson CIs,
per-category breakdown, and model comparison.
"""

import json
import math
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(__file__))


def wilson_ci(successes: int, total: int, z: float = 1.96):
    """Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return 0, 0
    p_hat = successes / total
    denom = 1 + z ** 2 / total
    center = (p_hat + z ** 2 / (2 * total)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / total + z ** 2 / (4 * total ** 2)) / denom
    return max(0, center - margin), min(1, center + margin)


# Extended adversarial OOD snippets: 50 total across 10 categories.
# Each snippet uses real-world idioms deliberately different from the
# canonical litmus test templates.
EXTENDED_OOD_SNIPPETS = [
    # === Cryptography (5) ===
    {
        'name': 'crypto_aes_ctr',
        'code': '''
counter = atomic_load(&shared_ctr);
atomic_store(&shared_ctr, counter + 1);
encrypt_block(key, counter, &output);
// Reader
if (atomic_load(&output_ready)) { memcpy(buf, output, 16); }
''',
        'expected': 'mp',
        'category': 'cryptography',
    },
    {
        'name': 'crypto_chacha_nonce',
        'code': '''
// Thread 0: generate nonce
nonce = atomic_fetch_add(&nonce_ctr, 1, memory_order_relaxed);
compute_keystream(key, nonce, &stream);
atomic_store_explicit(&stream_ready, 1, memory_order_release);
// Thread 1: consume keystream
while (!atomic_load_explicit(&stream_ready, memory_order_acquire)) ;
xor_encrypt(plaintext, stream, ciphertext);
''',
        'expected': 'mp_fence',
        'category': 'cryptography',
    },
    {
        'name': 'crypto_hmac_key_update',
        'code': '''
// Key rotation thread
new_key = derive_key(master, epoch);
memcpy(key_slot, new_key, 32);
smp_wmb();
WRITE_ONCE(key_epoch, epoch);
// HMAC compute thread
e = READ_ONCE(key_epoch);
smp_rmb();
hmac_compute(key_slot, data, &tag);
''',
        'expected': 'mp_fence',
        'category': 'cryptography',
    },
    {
        'name': 'crypto_rng_pool',
        'code': '''
// Entropy collector
pool[idx] ^= hardware_random();
idx = (idx + 1) % POOL_SIZE;
atomic_store(&entropy_count, atomic_load(&entropy_count) + 8);
// Consumer
if (atomic_load(&entropy_count) >= needed) {
    memcpy(out, pool, needed/8);
}
''',
        'expected': 'mp',
        'category': 'cryptography',
    },
    {
        'name': 'crypto_tls_session',
        'code': '''
// Handshake thread
session->cipher = negotiate_cipher(hello);
session->key = derive_keys(premaster);
atomic_store_explicit(&session->state, ESTABLISHED, memory_order_release);
// Data thread
if (atomic_load_explicit(&session->state, memory_order_acquire) == ESTABLISHED) {
    encrypt(session->key, data, &ct);
}
''',
        'expected': 'mp_fence',
        'category': 'cryptography',
    },
    # === Embedded Systems (5) ===
    {
        'name': 'embedded_dma',
        'code': '''
dma_buf[0] = data; dma_buf[1] = len;
mmio_write(DMA_CTRL, START);
// IRQ handler
status = mmio_read(DMA_STATUS);
if (status & DONE) result = dma_buf[0];
''',
        'expected': 'mp',
        'category': 'embedded',
    },
    {
        'name': 'embedded_uart_flag',
        'code': '''
// ISR: UART receive
rx_buf[rx_head] = UART_DR;
rx_head = (rx_head + 1) % BUF_SIZE;
WRITE_ONCE(rx_count, rx_count + 1);
// Main loop
while (READ_ONCE(rx_count) == 0) wfi();
byte = rx_buf[rx_tail];
rx_tail = (rx_tail + 1) % BUF_SIZE;
''',
        'expected': 'mp',
        'category': 'embedded',
    },
    {
        'name': 'embedded_watchdog',
        'code': '''
// Main task: pet watchdog
atomic_store(&heartbeat, get_tick());
// Watchdog task
last = atomic_load(&heartbeat);
if (get_tick() - last > TIMEOUT) system_reset();
''',
        'expected': 'corr',
        'category': 'embedded',
    },
    {
        'name': 'embedded_sensor_pub',
        'code': '''
// Sensor ISR
sensor_data.temp = read_adc(TEMP_CH);
sensor_data.humidity = read_adc(HUM_CH);
__DMB();
sensor_data.seq++;
// Display task
s1 = sensor_data.seq;
__DMB();
t = sensor_data.temp; h = sensor_data.humidity;
__DMB();
s2 = sensor_data.seq;
if (s1 == s2 && !(s1 & 1)) display(t, h);
''',
        'expected': 'seqlock_read',
        'category': 'embedded',
    },
    {
        'name': 'embedded_mailbox',
        'code': '''
// Core 0: send message
mailbox->data = msg;
dsb();
mailbox->flag = 1;
sev();
// Core 1: receive
while (!mailbox->flag) wfe();
dmb();
result = mailbox->data;
mailbox->flag = 0;
''',
        'expected': 'mp_fence',
        'category': 'embedded',
    },
    # === Networking (5) ===
    {
        'name': 'network_packet_ring',
        'code': '''
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
        'name': 'network_conn_table',
        'code': '''
// Connection setup
conn = alloc_conn();
conn->state = ESTABLISHED;
conn->socket = sk;
smp_wmb();
WRITE_ONCE(conn_table[hash], conn);
// Lookup
c = READ_ONCE(conn_table[hash]);
smp_rmb();
if (c && c->state == ESTABLISHED) send(c->socket, data);
''',
        'expected': 'mp_fence',
        'category': 'networking',
    },
    {
        'name': 'network_refcount',
        'code': '''
// Thread 0: increment reference
old = atomic_fetch_add(&pkt->refcnt, 1, memory_order_relaxed);
process(pkt);
// Thread 1: decrement + free
if (atomic_fetch_sub(&pkt->refcnt, 1, memory_order_acq_rel) == 1) {
    free_packet(pkt);
}
''',
        'expected': 'amoswap',
        'category': 'networking',
    },
    {
        'name': 'network_flow_state',
        'code': '''
// Fast path: read flow entry
flow = READ_ONCE(flow_cache[hash]);
if (flow && flow->key == key) {
    return flow->action;
}
// Slow path: insert
new_flow = alloc_flow(key, action);
smp_wmb();
WRITE_ONCE(flow_cache[hash], new_flow);
''',
        'expected': 'mp_fence',
        'category': 'networking',
    },
    {
        'name': 'network_zero_copy',
        'code': '''
// TX thread: prepare buffer
memcpy(shinfo->frags, iov, n * sizeof(skb_frag_t));
smp_wmb();
atomic_set(&shinfo->dataref, 1);
// Completion thread
if (atomic_dec_and_test(&shinfo->dataref)) {
    recycle_buffer(shinfo);
}
''',
        'expected': 'mp_fence',
        'category': 'networking',
    },
    # === Lock-free Data Structures (8) ===
    {
        'name': 'lockfree_treiber_stack',
        'code': '''
node->data = value;
do {
    old_head = atomic_load_explicit(&head, memory_order_relaxed);
    node->next = old_head;
} while (!atomic_compare_exchange_weak(&head, &old_head, node));
// Pop
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
        'name': 'lockfree_ms_queue',
        'code': '''
node->data = value; node->next = NULL;
while (true) {
    tail = atomic_load(&Q->tail);
    next = atomic_load(&tail->next);
    if (next == NULL) {
        if (CAS(&tail->next, NULL, node)) {
            CAS(&Q->tail, tail, node); return;
        }
    } else { CAS(&Q->tail, tail, next); }
}
''',
        'expected': 'ms_queue_enq',
        'category': 'lockfree',
    },
    {
        'name': 'lockfree_spsc_bounded',
        'code': '''
// Producer
while (LOAD(write_idx) - LOAD(read_idx) >= SIZE) ;
buffer[LOAD(write_idx) % SIZE] = item;
smp_wmb();
STORE(write_idx, LOAD(write_idx) + 1);
// Consumer
while (LOAD(read_idx) == LOAD(write_idx)) ;
smp_rmb();
item = buffer[LOAD(read_idx) % SIZE];
STORE(read_idx, LOAD(read_idx) + 1);
''',
        'expected': 'lockfree_spsc_queue',
        'category': 'lockfree',
    },
    {
        'name': 'lockfree_hazard_ptr',
        'code': '''
hp[tid] = node; memory_barrier();
if (node == atomic_load(&head)) { val = node->data; }
hp[tid] = NULL;
// Reclaimer
atomic_store(&head, new_head);
memory_barrier();
for (i = 0; i < N; i++) { if (hp[i] == old) goto defer; }
free(old);
''',
        'expected': 'hazard_ptr',
        'category': 'lockfree',
    },
    {
        'name': 'lockfree_work_steal',
        'code': '''
buffer[bottom % SIZE] = task;
atomic_thread_fence(memory_order_release);
atomic_store_explicit(&bottom, bottom + 1, memory_order_relaxed);
// Thief
t = atomic_load_explicit(&top, memory_order_acquire);
b = atomic_load_explicit(&bottom, memory_order_acquire);
if (t < b) {
    task = buffer[t % SIZE];
    if (!CAS(&top, &t, t + 1)) return EMPTY;
    return task;
}
''',
        'expected': 'work_steal',
        'category': 'lockfree',
    },
    {
        'name': 'lockfree_epoch_reclaim',
        'code': '''
// Enter epoch
e = atomic_load_explicit(&global_epoch, memory_order_relaxed);
atomic_store_explicit(&local_epoch[tid], e, memory_order_release);
atomic_thread_fence(memory_order_seq_cst);
// Use protected data
val = ptr->data;
// Exit epoch
atomic_store_explicit(&local_epoch[tid], INACTIVE, memory_order_release);
''',
        'expected': 'mp_fence',
        'category': 'lockfree',
    },
    {
        'name': 'lockfree_mpmc_queue',
        'code': '''
// Enqueue
pos = atomic_fetch_add(&enq_pos, 1, memory_order_relaxed);
cell = &buffer[pos % SIZE];
while (atomic_load_explicit(&cell->seq, memory_order_acquire) != pos) ;
cell->data = value;
atomic_store_explicit(&cell->seq, pos + 1, memory_order_release);
// Dequeue
pos = atomic_fetch_add(&deq_pos, 1, memory_order_relaxed);
cell = &buffer[pos % SIZE];
while (atomic_load_explicit(&cell->seq, memory_order_acquire) != pos + 1) ;
value = cell->data;
atomic_store_explicit(&cell->seq, pos + SIZE, memory_order_release);
''',
        'expected': 'lockfree_spsc_queue',
        'category': 'lockfree',
    },
    {
        'name': 'lockfree_harris_list',
        'code': '''
// Insert
while (true) {
    prev = find(head, key, &curr);
    if (curr && curr->key == key) return false;
    node->next = curr;
    if (CAS(&prev->next, curr, node)) return true;
}
// Delete: logical then physical
succ = atomic_load(&curr->next);
marked = succ | MARK_BIT;
if (CAS(&curr->next, succ, marked))
    CAS(&prev->next, curr, succ);
''',
        'expected': 'lockfree_stack_push',
        'category': 'lockfree',
    },
    # === GPU (5) ===
    {
        'name': 'gpu_reduction',
        'code': '''
__shared__ float shared[256];
shared[threadIdx.x] = val;
__syncthreads();
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
}
if (threadIdx.x == 0) result[blockIdx.x] = shared[0];
''',
        'expected': 'gpu_mp_wg',
        'category': 'gpu',
    },
    {
        'name': 'gpu_cross_block_reduce',
        'code': '''
if (threadIdx.x == 0) atomicAdd(&global_result, shared[0]);
__threadfence();
if (threadIdx.x == 0) atomicAdd(&blocks_done, 1);
// Last block finalizes
if (atomicLoad(&blocks_done) == gridDim.x) {
    final_result = global_result;
}
''',
        'expected': 'gpu_mp_dev',
        'category': 'gpu',
    },
    {
        'name': 'gpu_producer_consumer',
        'code': '''
// Producer block
if (blockIdx.x == 0) {
    shared_buf[threadIdx.x] = compute(threadIdx.x);
    __threadfence_system();
    if (threadIdx.x == 0) atomicExch(&flag, 1);
}
// Consumer block
if (blockIdx.x == 1) {
    if (threadIdx.x == 0) while (atomicLoad(&flag) == 0) ;
    __threadfence_system();
    val = shared_buf[threadIdx.x];
}
''',
        'expected': 'gpu_mp_dev',
        'category': 'gpu',
    },
    {
        'name': 'gpu_warp_scan',
        'code': '''
// Inclusive scan within warp
val = input[tid];
for (int offset = 1; offset < 32; offset <<= 1) {
    float n = __shfl_up_sync(0xffffffff, val, offset);
    if (lane_id >= offset) val += n;
}
output[tid] = val;
''',
        'expected': 'gpu_mp_wg',
        'category': 'gpu',
    },
    {
        'name': 'gpu_persistent_kernel',
        'code': '''
while (true) {
    if (threadIdx.x == 0) {
        task_id = atomicAdd(&task_counter, 1);
    }
    task_id = __shfl_sync(0xffffffff, task_id, 0);
    if (task_id >= num_tasks) break;
    result[task_id] = process(input[task_id]);
    __threadfence();
    if (threadIdx.x == 0) atomicAdd(&completed, 1);
}
''',
        'expected': 'gpu_mp_dev',
        'category': 'gpu',
    },
    # === Kernel (5) ===
    {
        'name': 'kernel_rcu_update',
        'code': '''
new_data = kmalloc(sizeof(*new_data));
*new_data = compute_new();
smp_wmb();
rcu_assign_pointer(global_ptr, new_data);
synchronize_rcu(); kfree(old_data);
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
        'name': 'kernel_seqlock',
        'code': '''
write_seqlock(&sl);
shared_data.x = new_x; shared_data.y = new_y;
write_sequnlock(&sl);
// Reader
do {
    seq = read_seqbegin(&sl);
    x = shared_data.x; y = shared_data.y;
} while (read_seqretry(&sl, seq));
''',
        'expected': 'seqlock_read',
        'category': 'kernel',
    },
    {
        'name': 'kernel_percpu_counter',
        'code': '''
// Increment on this CPU
preempt_disable();
pcp = this_cpu_ptr(&counter);
pcp->count++;
if (abs(pcp->count) > pcp->batch) {
    atomic_long_add(pcp->count, &fbc->count);
    pcp->count = 0;
}
preempt_enable();
// Read global sum
sum = atomic_long_read(&fbc->count);
for_each_online_cpu(cpu) sum += per_cpu_ptr(&counter, cpu)->count;
''',
        'expected': 'mp',
        'category': 'kernel',
    },
    {
        'name': 'kernel_completion',
        'code': '''
// Waiter
wait_for_completion(&done);
result = shared_result;
// Completer
shared_result = compute();
smp_wmb();
complete(&done);
''',
        'expected': 'mp_fence',
        'category': 'kernel',
    },
    {
        'name': 'kernel_spinlock_release',
        'code': '''
spin_lock(&lock);
shared_data = new_value;
spin_unlock(&lock);
// Another CPU
spin_lock(&lock);
val = shared_data;
spin_unlock(&lock);
''',
        'expected': 'sb_fence',
        'category': 'kernel',
    },
    # === Synchronization (5) ===
    {
        'name': 'sync_dcl_singleton',
        'code': '''
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
        'category': 'synchronization',
    },
    {
        'name': 'sync_ticket_lock',
        'code': '''
my_ticket = atomic_fetch_add(&lock->ticket, 1);
while (atomic_load(&lock->serving) != my_ticket) cpu_relax();
shared_data = compute();
atomic_store(&lock->serving, my_ticket + 1);
''',
        'expected': 'ticket_lock',
        'category': 'synchronization',
    },
    {
        'name': 'sync_condition_var',
        'code': '''
// Producer
lock(&mtx);
queue_push(&q, item);
atomic_store_explicit(&has_data, 1, memory_order_release);
unlock(&mtx);
signal(&cond);
// Consumer
lock(&mtx);
while (!atomic_load_explicit(&has_data, memory_order_acquire))
    wait(&cond, &mtx);
item = queue_pop(&q);
unlock(&mtx);
''',
        'expected': 'mp_fence',
        'category': 'synchronization',
    },
    {
        'name': 'sync_barrier',
        'code': '''
// Phase 1: local computation
local_result[tid] = compute();
atomic_fetch_add(&barrier_count, 1, memory_order_acq_rel);
// Wait for all threads
while (atomic_load_explicit(&barrier_count, memory_order_acquire) < N_THREADS) ;
// Phase 2: use all results
total = 0;
for (int i = 0; i < N_THREADS; i++) total += local_result[i];
''',
        'expected': 'mp_fence',
        'category': 'synchronization',
    },
    {
        'name': 'sync_rwlock',
        'code': '''
// Reader
while (true) {
    old = atomic_load(&lock);
    if (old < 0) continue;
    if (CAS(&lock, old, old + 1)) break;
}
val = shared_data;
atomic_fetch_sub(&lock, 1);
// Writer
while (!CAS(&lock, 0, -1)) ;
shared_data = new_val;
atomic_store(&lock, 0);
''',
        'expected': 'dekker',
        'category': 'synchronization',
    },
    # === Coroutines / Async (3) ===
    {
        'name': 'async_channel',
        'code': '''
channel.data = value;
channel.ready.store(true, std::memory_order_release);
co_yield;
// Receiver
while (!channel.ready.load(std::memory_order_acquire)) co_yield;
auto val = channel.data;
''',
        'expected': 'mp_fence',
        'category': 'async',
    },
    {
        'name': 'async_future_promise',
        'code': '''
// Promise: set value
promise.value = result;
promise.state.store(READY, std::memory_order_release);
// Future: get value
while (promise.state.load(std::memory_order_acquire) != READY) ;
return promise.value;
''',
        'expected': 'mp_fence',
        'category': 'async',
    },
    {
        'name': 'async_task_queue',
        'code': '''
// Submit
task->fn = work_fn;
task->arg = arg;
smp_wmb();
WRITE_ONCE(task_slot[idx], task);
wake_worker();
// Worker
t = READ_ONCE(task_slot[idx]);
if (t) {
    smp_rmb();
    t->fn(t->arg);
    WRITE_ONCE(task_slot[idx], NULL);
}
''',
        'expected': 'mp_fence',
        'category': 'async',
    },
    # === Database / Application (4) ===
    {
        'name': 'db_skiplist_insert',
        'code': '''
// Insert
node->key = key; node->val = val;
for (int i = height-1; i >= 0; i--) {
    node->next[i] = succs[i];
}
atomic_thread_fence(memory_order_release);
for (int i = 0; i <= height; i++) {
    preds[i]->next[i] = node;
}
// Reader
curr = atomic_load_explicit(&head->next[level], memory_order_acquire);
while (curr && curr->key < target) {
    curr = atomic_load_explicit(&curr->next[level], memory_order_acquire);
}
''',
        'expected': 'mp_fence',
        'category': 'database',
    },
    {
        'name': 'db_write_batch',
        'code': '''
// Batch writer
for (int i = 0; i < batch_size; i++) {
    memtable_put(mt, ops[i].key, ops[i].val);
}
smp_wmb();
atomic_store(&wal->seq, new_seq);
// Reader
seq = atomic_load(&wal->seq);
smp_rmb();
val = memtable_get(mt, key);
''',
        'expected': 'mp_fence',
        'category': 'database',
    },
    {
        'name': 'db_mvcc_snapshot',
        'code': '''
// Writer
new_ver = alloc_version();
new_ver->data = new_data;
new_ver->txn_id = my_txn;
atomic_store_explicit(&row->latest, new_ver, memory_order_release);
// Reader (snapshot)
snap_txn = atomic_load_explicit(&global_txn, memory_order_acquire);
ver = atomic_load_explicit(&row->latest, memory_order_acquire);
while (ver->txn_id > snap_txn) ver = ver->prev;
return ver->data;
''',
        'expected': 'mp_fence',
        'category': 'database',
    },
    {
        'name': 'db_log_flush',
        'code': '''
// Logger
memcpy(&log_buf[log_pos], record, rec_len);
log_pos += rec_len;
smp_wmb();
WRITE_ONCE(commit_pos, log_pos);
// Checkpoint
pos = READ_ONCE(commit_pos);
smp_rmb();
flush_to_disk(log_buf, pos);
''',
        'expected': 'mp_fence',
        'category': 'database',
    },
    # === SIMD / Architecture-specific (5) ===
    {
        'name': 'simd_gather_scatter',
        'code': '''
// Thread 0: scatter results
for (int i = 0; i < 8; i++)
    atomic_store_explicit(&output[indices[i]], partial[i], memory_order_relaxed);
atomic_thread_fence(memory_order_release);
atomic_store_explicit(&done_flag, 1, memory_order_relaxed);
// Thread 1: gather
while (!atomic_load_explicit(&done_flag, memory_order_acquire)) ;
for (int i = 0; i < 8; i++)
    results[i] = atomic_load_explicit(&output[indices[i]], memory_order_relaxed);
''',
        'expected': 'mp_fence',
        'category': 'simd',
    },
    {
        'name': 'simd_vector_publish',
        'code': '''
// Writer: compute vector then publish
float4 vec;
vec.x = compute_x(); vec.y = compute_y();
vec.z = compute_z(); vec.w = compute_w();
smp_wmb();
WRITE_ONCE(vec_ready, 1);
// Reader
if (READ_ONCE(vec_ready)) {
    smp_rmb();
    use_vector(shared_vec);
}
''',
        'expected': 'mp_fence',
        'category': 'simd',
    },
    {
        'name': 'riscv_lr_sc',
        'code': '''
// RISC-V atomic increment
retry:
    lr.w a0, (a1)
    addi a0, a0, 1
    sc.w a2, a0, (a1)
    bnez a2, retry
// Reader on another hart
    lw a3, 0(a1)
    fence r,r
    lw a4, 4(a1)  // data dependent on atomic
''',
        'expected': 'amoswap',
        'category': 'simd',
    },
    {
        'name': 'arm_exclusive_monitor',
        'code': '''
// ARM LDREX/STREX spinlock
loop:
    ldrex r0, [r1]
    cmp r0, #0
    bne loop
    strex r2, r3, [r1]
    cmp r2, #0
    bne loop
    dmb ish
    // critical section
    ldr r4, [r5]
    dmb ish
    str r0, [r1]
''',
        'expected': 'sb_fence',
        'category': 'simd',
    },
    {
        'name': 'x86_cmpxchg_publish',
        'code': '''
// x86 CMPXCHG for publication
mov rax, [shared_ptr]
test rax, rax
jnz .done
; Allocate and init
call alloc
mov [rax + field], init_val
lock cmpxchg [shared_ptr], rax
.done:
; Reader
mov rdi, [shared_ptr]
test rdi, rdi
jz .retry
mov rsi, [rdi + field]
''',
        'expected': 'dcl_init',
        'category': 'simd',
    },
]

assert len(EXTENDED_OOD_SNIPPETS) == 50, f"Expected 50 snippets, got {len(EXTENDED_OOD_SNIPPETS)}"


def run_extended_evaluation(model: str = "gpt-4.1-nano", 
                             use_mock: bool = False) -> Dict:
    """Run extended OOD evaluation with full statistics."""
    from llm_pattern_recognizer import llm_recognize_patterns

    results = {
        'model': model,
        'n_total': len(EXTENDED_OOD_SNIPPETS),
        'exact_match': 0,
        'top3_match': 0,
        'no_match': 0,
        'errors': 0,
        'per_category': {},
        'details': [],
        'timings': [],
    }

    for i, snip in enumerate(EXTENDED_OOD_SNIPPETS):
        cat = snip['category']
        if cat not in results['per_category']:
            results['per_category'][cat] = {'total': 0, 'exact': 0, 'top3': 0}
        results['per_category'][cat]['total'] += 1

        start = time.time()
        try:
            llm_result = llm_recognize_patterns(snip['code'], model=model)
            elapsed = time.time() - start
            patterns = llm_result.get('patterns', [])
            error = False
        except Exception as e:
            elapsed = time.time() - start
            patterns = []
            error = True
            results['errors'] += 1

        expected = snip['expected']
        exact = len(patterns) > 0 and patterns[0] == expected
        top3 = expected in patterns[:3]

        if exact:
            results['exact_match'] += 1
            results['per_category'][cat]['exact'] += 1
        if top3:
            results['top3_match'] += 1
            results['per_category'][cat]['top3'] += 1
        if not patterns:
            results['no_match'] += 1

        results['timings'].append(elapsed)
        results['details'].append({
            'name': snip['name'],
            'category': cat,
            'expected': expected,
            'predicted': patterns[:3],
            'exact': exact,
            'top3': top3,
            'time_s': round(elapsed, 3),
            'error': error,
        })

        if (i + 1) % 10 == 0 or i == len(EXTENDED_OOD_SNIPPETS) - 1:
            pct = results['exact_match'] / (i + 1) * 100
            print(f"  [{i+1}/50] exact={pct:.1f}%")

    # Compute statistics
    n = results['n_total']
    k_exact = results['exact_match']
    k_top3 = results['top3_match']

    results['exact_rate'] = k_exact / n
    results['top3_rate'] = k_top3 / n
    results['exact_wilson_ci'] = list(wilson_ci(k_exact, n))
    results['top3_wilson_ci'] = list(wilson_ci(k_top3, n))

    # Per-category Wilson CIs
    for cat, stats in results['per_category'].items():
        ct = stats['total']
        stats['exact_rate'] = stats['exact'] / ct if ct else 0
        stats['top3_rate'] = stats['top3'] / ct if ct else 0
        stats['exact_wilson_ci'] = list(wilson_ci(stats['exact'], ct))
        stats['top3_wilson_ci'] = list(wilson_ci(stats['top3'], ct))

    # Timing statistics
    ts = results['timings']
    results['timing_stats'] = {
        'mean_s': sum(ts) / len(ts),
        'median_s': sorted(ts)[len(ts) // 2],
        'max_s': max(ts),
        'total_s': sum(ts),
    }

    return results


def run_ast_only_baseline() -> Dict:
    """Run AST-only baseline on the extended OOD snippets."""
    from ast_analyzer import ast_analyze_code

    results = {
        'n_total': len(EXTENDED_OOD_SNIPPETS),
        'exact_match': 0,
        'top3_match': 0,
        'per_category': {},
        'details': [],
    }

    for snip in EXTENDED_OOD_SNIPPETS:
        cat = snip['category']
        if cat not in results['per_category']:
            results['per_category'][cat] = {'total': 0, 'exact': 0, 'top3': 0}
        results['per_category'][cat]['total'] += 1

        try:
            analysis = ast_analyze_code(snip['code'], language='c')
            patterns = [m.pattern_name for m in analysis.patterns_found[:3]]
        except Exception:
            patterns = []

        expected = snip['expected']
        exact = len(patterns) > 0 and patterns[0] == expected
        top3 = expected in patterns[:3]

        if exact:
            results['exact_match'] += 1
            results['per_category'][cat]['exact'] += 1
        if top3:
            results['top3_match'] += 1
            results['per_category'][cat]['top3'] += 1

        results['details'].append({
            'name': snip['name'],
            'category': cat,
            'expected': expected,
            'predicted': patterns[:3],
            'exact': exact,
            'top3': top3,
        })

    n = results['n_total']
    results['exact_rate'] = results['exact_match'] / n
    results['top3_rate'] = results['top3_match'] / n
    results['exact_wilson_ci'] = list(wilson_ci(results['exact_match'], n))
    results['top3_wilson_ci'] = list(wilson_ci(results['top3_match'], n))

    for cat, stats in results['per_category'].items():
        ct = stats['total']
        stats['exact_rate'] = stats['exact'] / ct if ct else 0
        stats['exact_wilson_ci'] = list(wilson_ci(stats['exact'], ct))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-4.1-nano')
    parser.add_argument('--ast-only', action='store_true', help='Run AST-only baseline')
    args = parser.parse_args()

    print("=" * 70)
    print("LITMUS∞ Extended LLM OOD Evaluation (n=50)")
    print("=" * 70)

    # Always run AST baseline
    print("\n--- AST-only baseline ---")
    ast_results = run_ast_only_baseline()
    print(f"AST-only: {ast_results['exact_match']}/{ast_results['n_total']} "
          f"({ast_results['exact_rate']:.1%})")
    print(f"  Wilson CI: [{ast_results['exact_wilson_ci'][0]:.3f}, "
          f"{ast_results['exact_wilson_ci'][1]:.3f}]")

    if not args.ast_only:
        print(f"\n--- LLM evaluation ({args.model}) ---")
        llm_results = run_extended_evaluation(model=args.model)
        n = llm_results['n_total']
        print(f"\nResults (n={n}):")
        print(f"  Exact: {llm_results['exact_match']}/{n} ({llm_results['exact_rate']:.1%})")
        print(f"    Wilson 95% CI: [{llm_results['exact_wilson_ci'][0]:.3f}, "
              f"{llm_results['exact_wilson_ci'][1]:.3f}]")
        print(f"  Top-3: {llm_results['top3_match']}/{n} ({llm_results['top3_rate']:.1%})")
        print(f"    Wilson 95% CI: [{llm_results['top3_wilson_ci'][0]:.3f}, "
              f"{llm_results['top3_wilson_ci'][1]:.3f}]")

        print("\nPer-category breakdown:")
        for cat, stats in sorted(llm_results['per_category'].items()):
            ci = stats['exact_wilson_ci']
            print(f"  {cat:20s}: {stats['exact']}/{stats['total']} "
                  f"({stats['exact_rate']:.0%}) CI=[{ci[0]:.2f},{ci[1]:.2f}]")

        # Save results
        os.makedirs('paper_results_v10', exist_ok=True)
        combined = {
            'ast_baseline': ast_results,
            'llm_evaluation': llm_results,
            'methodology': {
                'n_snippets': 50,
                'categories': 10,
                'snippet_construction': 'adversarial out-of-distribution concurrent code',
                'adversarial_criteria': [
                    'Real-world API idioms (not canonical litmus test form)',
                    'Complex control flow (loops, conditionals, function calls)',
                    'Domain-specific macros and abstractions',
                    'Multi-statement patterns spanning 5+ operations',
                ],
                'statistical_tests': ['Wilson score 95% CI', 'per-category breakdown'],
            },
        }
        with open('paper_results_v10/extended_llm_ood_eval.json', 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\nResults saved to paper_results_v10/extended_llm_ood_eval.json")
    else:
        os.makedirs('paper_results_v10', exist_ok=True)
        with open('paper_results_v10/ast_ood_baseline.json', 'w') as f:
            json.dump(ast_results, f, indent=2)
