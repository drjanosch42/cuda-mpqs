#!/usr/bin/env bash
# Example 2-node cluster launch for cuda-mpqs.
# Coordinator runs on this machine; one worker runs on a remote host.
# Edit COORDINATOR_IP, WORKER_SSH, and the per-node parameters before use.

set -euo pipefail

CUDA_MPQS="./build/tests/cuda-mpqs"
COORDINATOR_IP="<coordinator-ip>"
WORKER_SSH="user@<worker-host>"
LOG_DIR="/tmp"
N_VALUE="--RSA100"         # or --N <decimal>

# Coordinator (runs locally in background)
nohup ${CUDA_MPQS} ${N_VALUE} --verbose \
    --cluster_mode coordinator \
    --expected_workers 1 \
    --listen_port 9100 \
    --sieve_batch_size 8 --cuda_graph_unroll 4 \
    > "${LOG_DIR}/coord.log" 2>&1 &

# Worker (runs on remote host via SSH in background)
ssh "${WORKER_SSH}" "nohup ~/cuda-mpqs/build/tests/cuda-mpqs ${N_VALUE} --verbose \
    --cluster_mode worker \
    --coordinator_host ${COORDINATOR_IP} \
    --coordinator_port 9100 \
    --sieve_batch_size 8 --cuda_graph_unroll 4 \
    > ${LOG_DIR}/worker.log 2>&1 &"

echo "Coordinator log: ${LOG_DIR}/coord.log"
echo "Worker log:      tail -f ${LOG_DIR}/worker.log (on ${WORKER_SSH})"
