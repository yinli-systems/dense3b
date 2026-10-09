# Shared settings for the Day-1 jobs. Edit the three lines marked EDIT for your site.
export PYTHON=${PYTHON:-python}                        # EDIT: interpreter with torch 2.13.0+cu130
export RESULTS=${RESULTS:-$PWD/results/$(date +%Y%m%d)-${SLURM_JOB_ID:-local}}
export LOCAL_SCRATCH=${LOCAL_SCRATCH:-/tmp/$USER/moe7b}   # EDIT: node-local NVMe
mkdir -p "$RESULTS" "$LOCAL_SCRATCH"

# NCCL: no P2P on these boxes; log which transport each channel uses (verify SHM intra-node, NET/IB inter-node)
export NCCL_P2P_DISABLE=1
# compute nodes carry a USB/BMC NIC with link-local [LINK_LOCAL_ADDRESS] on every node: keep NCCL/gloo on the data network
export NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
export NCCL_DEBUG_FILE="$RESULTS/nccl.%h.%p.log"
# EDIT if needed: RoCE GID / interface, and spreading flows over both bond legs
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_QPS_PER_CONNECTION=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

head_node() { scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1; }
# run_torchrun <module> [args...]: one torchrun per node, c10d rendezvous on the first node
run_torchrun() {
  srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "cd $PWD && $PYTHON -m torch.distributed.run \
    --nnodes $SLURM_NNODES --nproc-per-node 8 --rdzv-backend c10d --rdzv-endpoint $(head_node):29511 \
    --rdzv-id $SLURM_JOB_ID-$RANDOM -m $*"
}
diagnostics() {
  srun --ntasks-per-node=1 bash -c '
    out='"$RESULTS"'/diag.$(hostname).txt
    { nvidia-smi; nvidia-smi topo -m; numactl -H; lscpu | head -25; cat /proc/net/bonding/bond0;
      ibv_devinfo 2>/dev/null | head -40; free -g; df -h '"$LOCAL_SCRATCH"'; } > $out 2>&1'
}
