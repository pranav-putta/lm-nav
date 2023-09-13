import torch.distributed as dist
import torch
import socket
import os
import fcntl

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
header = f"{socket.gethostname()}-{local_rank}"
print("starting test on ", header)
try:
    dist.barrier()
    print(f"{header}: NCCL {torch.cuda.nccl.version()} is OK")
except:
    print(f"{header}: NCCL {torch.cuda.nccl.version()} is broken")
    raise
