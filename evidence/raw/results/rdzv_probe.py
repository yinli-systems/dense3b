import os, socket, torch.distributed as dist
dist.init_process_group("gloo"); print(socket.gethostname(), "RANK", dist.get_rank(), "WORLD", dist.get_world_size(), flush=True)
