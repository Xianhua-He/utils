import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.backends import cudnn

#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10008 --nproc_per_node=4 train_dist.py \

if speed:
    # 优化加速，结果值不能完全复现
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    # 结果值完全复现
    cudnn.benchmark = False
    cudnn.deterministic = True

dist.init_process_group()
samplerdata = DistributedSampler(dataset)

model = DDP(model)

# model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
# python -m torch.distributed.launch --nproc_per_node 8 --use_env train.py \


mp.s