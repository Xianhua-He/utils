import argparse
import logging
import os
import time

import torch
import yaml
import random
import numpy as np
import heapq
from torch.backends import cudnn
import apex
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel.DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())


def load_cfg_from_cfg_file(file):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg

def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', type=str, default='config/config.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    return cfg

def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def get_cuda_devices():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    memory_gpu = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    clen = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    devices = heapq.nlargest(clen, range(len(memory_gpu)), memory_gpu.take)
    test_gpu = devices
    os.environ['CUDA_VISIBLE_DEVICES']=str(devices)[1:-1]
    os.system('rm tmp')
    return devices

# 写多进程分布式：args.multiprocessing_distributed
def main():
    args = get_parse()
    if args.train_gpu:
        # train_gpu: [1, 2]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    else:
        args.train_gpu = get_cuda_devices()
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        # 果我们的网络模型一直变的话，那肯定是不能设置 cudnn.benchmark=True
        cudnn.benchmark = False
        # 确定性的
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        # world size指进程总数
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # args.ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        # 总进程数 = 每个节点的GPU数 * 节点数目
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


# 区分是多GPU
# 单GPU
# 使用CPU
def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    # 如果是多GPU训练，参数gpu（0， 1， 2）表示当前使用的gpu编号
    if args.sync_bn:
        if args.multiprocessing_distributed:
            BatchNorm = apex.parallel.SyncBatchNorm
        else:
            from sync_bn.modules import BatchNorm2d
            BatchNorm = BatchNorm2d
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rand = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # rank为node编号
            # local rank 当前进程编号，等级，0为主进程、主GPU
            # local rank = node_num * 每个node的gpu + 当前gpu编号
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    """===== Model and Criterion Definition ====="""
    # define model
    model = ResNet()
    # define loss
    criterion = nn.CrossEntropyLoss()

    """===== Model Initialization ====="""
    if args.pretrain:
        import torchvision.models as models
        if args.backbone == 'resnet50':
            resnet50 = models.resnet50(pretrained=True)

    """===== Optimizer Definition ====="""
    if args.pretrain and args.diff_lr:
        params_list = []
        for module in modules_ori:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr))
        for module in modules_new:
            params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))
    else:
        params_list = model.parameters()
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_list, lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(params_list, lr=args.base_lr, betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("===> creating model ...")
        logger.info("Class: {}".format(args.classes))
        logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        model.cuda(gpu)
        criterion.cuda(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        # workers多进程加载数据
        args.workers = int(args.workers / ngpus_per_node)
        if args.use_apex:
            model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.opt_level,
                                                   keep_batchnorm_fp32=args.keep_batchnorm_fp32, loss_scale=args.loss_scale)
            model = apex.parallel.DistributedDataParallel(model)
        else:
            model = DDP(model, device_ids=[gpu])
    else:
        model = nn.DataParallel(model.cuda())
    """===== Load Initialization Weight ====="""
    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("===> loading weight {}".format(args.weight))
            checkpoint = torch.load(args.weight)
            state_dict = model.state_dict()
            state_dict.update(checkpoint['state_dict'])
            model.load_state_dict(state_dict)
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))

    """===== Resume Checkpoint ====="""
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            old_epo = checkpoint['epoch']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))
    """===== Training Set Definition ====="""
    train_data = MyDataset()
    if args.distributed:
        train_sampler = DistributedSampler(train_data)
    else:
        train_sampler = None
    # workers多进程加载数据
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers,
                                               pin_memory=True, sampler=train_sampler, drop_last=True)
    """===== Validation Set Definition ====="""
    val_data = MyDatasset()
    """===== Scheduler Definition ====="""
    if args.use_scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader) - args.warmup_iter)
    else:
        scheduler = None
    """===== Training Epoches ====="""
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train(train_loader, model, optimizer, epoch)   # TODO

        if main_process():
            writer.add_scalar('train_loss', train_loss, epoch_log)
        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/train_epoch_' + str(epoch_log) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            if epoch_log / args.save_freq > 100:
                deletename = args.save_path + '/train_epoch_' + str(epoch_log - args.save_freq * 2) + '.pth'
                os.remove(deletename)
        if args.evaluate:
            if args.evaluate == 'loss':
                loss_val, _ = validate(val_loader, model, criterion)
                if main_process():
                    writer.add_scalar('loss_val', loss_val, epoch_log)
                    is_best = loss_val < best_loss_val
                    best_loss_val = min(loss_val, best_loss_val)
                    if is_best:
                        best_loss_epoch = epoch_log
                        logger.info('update best val loss in epoch {} : {:.4f}'.format(best_loss_epoch, best_loss_val))
                        filename = args.save_path + '/best_checkpoint.pth'
                        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
            elif args.evaluate == 'miou':
                _, miou_val = validate(val_loader, model, criterion)
                if main_process():
                    writer.add_scalar('miou', miou_val, epoch_log)
                    is_best = miou_val > best_miou_val
                    best_miou_val = max(miou_val, best_miou_val)
                    if is_best:
                        best_loss_epoch = epoch_log
                        logger.info('update best val iou in epoch {} : {:.4f}'.format(best_loss_epoch, best_miou_val))
                        filename = args.save_path + '/best_checkpoint.pth'
                        torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr

def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr

def train(train_loader, model, optimizer, epoch, scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    model.train()

    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time()-end)
        """Update Learning Rate"""
        current_iter = epoch * len(train_loader) + i + 1
        if scheduler is not None:
            if args.lr_policy == 'warmup':
                if current_iter < args.warmup_iter and args.resume is None:
                    current_lr = current_iter / args.warmup_iter * args.base_lr
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = current_lr
                else:
                    scheduler.step()
                    current_lr = scheduler.get_lr()[0]
            else:
                if args.lr_policy == 'poly':
                    current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)
                    if args.pretrain and args.diff_lr:
                        for index in range(0, args.index_split):
                            optimizer.param_groups[index]['lr'] = current_lr
                        for index in range(args.index_split, len(optimizer.param_groups)):
                            optimizer.param_groups[index]['lr'] = current_lr * 10
                    else:
                        for index in range(0, len(optimizer.param_groups)):
                            optimizer.param_groups[index]['lr'] = current_lr
                elif args.lr_policy == 'step':
                    current_lr = step_learning_rate(args.base_lr, epoch, step_epoch=40, multiplier=0.1)
                    if args.pretrain and args.diff_lr:
                        for index in range(0, args.index_split):
                            optimizer.param_groups[index]['lr'] = current_lr
                        for index in range(args.index_split, len(optimizer.param_groups)):
                            optimizer.param_groups[index]['lr'] = current_lr * 10
                    else:
                        for index in range(0, len(optimizer.param_groups)):
                            optimizer.param_groups[index]['lr'] = current_lr
                else:
                    current_lr = optimizer.param_group[0]['lr']

        """Forward Model and Calculate Loss"""
        inputs, targets = sample['inputs'], sample['targets']
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        pred = model(inputs)
        loss = criterion(pred, targets)
        optimizer.zero_grad()
        if args.use_apex and args.multiprocessing_distributed:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        n = inputs.size(0)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time()-end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss {aux_loss_meter.val:.4f} '
                        'Loss {loss_meter.val:.4f} '.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time,
                                                            remain_time=remain_time,
                                                            main_loss_meter=main_loss_meter,
                                                            aux_loss_meter=aux_loss_meter,
                                                            loss_meter=loss_meter))

        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
    return loss_meter.avg

def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    iou_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, sample_batched in enumerate(val_loader):
        data_time.update(time.time() - end)
        inputs, targets = sample_batched['concat'], sample_batched['crop_gt']
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        outputs = model(inputs)
        # loss
        loss = criterion(outputs, targets)
        n = inputs.size(0)
        if args.multiprocessing_distributed:
            loss = loss * n  # not considering ignore pixels
            count = targets.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss = loss / n
        else:
            loss = torch.mean(loss)
        loss_meter.update(loss.item(), inputs.size(0))
        # iou
        outputs = 1 / (1 + torch.exp(-outputs))
        pred_thres = args.pred_th
        outputs = (outputs > pred_thres).float()
        intersection = torch.sum(targets*outputs)
        union = torch.sum(targets) + torch.sum(outputs) - intersection
        iou = intersection / union
        n = inputs.size(0)
        if args.multiprocessing_distributed:
            iou = iou * n
            count = targets.new_tensor([n], dtype=torch.long)
            dist.all_reduce(iou), dist.all_reduce(count)
            n = count.item()
            iou = iou / n
        else:
            iou = torch.mean(iou)
        iou_meter.update(iou.item(), inputs.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % args.print_freq == 0) and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'iou {iou_meter.val:.4f} ({iou_meter.avg:.4f}) '.format(i + 1, len(val_loader),
                                                                                   data_time=data_time,
                                                                                   batch_time=batch_time,
                                                                                   loss_meter=loss_meter,
                                                                                   iou_meter=iou_meter))
    return loss_meter.avg, iou_meter.avg


if __name__ == '__main__':
    main()