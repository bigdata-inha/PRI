import argparse
import shutil
import time
import os
import sys
import random
from sched import scheduler

import numpy as np
# import timm
import torch.optim.lr_scheduler
import tempfile
import types
from quantization import *
from synthesis import PMI
# from synthesis import PSAQ_VIT_SMI
from utils import *
from utils.data_utils import find_non_zero_patches
from quantization.vit_lsq import deit_tiny_patch16_224_lsq, deit_base_patch16_224_lsq, deit_small_patch16_224_lsq, lsq_initialize
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP

def get_args_parser():
    parser = argparse.ArgumentParser(description="pri", add_help=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--local_rank", type=int, help="local rank for distributed training")
    parser.add_argument("--model", default="deit_tiny_16_imagenet",
                        help="deit_base_16_imagenet/deit_small_16_imagenet")
    parser.add_argument('--dataset', default="./dataset/imagenet",
                        help='path to dataset')
    parser.add_argument('--datapool', default="./inverted_data_q",
                        help='path to datapool')
    parser.add_argument("--calib-batchsize", default=128,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batchsize", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--g_epoches", default=79, type=int, help="how many samples will you generate?")
    parser.add_argument("--t_epoches", default=100, type=int, help="how many epoches will you train?")
    parser.add_argument("--mode", default=0,
                        type=int, help="mode of calibration data, 0: inversion, 1: Gaussian noise")
    parser.add_argument('--w_bit', default=8,
                        type=int, help='bit-precision of weights')
    parser.add_argument('--a_bit', default=8,
                        type=int, help='bit-precision of activation')
    parser.add_argument('--iterations', nargs='+', type=int, default=4000,
                        help='number of iteration, default: 4000')
    parser.add_argument('--div', nargs='+', type=float, default=4,
                        help='To control the number of patch detachments, we define a division factor $v$, which determines into how many partitions a full-size image will be split. If div=2, each image sparsity is 50%. If div=4, each image sparsity is 75%.')

    return parser
def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)

class Config:
    def __init__(self, w_bit, a_bit):
        self.weight_bit = w_bit
        self.activation_bit = a_bit

def get_teacher(name):
    teacher_name = {'deit_tiny_16_imagenet': 'deit_tiny_patch16_224',
                    'deit_small_16_imagenet': 'deit_small_patch16_224',
                    'deit_base_16_imagenet': 'deit_base_patch16_224',
                    'vit_tiny_16_imagenet': 'vit_tiny_patch16_224',
                    'vit_small_16_imagenet': 'vit_small_patch16_224',
                    'vit_base_16_imagenet': 'vit_base_patch16_224',
                    }
    if args.model.split("_")[-1]=='imagenet':
        teacher=build_model(teacher_name[name], Pretrained=True)
    else:
        raise NotImplementedError
    return teacher
def get_student(name):
    model_zoo = {'deit_tiny_16_imagenet': deit_tiny_patch16_224_lsq,
                 'deit_small_16_imagenet': deit_small_patch16_224_lsq,
                 'deit_base_16_imagenet': deit_base_patch16_224_lsq,
                 }
    print('Model: %s' % model_zoo[name].__name__)
    return model_zoo[name]

def seed(seed=0):
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def setup_distributed(args):
    """Initialize distributed training."""
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # GPU 설정
    if len(args.gpu) > 1:
        dist.init_process_group(backend="nccl", init_method="env://")  # DDP 초기화
        args.world_size = dist.get_world_size()
        args.rank = dist.get_rank()
        args.distributed = args.world_size > 1  # 분산 학습 여부 결정
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)  # 현재 프로세스에 맞는 GPU 사용

    else:
        args.local_rank = 0
        args.distributed = False
    return args.local_rank

def main():
    seed(args.seed)
    print('seed:', args.seed)
    print(args)

    local_rank = setup_distributed(args)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    # Load bit-config
    cfg = Config(args.w_bit, args.a_bit)

    # Build model
    model = get_student(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)
    model.eval()
    teacher = get_teacher(args.model)
    teacher = teacher.to(device)
    teacher.eval()
    #
    #     # Build dataloader
    train_loader, val_loader, num_classes, train_transform, _, normalizer = build_dataset(args.model.split("_")[0],args.model.split("_")[-1],args.calib_batchsize,train_aug=False,keep_zero=True,train_inverse=True,dataset_path=args.dataset)
    T=20
    criterion_kd = KLDiv(T=T)
    optimizer_kd = torch.optim.SGD(model.parameters(), 0.001, weight_decay=1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_kd,milestones=[], gamma=0.1, last_epoch=-1)
    #     # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Get calibration set
    # Case 0: inversion
    if args.mode == 0:
        iterations=args.iterations#total number of iterations for inversion
        div=args.div
        lr_g=0.25#learning rate for inversion
        #coefficient for inversion
        adv=0 #coefficient of adversarial regularization, we do not use it in our work
        bn=0.0 #coefficient of batch normalization regularization, dose not apply to ViTs due to the absence of batch normalization, we borrow a CNN to only facilitate visualization (refer to GradViT: Gradient Inversion of Vision Transformers)
        oh=1 #coefficient of classification loss
        tv1=0#coefficient of total variance regularization with l1 norm, we do not use it in our work
        tv2=0.0001#coefficient of total variance regularization with l2 norm
        l2=0#coefficient of l2 norm regularization, we do not use it in our work
        patch_size=16 if '16' in args.model else 32
        patch_num=197 if patch_size==16 else 50
        img_tag = "{}-{}-{}-{}-{}-{}-{}-{}-{}-test".format(iterations, lr_g, adv, bn, oh, tv1, tv2, l2, str(div))
        datapool_path = os.path.join(args.datapool, '%s/%s' % (args.model, img_tag))  # the path to store inverted data

        synthesizer = PMI(
            teacher=teacher, teacher_name=args.model, student=model, num_classes=num_classes,
            img_shape=(3, 224, 224), iterations=iterations, patch_size=patch_size, lr_g=lr_g,
            synthesis_batch_size=args.calib_batchsize // div, sample_batch_size=args.calib_batchsize,
            adv=adv, bn=bn, oh=oh, tv1=tv1, tv2=tv2, l2=l2,
            save_dir=datapool_path, transform=train_transform,
            normalizer=normalizer, device=device, bnsource='resnet50v1', init_dataset=None)
        for g_epoch in range(args.g_epoches):
            data_loader = synthesizer.synthesize(num_patches=patch_num, div=div)
        lsq_initialize(model, device)
        model.train()

        for t_epoch in tqdm(range(args.t_epoches)):
            for images in data_loader:
                images = images.to(device)
                with torch.no_grad():
                    output_t=teacher(images,current_abs_index=torch.arange(patch_num).repeat(images.shape[0], 1).to(images.device),next_relative_index=torch.cat([torch.zeros(images.shape[0], 1, dtype=torch.long).to(images.device), find_non_zero_patches(images=images,patch_size=patch_size)], dim=1))
                aaa=torch.cat([torch.zeros(images.shape[0], 1, dtype=torch.long).to(images.device), find_non_zero_patches(images=images,patch_size=patch_size)], dim=1)
                output_s = model(images,current_abs_index=torch.arange(patch_num).repeat(images.shape[0], 1).to(images.device),next_relative_index=aaa)
                loss_cal=criterion_kd(output_s,output_t[0].detach())
                optimizer_kd.zero_grad()
                loss_cal.backward()
                optimizer_kd.step()

        model.eval()
        # Validate the quantized model
        print("Validating...")
        val_loss, val_prec1, val_prec5 = validate(
            args, val_loader, model, criterion, device
        )



def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for data, target in tqdm(val_loader):
        data = data.to(device)
        target = target.to(device)
        patch_num = 197

        with torch.no_grad():
            output = model(data)
            # output = model(data)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    val_end_time = time.time()
    print(" * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}".format(
        top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser('PMI_test_quant', parents=[get_args_parser()])
    args = parser.parse_args()
    main()

