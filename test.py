"""
eval a single checkpoint using exactly the same val() logic as train.py.

Single GPU:
    python test.py --model SpikingLET --dataset udd6 \
        --checkpoint /path/to/model_40.pth \
        --T 8 --thresh 1.0 --tau 0.5 --gamma 2.0 --cfg_n 2

DDP (faster, same as train.py's distributed val):
    torchrun --nproc_per_node=4 test.py --model SpikingLET --dataset udd6 \
        --checkpoint /path/to/model_40.pth \
        --T 8 --thresh 1.0 --tau 0.5 --gamma 2.0 --cfg_n 2
"""
import os
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser

from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight
from utils.metric.metric import ConfusionMatrix

GLOBAL_SEED = 1234


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank       = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0; args.world_size = 1; args.local_rank = 0
        args.distributed = False
    if args.cuda:
        torch.cuda.set_device(args.local_rank)
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        if args.cuda:
            dist.barrier(device_ids=[args.local_rank])
        else:
            dist.barrier()


def is_main(args):
    return args.rank == 0


def _compute_iou_from_confusion_matrix(matrix):
    matrix = matrix.astype(np.float64)
    diagonal = np.diag(matrix)
    denominator = matrix.sum(axis=1) + matrix.sum(axis=0) - diagonal
    per_class_iou = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    valid = denominator > 0
    per_class_iou[valid] = diagonal[valid] / denominator[valid]
    mean_iou = np.nanmean(per_class_iou) if np.any(valid) else 0.0
    return mean_iou, per_class_iou


def _sliding_window_inference(model, images, crop_size, stride):
    B, _, H, W = images.shape
    ch, cw = crop_size
    sh, sw = stride
    pad_h = max(ch - H, 0); pad_w = max(cw - W, 0)
    if pad_h > 0 or pad_w > 0:
        images = F.pad(images, (0, pad_w, 0, pad_h))
    ph, pw = images.shape[2], images.shape[3]
    ys = list(range(0, max(ph - ch, 0) + 1, sh))
    xs = list(range(0, max(pw - cw, 0) + 1, sw))
    if ys[-1] != ph - ch: ys.append(ph - ch)
    if xs[-1] != pw - cw: xs.append(pw - cw)
    score_map = count_map = None
    for y in ys:
        for x in xs:
            logits = model(images[:, :, y:y+ch, x:x+cw])
            if score_map is None:
                score_map = torch.zeros((B, logits.shape[1], ph, pw),
                                        device=logits.device, dtype=logits.dtype)
                count_map = torch.zeros((B, 1, ph, pw),
                                        device=logits.device, dtype=logits.dtype)
            score_map[:, :, y:y+ch, x:x+cw] += logits
            count_map[:, :, y:y+ch, x:x+cw] += 1
    return (score_map / count_map)[:, :, :H, :W]


# ── exact copy of train.py val() ─────────────────────────────────────────────
def val(args, val_loader, model):
    model.eval()
    total_batches = len(val_loader)
    use_ottt = args.model == 'SpikingLET' and getattr(args, 'neuron_mode', 'bptt') == 'ottt'
    crop_size = tuple(int(v) for v in args.val_crop_size.split(','))
    stride    = tuple(int(v) for v in args.val_stride.split(','))

    conf_matrix = ConfusionMatrix(args.classes, ignore_label=args.ignore_label)
    for i, (input, label, size, name) in enumerate(val_loader):
        start_time = time.time()
        with torch.no_grad():
            input_var = input.cuda()
            if args.val_mode == 'slidingwindow':
                output = _sliding_window_inference(model, input_var, crop_size, stride)
            else:
                output = model(input_var)
        time_taken = time.time() - start_time
        if is_main(args):
            print(f'[{i+1}/{total_batches}]  time: {time_taken:.2f}', flush=True)
        output = output.cpu().data[0].numpy()
        gt     = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = np.asarray(np.argmax(output.transpose(1, 2, 0), axis=2), dtype=np.uint8)
        conf_matrix.add(gt.flatten(), output.flatten())

    if args.distributed:
        m = torch.from_numpy(conf_matrix.M).to(device=input_var.device, dtype=torch.float32)
        dist.all_reduce(m, op=dist.ReduceOp.SUM)
        conf_matrix.M = m.cpu().numpy()

    meanIoU, per_class_iu = _compute_iou_from_confusion_matrix(conf_matrix.M)
    if is_main(args):
        for idx, iou in enumerate(per_class_iu):
            print(f'class_{idx}: {iou}')
        print(f'mIoU: {meanIoU}')
    return meanIoU, per_class_iu


# ── args ──────────────────────────────────────────────────────────────────────
def parse_args():
    def str2bool(v):
        if isinstance(v, bool): return v
        if v.lower() in ('yes','true','t','y','1'): return True
        if v.lower() in ('no','false','f','n','0'): return False
        raise ValueError(v)

    p = ArgumentParser()
    p.add_argument('--model',       default="ENet")
    p.add_argument('--dataset',     default="camvid")
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--cuda',        type=str2bool, default=True)
    p.add_argument('--gpus',        default="0")
    p.add_argument('--checkpoint',  type=str, required=True)
    # inference (same defaults as train.py)
    p.add_argument('--val_mode',      default='slidingwindow',
                   choices=['direct', 'slidingwindow'])
    p.add_argument('--val_crop_size', default='1024,1024')
    p.add_argument('--val_stride',    default='768,768')
    # SpikingLET
    p.add_argument('--T',           type=int,   default=8)
    p.add_argument('--thresh',      type=float, default=1.0)
    p.add_argument('--tau',         type=float, default=0.5)
    p.add_argument('--gamma',       type=float, default=2.0)
    p.add_argument('--cfg_n',       type=int,   default=2)
    p.add_argument('--neuron_mode', default='bptt', choices=['bptt','ottt'])
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    dataset_key = args.dataset.lower()
    if dataset_key == 'cityscapes':
        args.classes = 19; args.ignore_label = 255
    elif dataset_key == 'camvid':
        args.classes = 11; args.ignore_label = 11
    elif dataset_key == 'udd6':
        args.classes = 6;  args.ignore_label = 255
    else:
        raise NotImplementedError(args.dataset)

    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("No GPU found")

    init_distributed_mode(args)

    if not args.distributed:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    setup_seed(GLOBAL_SEED + args.rank)
    cudnn.enabled = True; cudnn.benchmark = True

    # build model
    if args.model == 'SpikingLET':
        model = build_model(args.model, num_classes=args.classes,
                            T=args.T, thresh=args.thresh, tau=args.tau,
                            gamma=args.gamma, cfg_n=args.cfg_n,
                            neuron_mode=args.neuron_mode)
    else:
        model = build_model(args.model, num_classes=args.classes)
    init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, 1e-3, 0.1, mode='fan_in')

    # load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if is_main(args):
        print(f"Loaded: {args.checkpoint}")
        if missing:    print(f"  missing keys: {len(missing)}")
        if unexpected: print(f"  unexpected keys: {len(unexpected)}")

    model = model.cuda()
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank)

    # val loader — same call as train.py
    h, w = 1024, 1024
    datas, _, valLoader, _, _ = build_dataset_train(
        args.dataset, (h, w), batch_size=1, train_type='train',
        random_scale=False, random_mirror=False, num_workers=args.num_workers,
        distributed=args.distributed, rank=args.rank, world_size=args.world_size,
        seed=GLOBAL_SEED, val_batch_size=1, repeat_times=1,
        random_rotate=False, vertical_flip=False, normalize=True,
        train_mode='direct', train_crop_size=None, train_stride=None,
        ignore_label=args.ignore_label)

    if is_main(args):
        print(f"Val set: {len(valLoader)} images")

    meanIoU, per_class_iu = val(args, valLoader, model)

    # save result next to checkpoint
    if is_main(args):
        log_path = os.path.splitext(args.checkpoint)[0] + '_test.txt'
        with open(log_path, 'w') as f:
            f.write(f"checkpoint: {args.checkpoint}\n")
            f.write(f"mIoU: {meanIoU:.4f}\n")
            for idx, v in enumerate(per_class_iu):
                f.write(f"class_{idx}: {v:.4f}\n")
        print(f"Saved → {log_path}")

    if args.distributed:
        dist.destroy_process_group()
