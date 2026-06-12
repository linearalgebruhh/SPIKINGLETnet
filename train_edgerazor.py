import os
import time
import math
import timeit
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch import optim
from contextlib import nullcontext
from argparse import ArgumentParser

# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_train
from utils.utils import setup_seed, init_weight, netParams
from utils.metric.metric import ConfusionMatrix
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth, \
    ProbOhemCrossEntropy2d, FocalLoss2d
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR

import EdgeRazor 

GLOBAL_SEED = 1234

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.distributed = True
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if args.cuda:
        torch.cuda.set_device(args.local_rank)

    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        if args.cuda:
            dist.barrier(device_ids=[args.local_rank])
        else:
            dist.barrier()


def is_main_process(args):
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


def _forward_logits(model, images, use_ottt, t_step):
    if not use_ottt:
        return model(images)

    logits_sum = None
    for t in range(t_step):
        output_t = model(images, init=(t == 0), step_mode=True)
        if logits_sum is None:
            logits_sum = output_t
        else:
            logits_sum = logits_sum + output_t
    return logits_sum / t_step


def _sliding_window_inference(model, images, crop_size=(1024, 1024), stride=(768, 768), use_ottt=False, t_step=1):
    batch_size, _, height, width = images.shape
    crop_h, crop_w = crop_size
    stride_h, stride_w = stride

    pad_h = max(crop_h - height, 0)
    pad_w = max(crop_w - width, 0)
    if pad_h > 0 or pad_w > 0:
        images = F.pad(images, (0, pad_w, 0, pad_h))

    padded_h, padded_w = images.shape[2], images.shape[3]
    y_positions = list(range(0, max(padded_h - crop_h, 0) + 1, stride_h))
    x_positions = list(range(0, max(padded_w - crop_w, 0) + 1, stride_w))
    if y_positions[-1] != padded_h - crop_h:
        y_positions.append(padded_h - crop_h)
    if x_positions[-1] != padded_w - crop_w:
        x_positions.append(padded_w - crop_w)

    score_map = None
    count_map = None

    for y in y_positions:
        for x in x_positions:
            crop = images[:, :, y:y + crop_h, x:x + crop_w]
            logits = _forward_logits(model, crop, use_ottt, t_step)
            if score_map is None:
                score_map = torch.zeros((batch_size, logits.shape[1], padded_h, padded_w), device=logits.device, dtype=logits.dtype)
                count_map = torch.zeros((batch_size, 1, padded_h, padded_w), device=logits.device, dtype=logits.dtype)
            score_map[:, :, y:y + crop_h, x:x + crop_w] += logits
            count_map[:, :, y:y + crop_h, x:x + crop_w] += 1

    score_map = score_map / count_map
    return score_map[:, :, :height, :width]


def _infer_logits(model, images, infer_mode='slidingwindow', crop_size=(1024, 1024), stride=(768, 768), use_ottt=False, t_step=1):
    if infer_mode == 'direct':
        return _forward_logits(model, images, use_ottt, t_step)
    return _sliding_window_inference(model, images, crop_size=crop_size, stride=stride, use_ottt=use_ottt, t_step=t_step)


def _set_optimizer_lr(optimizer, lr_value):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_value
        param_group['initial_lr'] = lr_value


def _load_checkpoint(path, model, is_main):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if is_main:
        if missing:
            print(f"[EdgeRazor] Missing keys when loading {path}: {len(missing)}")
        if unexpected:
            print(f"[EdgeRazor] Unexpected keys when loading {path}: {len(unexpected)}")
    return checkpoint


def _make_deploy_model(model, edgerazor, distributed):
    """Deep-copy the QAT model and convert weights to int4 without touching the training model."""
    import copy
    src = model.module if distributed else model
    deploy = copy.deepcopy(src)
    # Force is_w_quantized=False so replace_quantized_weights always runs,
    # regardless of what the yaml config says.
    for m in deploy.modules():
        if hasattr(m, 'is_w_quantized'):
            m.is_w_quantized = False
    deploy = edgerazor.replace_quantized_weights(deploy)
    return deploy


def _prepare_kd_tensors(student_logits, teacher_logits, labels):
    """
    Align logits/labels for KD loss.

    EdgeRazor KD expects logits shaped like [B, seq_len, vocab].
    For segmentation logits [B, C, H, W], flatten to [B, H*W, C].
    """
    if student_logits.dim() == 4:
        # [B, C, H, W] -> [B, H*W, C]
        b, c, h, w = student_logits.shape
        student_logits = student_logits.permute(0, 2, 3, 1).reshape(b, h * w, c)
        teacher_logits = teacher_logits.permute(0, 2, 3, 1).reshape(b, h * w, c)
        labels = labels.view(b, -1)
    elif student_logits.dim() == 2:
        # [B, C] -> [B, 1, C]
        student_logits = student_logits.unsqueeze(1)
        teacher_logits = teacher_logits.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

    return student_logits, teacher_logits, labels


def parse_args():
    def str2bool(value):
        if isinstance(value, bool):
            return value
        value = value.lower()
        if value in ('yes', 'true', 't', 'y', '1'):
            return True
        if value in ('no', 'false', 'f', 'n', '0'):
            return False
        raise ValueError(f"Unsupported bool string format: {value}")

    parser = ArgumentParser(description='Efficient semantic segmentation (EdgeRazor QAT)')
    # model and dataset
    parser.add_argument('--model', type=str, default="ENet", help="model name: (default ENet)")
    parser.add_argument('--dataset', type=str, default="camvid", help="dataset: cityscapes, camvid, or UDD6")
    parser.add_argument('--input_size', type=str, default="360,480", help="input size of model")
    parser.add_argument('--T', type=int, default=1, help="time steps for spiking models")
    parser.add_argument('--thresh', type=float, default=0.5, help="spiking threshold")
    parser.add_argument('--tau', type=float, default=0.5, help="membrane decay")
    parser.add_argument('--gamma', type=float, default=2.0, help="surrogate gradient scale")
    parser.add_argument('--neuron_mode', type=str, default='bptt', choices=['bptt', 'ottt'],
                        help="spiking neuron mode for SpikingLET")
    parser.add_argument('--cfg_n', type=int, default=2, help="choose SpikingLET config: 1 small, 2 base, 3 large")
    parser.add_argument('--num_workers', type=int, default=4, help=" the number of parallel threads")
    parser.add_argument('--classes', type=int, default=11,
                        help="the number of classes in the dataset. 19 and 11 for cityscapes and camvid, respectively")
    parser.add_argument('--train_type', type=str, default="trainval",
                        help="ontrain for training on train set, ontrainval for training on train+val set")
    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help="the number of epochs: 300 for train set, 350 for train+val set")
    parser.add_argument('--random_mirror', type=str2bool, default=True, help="input image random mirror")
    parser.add_argument('--random_scale', type=str2bool, default=True, help="input image resize 0.5 to 2")
    parser.add_argument('--random_rotate', type=str2bool, default=True, help="input image random rotate by 90 or 270 degrees")
    parser.add_argument('--vertical_flip', type=str2bool, default=True, help="input image random vertical flip")
    parser.add_argument('--normalize', type=str2bool, default=True, help="normalize image with dataset mean/std")
    parser.add_argument('--repeat_times', type=int, default=5, help="repeat the training set this many times per epoch")
    parser.add_argument('--lr', type=float, default=6e-4, help="initial learning rate")
    parser.add_argument('--batch_size', type=int, default=8, help="the batch size is set to 16 for 2 GPUs")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam', 'adamw', 'radam', 'ranger'], help="select optimizer")
    parser.add_argument('--lr_schedule', type=str, default='poly', help='name of lr schedule: poly')
    parser.add_argument('--num_cycles', type=int, default=1, help='Cosine Annealing Cyclic LR')
    parser.add_argument('--poly_exp', type=float, default=0.9, help='polynomial LR exponent')
    parser.add_argument('--warmup_iters', type=int, default=500, help='warmup iterations')
    parser.add_argument('--warmup_factor', type=float, default=1.0 / 3, help='warm up start lr=warmup_factor*lr')
    parser.add_argument('--amp', type=str2bool, default=False, help="use mixed precision (autocast + GradScaler)")
    parser.add_argument('--use_label_smoothing', action='store_true', default=False, help="CrossEntropy2d Loss with label smoothing or not")
    parser.add_argument('--use_ohem', action='store_true', default=False, help='OhemCrossEntropy2d Loss for cityscapes dataset')
    parser.add_argument('--use_lovaszsoftmax', action='store_true', default=False, help='LovaszSoftmax Loss for cityscapes dataset')
    parser.add_argument('--use_focal', action='store_true', default=False, help=' FocalLoss2d for cityscapes dataset')
    parser.add_argument('--val_mode', type=str, default='slidingwindow', choices=['direct', 'slidingwindow'],
                        help='validation inference mode')
    parser.add_argument('--val_crop_size', type=str, default='1024,1024', help='validation crop size for sliding-window mode')
    parser.add_argument('--val_stride', type=str, default='768,768', help='validation stride for sliding-window mode')
    parser.add_argument('--train_mode', type=str, default='direct', choices=['direct', 'slidingwindow'],
                        help='training data mode: direct (random crop) or slidingwindow (pre-cropped dataset on disk)')
    parser.add_argument('--train_crop_size', type=str, default='512,512', help='training crop size for sliding-window mode')
    parser.add_argument('--train_stride', type=str, default='256,256', help='training stride for sliding-window mode')
    # cuda setting
    parser.add_argument('--cuda', type=str2bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="0", help="default GPU devices (0,1)")
    # checkpoint and log
    parser.add_argument('--resume', type=str, default="",
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--resume_lr_mode', type=str, default='args', choices=['args', 'checkpoint'],
                        help="when resuming, use args.lr or the checkpoint base lr")
    parser.add_argument('--savedir', default="./checkpoint/", help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")

    # EdgeRazor
    parser.add_argument('--edgerazor_config', type=str, default="", help="path to EdgeRazor config (yaml/json)")
    parser.add_argument('--student_pretrained_path', type=str, default="", help="path to pretrained student checkpoint")
    parser.add_argument('--teacher_pretrained_path', type=str, default="", help="path to pretrained teacher checkpoint")

    args = parser.parse_args()
    if not args.edgerazor_config:
        raise ValueError("--edgerazor_config is required for train_edgerazor")
    if args.student_pretrained_path and not os.path.isfile(args.student_pretrained_path):
        raise FileNotFoundError(f"student checkpoint not found: {args.student_pretrained_path}")
    if args.teacher_pretrained_path and not os.path.isfile(args.teacher_pretrained_path):
        raise FileNotFoundError(f"teacher checkpoint not found: {args.teacher_pretrained_path}")

    return args


def train_model(args):
    dataset_key = args.dataset.lower()
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    print("=====> input size:{}".format(input_size))

    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    init_distributed_mode(args)

    # set the seed
    setup_seed(GLOBAL_SEED + args.rank)
    if is_main_process(args):
        print("=====> set Global Seed: ", GLOBAL_SEED)

    cudnn.enabled = True
    print("=====> building network")

    # build the model and initialization
    if args.model == 'SpikingLET':
        model = build_model(args.model, num_classes=args.classes, T=args.T, thresh=args.thresh, tau=args.tau,
                            gamma=args.gamma, cfg_n=args.cfg_n, neuron_mode=args.neuron_mode)
    else:
        model = build_model(args.model, num_classes=args.classes)
    init_weight(model, nn.init.kaiming_normal_,
                nn.BatchNorm2d, 1e-3, 0.1,
                mode='fan_in')

    if args.student_pretrained_path:
        _load_checkpoint(args.student_pretrained_path, model, is_main_process(args))
        if is_main_process(args):
            print(f"=====> loaded student checkpoint: {args.student_pretrained_path}")

    # EdgeRazor setup (QAT or QAT+KD)
    edgerazor = EdgeRazor(config=args.edgerazor_config)
    if edgerazor.is_qat_enabled:
        model = edgerazor.quantize(model)
        if is_main_process(args):
            print("=====> EdgeRazor QAT enabled")
    else:
        if is_main_process(args):
            print("=====> EdgeRazor QAT disabled")

    teacher_model = None
    if edgerazor.is_kd_enabled and args.teacher_pretrained_path:
        if args.model == 'SpikingLET':
            teacher_model = build_model(args.model, num_classes=args.classes, T=args.T, thresh=args.thresh,
                                        tau=args.tau, gamma=args.gamma, cfg_n=args.cfg_n, neuron_mode=args.neuron_mode)
        else:
            teacher_model = build_model(args.model, num_classes=args.classes)
        init_weight(teacher_model, nn.init.kaiming_normal_,
                    nn.BatchNorm2d, 1e-3, 0.1,
                    mode='fan_in')
        _load_checkpoint(args.teacher_pretrained_path, teacher_model, is_main_process(args))
        for p in teacher_model.parameters():
            p.requires_grad = False
        teacher_model.eval()
        if is_main_process(args):
            print(f"=====> loaded teacher checkpoint: {args.teacher_pretrained_path}")
    elif edgerazor.is_kd_enabled and is_main_process(args):
        print("=====> KD enabled but no teacher checkpoint provided; KD will be skipped")

    print("=====> computing network parameters and FLOPs")
    total_paramters = netParams(model)
    if is_main_process(args):
        print("the number of parameters: %d ==> %.2f M" % (total_paramters, (total_paramters / 1e6)))

    # load data and data augmentation
    train_mode = getattr(args, 'train_mode', 'direct')
    train_crop_size = tuple(int(v) for v in args.train_crop_size.split(',')) if train_mode == 'slidingwindow' else None
    train_stride = tuple(int(v) for v in args.train_stride.split(',')) if train_mode == 'slidingwindow' else None
    datas, trainLoader, valLoader, train_sampler, val_sampler = build_dataset_train(
        args.dataset, input_size, args.batch_size, args.train_type,
        args.random_scale, args.random_mirror, args.num_workers,
        distributed=args.distributed, rank=args.rank, world_size=args.world_size, seed=GLOBAL_SEED,
        val_batch_size=1, repeat_times=args.repeat_times, random_rotate=args.random_rotate,
        vertical_flip=args.vertical_flip, normalize=args.normalize,
        train_mode=train_mode, train_crop_size=train_crop_size, train_stride=train_stride,
        ignore_label=args.ignore_label)

    args.per_iter = len(trainLoader)
    args.max_iter = args.max_epochs * args.per_iter

    if is_main_process(args):
        print('=====> Dataset statistics')
        print("data['classWeights']: ", datas['classWeights'])
        print('mean and std: ', datas['mean'], datas['std'])

    # define loss function, respectively
    weight = torch.from_numpy(datas['classWeights'])

    if dataset_key == 'camvid' and args.use_label_smoothing:
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
    elif dataset_key == 'camvid':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif dataset_key == 'udd6' and args.use_label_smoothing:
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
    elif dataset_key == 'udd6' and args.use_lovaszsoftmax:
        criteria = LovaszSoftmax(ignore_index=ignore_label)
    elif dataset_key == 'udd6' and args.use_ohem:
        min_kept = int(args.batch_size * h * w // 16)
        udd6_class_ratio = torch.tensor([18.64, 15.33, 13.13, 27.77, 0.92, 24.21], dtype=torch.float32)
        udd6_ohem_weight = 1.0 / udd6_class_ratio
        udd6_ohem_weight = udd6_ohem_weight / udd6_ohem_weight.mean()
        criteria = ProbOhemCrossEntropy2d(
            use_weight=True,
            weight=udd6_ohem_weight,
            ignore_label=ignore_label,
            thresh=0.7,
            min_kept=min_kept,
        )
    elif dataset_key == 'udd6' and args.use_focal:
        criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)
    elif dataset_key == 'udd6':
        criteria = CrossEntropyLoss2d(weight=weight, ignore_label=ignore_label)
    elif dataset_key == 'cityscapes' and args.use_ohem:
        gpu_count = args.world_size if args.distributed else len(args.gpus.split(','))
        min_kept = int(args.batch_size // gpu_count * h * w // 16)
        criteria = ProbOhemCrossEntropy2d(use_weight=True, ignore_label=ignore_label, thresh=0.7, min_kept=min_kept)
    elif dataset_key == 'cityscapes' and args.use_label_smoothing:
        criteria = CrossEntropyLoss2dLabelSmooth(weight=weight, ignore_label=ignore_label)
    elif dataset_key == 'cityscapes' and args.use_lovaszsoftmax:
        criteria = LovaszSoftmax(ignore_index=ignore_label)
    elif dataset_key == 'cityscapes' and args.use_focal:
        criteria = FocalLoss2d(weight=weight, ignore_index=ignore_label)
    else:
        raise NotImplementedError(
            "This repository now supports cityscapes, camvid, and udd6, %s is not included" % args.dataset)

    if args.cuda:
        criteria = criteria.cuda()
        model = model.cuda()
        if teacher_model is not None:
            teacher_model = teacher_model.cuda()
        if args.distributed:
            args.gpu_nums = args.world_size
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank)
        else:
            args.gpu_nums = 1
            if is_main_process(args):
                print("single GPU for training")

    config_stem = os.path.splitext(os.path.basename(args.edgerazor_config))[0]
    args.savedir = (args.savedir + args.dataset + '/' + args.model + 'bs'
                    + str(args.batch_size) + 'gpu' + str(args.gpu_nums) + 'cfg' + str(args.cfg_n)
                    + 'lovasz' + str(args.use_lovaszsoftmax) + '_' + str(args.train_type)
                    + '_edgerazor_' + config_stem + '/')

    if is_main_process(args) and not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    start_epoch = 0
    resume_checkpoint = None
    if args.resume and os.path.isfile(args.resume):
        resume_checkpoint = torch.load(args.resume, map_location='cpu')
        if args.resume_lr_mode == 'checkpoint':
            args.lr = resume_checkpoint.get('base_lr', resume_checkpoint.get('lr', args.lr))

    # continue training
    if args.resume:
        if resume_checkpoint is not None:
            start_epoch = resume_checkpoint.get('epoch', 0)
            if resume_checkpoint.get('deploy_mode') == 'int4':
                if is_main_process(args):
                    print("=====> int4 checkpoint detected; model weights skipped (QAT starts fresh from epoch {})".format(start_epoch))
            else:
                target_model = model.module if args.distributed else model
                target_model.load_state_dict(resume_checkpoint['model'])
                if is_main_process(args):
                    print("=====> loaded checkpoint '{}' (epoch {})".format(args.resume, start_epoch))
        else:
            if is_main_process(args):
                print("=====> no checkpoint found at '{}'".format(args.resume))

    model.train()
    cudnn.benchmark = True

    logger = None
    if is_main_process(args):
        logFileLoc = args.savedir + args.logFile
        if os.path.isfile(logFileLoc):
            logger = open(logFileLoc, 'a')
        else:
            logger = open(logFileLoc, 'w')
            logger.write("Parameters: %s Seed: %s" % (str(total_paramters), GLOBAL_SEED))
            logger.write("\n%s\t\t%s\t%s\t%s" % ('Epoch', 'Loss(Tr)', 'mIOU (val)', 'lr'))
        logger.flush()

    # define optimization strategy
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.optim == 'radam':
        optimizer = RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.90, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.optim == 'ranger':
        optimizer = Ranger(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4)
    elif args.optim == 'adamw':
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    if resume_checkpoint is not None and 'optimizer' in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint['optimizer'])
        _set_optimizer_lr(optimizer, args.lr)

    lossTr_list = []
    epoches = []
    mIOU_val_list = []

    if is_main_process(args):
        print('=====> beginning training')
    for epoch in range(start_epoch, args.max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        lossTr, lr = train(args, trainLoader, model, criteria, optimizer, epoch, edgerazor, teacher_model)
        lossTr_list.append(lossTr)

        if epoch % 10 == 0 or epoch == (args.max_epochs - 1):
            epoches.append(epoch)
            if edgerazor.is_qat_enabled:
                eval_model = _make_deploy_model(model, edgerazor, args.distributed)
                if args.cuda:
                    eval_model = eval_model.cuda()
                mIOU_val, per_class_iu = val(args, valLoader, eval_model)
                del eval_model
            else:
                mIOU_val, per_class_iu = val(args, valLoader, model)
            mIOU_val_list.append(mIOU_val)
            if logger is not None:
                logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.7f" % (epoch, lossTr, mIOU_val, lr))
                logger.flush()
            if is_main_process(args):
                print("Epoch : " + str(epoch) + ' Details')
                print("Epoch No.: %d\tTrain Loss = %.4f\t mIOU(val) = %.4f\t lr= %.6f\n" % (epoch,
                                                                                            lossTr,
                                                                                            mIOU_val, lr))
        else:
            if logger is not None:
                logger.write("\n%d\t\t%.4f\t\t\t\t%.7f" % (epoch, lossTr, lr))
                logger.flush()
            if is_main_process(args):
                print("Epoch : " + str(epoch) + ' Details')
                print("Epoch No.: %d\tTrain Loss = %.4f\t lr= %.6f\n" % (epoch, lossTr, lr))

        model_file_name = args.savedir + '/model_' + str(epoch + 1) + '.pth'
        if is_main_process(args):
            should_save = (
                dataset_key in ('camvid', 'udd6') or
                (dataset_key == 'cityscapes' and (epoch >= args.max_epochs - 10 or not epoch % 50))
            )
            if should_save:
                if edgerazor.is_qat_enabled:
                    deploy = _make_deploy_model(model, edgerazor, args.distributed)
                    int4_state = {
                        "epoch": epoch + 1,
                        "model": deploy.state_dict(),
                        "deploy_mode": "int4",
                        "base_lr": args.lr,
                    }
                    torch.save(int4_state, model_file_name)
                    del deploy
                else:
                    if args.distributed:
                        state = {"epoch": epoch + 1, "model": model.module.state_dict(), "optimizer": optimizer.state_dict(), "base_lr": args.lr}
                    else:
                        state = {"epoch": epoch + 1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "base_lr": args.lr}
                    torch.save(state, model_file_name)

    if logger is not None:
        logger.close()

    if args.distributed:
        dist.destroy_process_group()


def train(args, train_loader, model, criterion, optimizer, epoch, edgerazor, teacher_model):
    model.train()
    epoch_loss = []
    use_ottt = args.model == 'SpikingLET' and getattr(args, 'neuron_mode', 'bptt') == 'ottt'

    total_batches = len(train_loader)
    if is_main_process(args):
        print("=====> the number of iterations per epoch: ", total_batches)
    st = time.time()
    args.per_iter = total_batches
    args.max_iter = args.max_epochs * args.per_iter
    args.cur_iter = epoch * args.per_iter

    if args.lr_schedule == 'poly':
        lambda1 = lambda step: math.pow((1 - (args.cur_iter / args.max_iter)), args.poly_exp)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    elif args.lr_schedule == 'warmpoly':
        scheduler = WarmupPolyLR(optimizer, T_max=args.max_iter, cur_iter=args.cur_iter, warmup_factor=1.0 / 3,
                                 warmup_iters=args.warmup_iters, power=0.9)

    for iteration, batch in enumerate(train_loader, 0):

        args.cur_iter = epoch * args.per_iter + iteration
        lr = optimizer.param_groups[0]['lr']

        start_time = time.time()
        images, labels, _, _ = batch

        if torch_ver == '0.3':
            images = Variable(images).cuda()
            labels = Variable(labels.long()).cuda()
        else:
            images = images.cuda()
            labels = labels.long().cuda()

        use_kd = edgerazor is not None and edgerazor.is_kd_enabled and teacher_model is not None
        if use_kd:
            student_logits = _forward_logits(model, images, use_ottt, args.T)
            task_loss = criterion(student_logits, labels)
            with torch.no_grad():
                teacher_use_ottt = args.model == 'SpikingLET' and getattr(args, 'neuron_mode', 'bptt') == 'ottt'
                teacher_logits = _forward_logits(teacher_model, images, teacher_use_ottt, args.T)
            student_logits, teacher_logits, labels_kd = _prepare_kd_tensors(
                student_logits, teacher_logits, labels
            )
            student_outputs = {"loss": task_loss, "logits": student_logits}
            teacher_outputs = {"logits": teacher_logits}
            loss, loss_dict = edgerazor.compute_loss(student_outputs, teacher_outputs, labels_kd)
        else:
            student_logits = _forward_logits(model, images, use_ottt, args.T)
            loss = criterion(student_logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time

        if is_main_process(args):
            print('=====> epoch[%d/%d] iter: (%d/%d) \tcur_lr: %.6f loss: %.3f time:%.2f' % (epoch + 1, args.max_epochs,
                                                                                             iteration + 1, total_batches,
                                                                                             lr, loss.item(), time_taken))

    time_taken_epoch = time.time() - st
    remain_time = time_taken_epoch * (args.max_epochs - 1 - epoch)
    m, s = divmod(remain_time, 60)
    h, m = divmod(m, 60)
    if is_main_process(args):
        print("Remaining training time = %d hour %d minutes %d seconds" % (h, m, s))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def val(args, val_loader, model):
    model.eval()
    total_batches = len(val_loader)
    use_ottt = args.model == 'SpikingLET' and getattr(args, 'neuron_mode', 'bptt') == 'ottt'
    crop_size = tuple(int(v) for v in args.val_crop_size.split(','))
    stride = tuple(int(v) for v in args.val_stride.split(','))

    conf_matrix = ConfusionMatrix(args.classes, ignore_label=ignore_label)
    for i, (input, label, size, name) in enumerate(val_loader):
        start_time = time.time()
        with torch.no_grad():
            input_var = input.cuda()
            output = _infer_logits(model, input_var, infer_mode=args.val_mode, crop_size=crop_size, stride=stride,
                                   use_ottt=use_ottt, t_step=args.T)
        time_taken = time.time() - start_time
        if is_main_process(args):
            print("[%d/%d]  time: %.2f" % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        gt = np.asarray(label[0].numpy(), dtype=np.uint8)
        output = output.transpose(1, 2, 0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        conf_matrix.add(gt.flatten(), output.flatten())

    if args.distributed:
        m_tensor = torch.from_numpy(conf_matrix.M).to(device=input_var.device, dtype=torch.float32)
        dist.all_reduce(m_tensor, op=dist.ReduceOp.SUM)
        conf_matrix.M = m_tensor.cpu().numpy()

    meanIoU, per_class_iu = _compute_iou_from_confusion_matrix(conf_matrix.M)
    if is_main_process(args):
        for class_idx, class_iou in enumerate(per_class_iu):
            print(f'class_{class_idx}: {class_iou}')
        print(f'mIoU: {meanIoU}')
    return meanIoU, per_class_iu


if __name__ == '__main__':
    start = timeit.default_timer()
    args = parse_args()

    dataset_key = args.dataset.lower()

    if dataset_key == 'cityscapes':
        args.classes = 19
        args.input_size = '512,1024'
        ignore_label = 255
    elif dataset_key == 'camvid':
        args.classes = 11
        args.input_size = '360,480'
        ignore_label = 11
    elif dataset_key == 'udd6':
        args.classes = 6
        args.input_size = '1024,1024'
        ignore_label = 255
    else:
        raise NotImplementedError(
            "This repository now supports cityscapes, camvid, and udd6, %s is not included" % args.dataset)

    args.ignore_label = ignore_label
    train_model(args)
    end = timeit.default_timer()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    print("training time: %d hour %d minutes" % (int(hour), int(minute)))
