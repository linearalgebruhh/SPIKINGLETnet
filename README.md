# SPIKINGLET implementation detail

## 设备
- GPU: NVIDIA GeForce RTX 3090

## 编译平台
- UBUNTU 20.4


## 算法环境
- conda环境：
- python 3.10
- requirements: `requirements.txt`

## 基础结构
- SPIKINGLET 网络主体：`model/SpikingLET.py`
- SNN神经元：`model/SNNNeurons`
- 数据集：`dataset/UDD6/`(数据集放在这里)

## 训练设置
- init_lr: 0.0006
- batch_size: 4
- input_size: 1024x1024

## 训练命令


```
nohup torchrun \
  --nproc_per_node=2 \
  --master_port=29504 \
  train.py \
  --gpus 0,1 \
  --model SpikingLET \
  --dataset udd6 \
  --input_size 512,512 \
  --train_crop_size 512,512 \
  --train_stride 256,256 \
  --val_crop_size 512,512 \
  --val_stride 256,256 \
  --train_mode slidingwindow \
  --val_mode slidingwindow \
  --num_workers 4 \
  --classes 6 \
  --train_type train \
  --max_epochs 200 \
  --random_mirror True \
  --random_scale True \
  --lr 0.0006 \
  --batch_size 3 \
  --optim adamw \
  --lr_schedule poly \
  --num_cycles 1 \
  --poly_exp 0.9 \
  --warmup_iters 500 \
  --warmup_factor 0.3333333333333333 \
  --T 8 \
  --cfg_n 2 \
  --gamma 1.0 \
  > train_snn_msL4.log 2>&1 &

```
for edgerazor:
```
nohup torchrun --nproc_per_node=2 train_edgerazor.py   --model SpikingLET --dataset udd6   --T 8 --thresh 1.0 --tau 0.5 --gamma 2.0 --neuron_mode bptt --cfg_n 2   --num_workers 4 --train_type train --max_epochs 200   --repeat_times 5 --lr 0.0006 --batch_size 3 --optim adamw   --lr_schedule poly --poly_exp 0.9   --train_mode slidingwindow --train_crop_size 512,512 --train_stride 256,256   --val_mode slidingwindow --val_crop_size 1024,1024 --val_stride 768,768   --gpus 0,1,2   --edgerazor_config configs/edgerazor/qat_w4_a8_kld.yaml   --student_pretrained_path checkpoint/udd6/SpikingLETbs3gpu8cfg2_traina/model_42.pth   --teacher_pretrained_path checkpoint/udd6/SpikingLETbs3gpu8cfg2_traina/model_42.pth   > train_edgerazor_qat.log 2>&1 &
```

## ckpt
```
通过网盘分享的文件：ckpt
链接: https://pan.baidu.com/s/1E-Vz2B4BtXqr6vouV7Ygag 提取码: cypb 
--来自百度网盘超级会员v3的分享
```