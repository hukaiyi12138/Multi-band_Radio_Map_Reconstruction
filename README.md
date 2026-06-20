# 改动
把CNN改成UNet网络，避开CNN局限的感受野，使用UNet扩充至全局视野
目前在fiber1条件下效果超越RadioUNet，并且fiber整体除了fiber10都有提升
但问题是mask所有比CNN效果差，从可视化图来看是UNet把采样点做成了噪点
目前UNet + mu_grad=0.15效果不好，现在尝试去掉mu_grad=0.15，改成mu_grad=0，只用L1损失

跑完 mu_grad=0.0 后：
    如果可视化图里噪点没了 → 梯度损失就是元凶，论文用 mu_grad=0.0 重跑
    如果可视化图里噪点还在，只是淡了一些 → U-Net 本身也是问题，需要更深入调整
    如果可视化图里噪点还是很明显 → 必须回退到 CNN

# Train
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 15 \
    --mask-type mask \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/mask15


CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 10 \
    --mask-type mask \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/mask10


CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 5 \
    --mask-type mask \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/mask5

 
CUDA_VISIBLE_DEVICES=1 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type mask \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/mask1

 
CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 15 \
    --mask-type fiber \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/fiber15

 
CUDA_VISIBLE_DEVICES=2 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 10 \
    --mask-type fiber \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/fiber10

 
CUDA_VISIBLE_DEVICES=4 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 5 \
    --mask-type fiber \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/fiber5

 
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type fiber \
    --N-iter 5 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/fiber1

######
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type fiber \
    --N-iter 10 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/fiber1_iter10

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type mask \
    --N-iter 10 \
    --epochs 50 \
    --batch-size 1 \
    --lr 1e-3 \
    --save-dir runs/mask1_iter10
######


# Test
CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/mask15/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 15 \
    --mask-type mask \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/mask10/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 10 \
    --mask-type mask \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/mask5/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 5 \
    --mask-type mask \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/mask1/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type mask \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/fiber15/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 15 \
    --mask-type fiber \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/fiber10/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 10 \
    --mask-type fiber \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/fiber5/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 5 \
    --mask-type fiber \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

CUDA_VISIBLE_DEVICES=0 python test.py \
    --checkpoint runs/fiber1/best.pt \
    --dataset dulrtc_triple \
    --root /data/home/hky/dataset/DULRTC_triple \
    --omega-num 1 \
    --mask-type fiber \
    --batch-size 1 \
    --output-path /data/home/hky/DULRTC/hky_try_3/test

# Evaluation
cd /data/home/hky/DULRTC/hky_try_3

python evaluation.py --root runs


## 不同 Iter 测试
CUDA_VISIBLE_DEVICES=0 python test.py \
  --checkpoint runs/mask15_Niter10/best.pt \
  --dataset dulrtc_triple \
  --root /data/home/hky/dataset/DULRTC_triple \
  --omega-num 15 \
  --mask-type mask \
  --R 3 \
  --K 3 \
  --N-iter 10 \
  --batch-size 1 \
  --num-workers 2 \
  --output-path runs/test \
  --max-save-figures 10

CUDA_VISIBLE_DEVICES=0 python test.py \
  --checkpoint runs/mask15_Niter10/best.pt \
  --dataset dulrtc_triple \
  --root /data/home/hky/dataset/DULRTC_triple \
  --omega-num 15 \
  --mask-type mask \
  --R 3 \
  --K 3 \
  --N-iter 10 \
  --batch-size 1 \
  --num-workers 2 \
  --output-path runs/test \
  --max-save-figures 10