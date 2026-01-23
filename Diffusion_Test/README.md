# WGD-MRI 深度学习项目

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 数据准备

将 0.35T 与 1.5T 图像分别放到如下目录，支持 NIfTI (.nii/.nii.gz)，会按轴向切片作为样本：

- data/0_35T
- data/1_5T

## 3. 训练

```bash
python src/train.py --data-035t data/0_35T --data-15t data/1_5T --output-dir outputs
```

可选参数含义：

- --data-035t：0.35T 条件图像目录
- --data-15t：1.5T 目标图像目录
- --output-dir：训练输出目录
- --batch-size：训练 batch 大小
- --num-epochs：训练 epoch 数
- --learning-rate：学习率
- --weight-decay：权重衰减
- --num-workers：数据加载线程数
- --image-size：输入图像边长
- --timesteps：扩散总步数
- --save-every：模型保存间隔 epoch
- --keep-checkpoints：最多保留的模型数量
- --lambda-nce：结构一致性损失权重
- --lambda-wave：小波低频约束权重
- --patch-nce-patches：PatchNCE 采样 patch 数
- --seed：随机种子

## 4. 测试/增强

```bash
python src/test.py --input-dir data/0_35T --checkpoint-path outputs/checkpoints/latest.pt --output-dir outputs/inference
```

输出结果为拼接对比图，左侧为原图，右侧为增强图，保存为 PNG/JPG。

可选参数含义：

- --input-dir：待增强 0.35T 图像目录
- --output-dir：输出目录
- --checkpoint-path：模型权重路径
- --timesteps：扩散总步数
- --start-step：SDEdit 起始步
- --manifold-alpha：低频流形约束系数
- --output-ext：输出图片格式（png 或 jpg）
- --batch-size：推理 batch 大小
- --num-workers：数据加载线程数
- --image-size：输入图像边长

## 5. 模型导出

```bash
python src/export.py --checkpoint-path outputs/checkpoints/latest.pt --output-path outputs/exported_model.pt
```

可选参数含义：

- --checkpoint-path：模型权重路径
- --output-path：导出模型保存路径
- --image-size：输入图像边长

## 6. 说明

- 训练使用非配对数据，通过 PatchNCE 与小波低频约束增强结构一致性。
- 训练中每个 epoch 输出耗时、ETA 与系统时间，并按配置保存模型，仅保留最近 10 个模型。
- 推理采用 SDEdit 风格起始步与小波低频流形约束。
