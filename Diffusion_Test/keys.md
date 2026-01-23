# 项目名称：Wavelet-Guided Diffusion for Neonatal MRI Enhancement (WGD-MRI)

## 1. 数据预处理与加载方案 (Data Pipeline)

鉴于0.35T图像的特殊噪声分布和物理分辨率差异，预处理是能否成功的关键。

### 1.1 0.35T 图像特异性预处理
*   这一部分已经完成了。
*   检查1.5T与0.35T图像并完成归一化操作即可。

### 1.2 数据加载机制 (In-Memory Caching)
*   **策略**：全量内存锁存（RAM Caching）。
*   **原因**：MRI 切片数据通常较小（单张 352x352 float32 仅约 0.5MB），全量加载可消除 IO 瓶颈，最大化 GPU 利用率。
*   **数据增强 (Augmentation)**：仅在训练时在线进行。
    *   `RandomFlip` (水平翻转)：概率 0.5。
    *   `RandomRotate` (随机旋转)：$\pm 10$ 度（模拟不同扫描体位）。
    *   **禁止**使用 ColorJitter 或强烈的各类噪声注入（会破坏 0.35T 本身的噪声特征）。

---

## 2. 网络架构设计：Wavelet-Diffusion U-Net

这是本框架的核心。我们将标准 U-Net 的上下采样层替换为小波变换，实现**无损特征缩放**。

### 2.1 全局参数配置
*   **输入分辨率**：$352 \times 352 \times 1$ (单通道)
*   **小波基 (Wavelet Basis)**：`bior1.3` (Biorthogonal 1.3)
    *   *选择理由*：Biorthogonal 小波比 Haar 更平滑，适合生物组织成像；1.3 阶数较低，计算量小且振铃效应（Ringing Artifacts）极低。
*   **填充模式**：`symmetric` (对称填充，防止边缘伪影)。
*   **主要激活函数**：`SiLU` (Swish)。
*   **特征通道基数 (Base Channels)**：64。

### 2.2 详细层级结构 (Layer-wise Specification)

网络结构呈 U 型，包含 4 次下采样和 4 次上采样。

#### **A. 编码器 (Encoder) - 下采样路径**

| 层级 (Stage) | 输入尺寸 (H, W) | 操作模块 (Operations)                            | 输出通道 (Out Channels) | 设计意图                                                 |
| :----------- | :-------------- | :----------------------------------------------- | :---------------------- | :------------------------------------------------------- |
| **Input**    | 352 x 352       | Conv3x3 (Initial)                                | 64                      | 初始特征提取                                             |
| **Level 1**  | 352 x 352       | **DWT-Block** $\to$ ResBlock $\times 2$          | 128                     | **DWT**将H,W减半，通道 $\times 4$。接1x1 Conv降维到128。 |
| **Level 2**  | 176 x 176       | **DWT-Block** $\to$ ResBlock $\times 2$ $\to$ SA | 256                     | 加入 **Self-Attention (SA)** 捕捉全局解剖关联。          |
| **Level 3**  | 88 x 88         | **DWT-Block** $\to$ ResBlock $\times 2$ $\to$ SA | 512                     | 特征图变小，增加通道数提取深层语义。                     |
| **Level 4**  | 44 x 44         | **DWT-Block** $\to$ ResBlock $\times 2$ $\to$ SA | 512                     | 显存控制，通道数不再翻倍。                               |

*   **DWT-Block 内部逻辑**：
    1.  输入 $C_{in} \times H \times W$。
    2.  执行 2D-DWT (`bior1.3`) $\rightarrow$ 输出 $4C_{in} \times \frac{H}{2} \times \frac{W}{2}$ (LL, LH, HL, HH)。
    3.  $1 \times 1$ 卷积：将 $4C_{in}$ 压缩至 $C_{out}$。
    4.  GroupNorm + SiLU。

#### **B. 瓶颈层 (Bottleneck)**
*   **尺寸**：$22 \times 22$ (经过 Level 4 下采样后)。
*   **操作**：ResBlock $\times 2$ $\to$ Multi-Head Self-Attention $\to$ ResBlock $\times 1$。
*   **通道**：512。
*   **作用**：最抽象的语义层，负责理解“这是大脑，这里是脑室”。

#### **C. 解码器 (Decoder) - 上采样路径**

| 层级 (Stage) | 输入尺寸 (H, W) | 操作模块 (Operations)                                        | 输出通道 (Out Channels) | 设计意图                          |
| :----------- | :-------------- | :----------------------------------------------------------- | :---------------------- | :-------------------------------- |
| **Level 4**  | 22 x 22         | **IDWT-Block** $\to$ Concat(Skip) $\to$ ResBlock $\times 2$ $\to$ SA | 512                     | 恢复空间分辨率，融合编码器特征。  |
| **Level 3**  | 44 x 44         | **IDWT-Block** $\to$ Concat(Skip) $\to$ ResBlock $\times 2$ $\to$ SA | 256                     |                                   |
| **Level 2**  | 88 x 88         | **IDWT-Block** $\to$ Concat(Skip) $\to$ ResBlock $\times 2$ $\to$ SA | 128                     |                                   |
| **Level 1**  | 176 x 176       | **IDWT-Block** $\to$ Concat(Skip) $\to$ ResBlock $\times 2$  | 64                      | 最后一层不加Attention以节省显存。 |
| **Output**   | 352 x 352       | Conv3x3 (Final)                                              | 1                       | 输出预测的噪声或图像。            |

*   **IDWT-Block 内部逻辑**：
    1.  输入 $C_{in} \times H \times W$。
    2.  $1 \times 1$ 卷积：将 $C_{in}$ 扩展至 $4C_{out}$ (预测 LL, LH, HL, HH 分量)。
    3.  执行 2D-IDWT (`bior1.3`) $\rightarrow$ 输出 $C_{out} \times 2H \times 2W$。

---

## 3. 训练策略与损失函数 (Training & Loss)

由于是非配对数据，我们采用 **"Unpaired Conditional Diffusion"** 策略。我们训练一个模型，学习从“0.35T引导的噪声”恢复到“1.5T域图像”。

### 3.1 训练流程
*   **模型角色**：噪声预测网络 $\epsilon_\theta(x_t, t, c)$，其中 $c$ 是 0.35T 原图。
*   **输入**：
    *   $x_0^{1.5T}$：真实的 1.5T 图像（目标域样本）。
    *   $x_t$：加噪后的图像（标准扩散过程）。
    *   $c = x^{0.35T}$：**作为条件输入**的 0.35T 图像（源域样本，**非配对**）。
    *   *注意*：由于是非配对，我们实际上是在训练网络**学会1.5T的分布**，同时利用**PatchNCE**强制网络在生成时保留$c$的内容。

### 3.2 复合损失函数 (Total Loss)

$$ L_{total} = L_{diff} + \lambda_{NCE} \cdot L_{NCE} + \lambda_{wave} \cdot L_{wave} $$

#### **1. 扩散去噪损失 ($L_{diff}$)**
标准 DDPM/DDIM 损失，仅在 1.5T 数据上计算：
$$ L_{diff} = \mathbb{E}_{x_0 \sim 1.5T, \epsilon \sim N(0,1), t} [ || \epsilon - \epsilon_\theta(x_t, t) ||_2^2 ] $$
*   *目的*：让网络学会生成高质量、无噪声的 1.5T 风格图像。

#### **2. 结构一致性损失 ($L_{NCE}$ - PatchNCE)**
这是解决“解剖结构不一致”的关键。我们在生成的图像 $\hat{x}_0$（由模型预测得出）和输入的 0.35T 图像 $c$ 之间计算。
*   **操作**：
    *   提取 U-Net 编码器第 4 层、第 8 层、第 12 层的特征图。
    *   随机选取 256 个 Patch。
    *   计算对比损失：要求 $\hat{x}_0$ 的 Patch 与 $c$ 对应位置的 Patch 相似度最大，与其他位置的 Patch 相似度最小。
*   *目的*：不管图像风格怎么变，脑沟脑回的位置关系必须对应。

#### **3. 小波解剖约束损失 ($L_{wave}$)**
这是解决“0.35T信息利用不足”的关键。
$$ L_{wave} = || \text{LL}(\hat{x}_0) - \text{LL}(c) ||_1 $$
*   **操作**：分别对预测图像 $\hat{x}_0$ 和输入图像 $c$ 进行小波变换，提取 **LL (低频)** 分量，计算 L1 距离。
*   **重要细节**：**只约束 LL 分量**。
    *   *为什么？* 因为 HH/HL/LH 包含 0.35T 的噪声，我们希望抛弃它们；而 LL 包含解剖结构，我们希望保留它们。
*   *目的*：物理层面上强制锁定解剖轮廓。

### 3.3 超参数设置
*   $\lambda_{NCE} = 1.0$ (结构权重)
*   $\lambda_{wave} = 10.0$ (低频刚性约束权重，给予较高权重以防止形变)
*   **Optimizer**: AdamW, $\beta_1=0.9, \beta_2=0.999$, Weight Decay $1e-4$.
*   **Learning Rate**: $1e-4$，采用 Cosine Annealing 调度。
*   **Batch Size**: 建议 4 或 8 (取决于显存)。
*   **显存预估**: 352分辨率 + Wavelet U-Net + Gradient Checkpointing $\approx$ 16GB - 24GB VRAM。

---

## 4. 推理/增强阶段 (Inference)

训练完成后，使用以下步骤增强 0.35T 图像：

1.  **输入**：一张 0.35T 图像 $y$。
2.  **编码**：对 $y$ 加噪到 $T/2$ 步 (例如 $T=1000$, 取 $t=500$)，得到 $y_{500}$。
    *   *理由*：不要从纯噪声开始生成，而是利用 SDEdit 的思想，保留部分原图信息。
3.  **反向扩散**：使用训练好的网络，从 $t=500$ 逐步去噪到 $t=0$。
    *   在每一步去噪预测 $\hat{x}_{0}$ 时，可以使用 **流形约束 (Manifold Constraint)**：
    $$ \hat{x}_{t-1} = \text{Sampler}(\hat{x}_t, \epsilon_\theta) $$
    $$ \text{Correction}: \text{LL}(\hat{x}_{t-1}) = \alpha \cdot \text{LL}(\hat{x}_{t-1}) + (1-\alpha) \cdot \text{LL}(y) $$
    *   用原图的低频分量，对每一步生成的低频分量进行“纠偏”（$\alpha \approx 0.8$），进一步确保解剖结构不跑偏。

---

## 5.其他的要求

在训练过程中：

- 每间隔一定epoch打印当前进度与ETA，同时显示系统时间

- 每间隔一定epoch保存当前模型，完成训练也要保存。不过这一步保存的模型不需要很多，一共能有10个模型即可。

编写train和test文件，允许使用不同的模型测试结果。
