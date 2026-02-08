# 第3章 线性回归 — 学习笔记

**参考 Notebooks：**
- [linear-regression.ipynb](linear-regression.ipynb) — 线性回归
- [oo-design.ipynb](oo-design.ipynb) — 面向对象的实现设计
- [synthetic-regression-data.ipynb](synthetic-regression-data.ipynb) — 合成回归数据
- [linear-regression-scratch.ipynb](linear-regression-scratch.ipynb) — 线性回归从零实现
- [linear-regression-concise.ipynb](linear-regression-concise.ipynb) — 线性回归简洁实现
- [generalization.ipynb](generalization.ipynb) — 泛化
- [weight-decay.ipynb](weight-decay.ipynb) — 权重衰减

---
---

# 第一部分：原理直觉篇

---

## 一、线性回归 <sub>([linear-regression.ipynb](linear-regression.ipynb))</sub>

### 1.1 模型在做什么

假设你是一个房产中介，工作了很多年，积累了大量成交记录——每一条记录包含房子的特征（面积、房龄、楼层等）以及最终的成交价格。现在有个新客户来问："我这套 80 平、10 年房龄的房子值多少？"

你会怎么估价？凭经验，你大概知道：面积越大，价格越高；房龄越老，价格越低。更精确地说，你脑子里有一个"模型"：**每个特征对价格有各自的影响力，把这些影响加起来，再加上一个基准价，就是估价。**

这就是线性回归在做的事情。它的核心假设是：**结果可以被写成各个特征的加权总和加上一个常数。** 这里的"权重"表示每个特征有多重要，"常数"（也叫偏置）表示所有特征都为零时的基础值。

为什么叫"线性"？因为预测值和每个特征之间的关系是一条直线——一个特征增大一点，预测值就增大（或减小）固定的一点，不会出现"翻倍"或"指数增长"的效果。这是一个非常强的简化假设，但在很多场景下出奇地好用。

线性回归属于**回归**任务——输出是一个连续数值（比如价格、温度、股价），而不是一个类别标签。

### 1.2 怎么衡量预测的好坏：损失函数

有了模型，怎么知道它预测得准不准？需要一个"打分标准"来衡量预测值和真实值之间的差距——这就是**损失函数**。

最直觉的想法：预测值减去真实值，差得越多越差。但直接相减有个问题——正偏差和负偏差会互相抵消（高估 10 万和低估 10 万加起来误差为 0，但模型显然不好）。解决办法是**把差值平方**：不管高估还是低估，平方后都是正数，而且差得越大，平方后的惩罚越重。

这就是**均方误差**（MSE）：对每个样本算出"预测值减真实值的平方"，然后对所有样本取平均。训练的目标就是找到一组权重，让这个平均平方误差尽可能小。

为什么选平方，而不是绝对值或者四次方？

- 绝对值在零点不可导，梯度计算会有麻烦
- 四次方对大误差的惩罚过于极端
- 平方误差有良好的数学性质（对应高斯噪声假设，后面会解释）
- 而且平方误差让优化问题变成一个"碗状"曲面，只有一个最低点，不用担心找到假的最优解

### 1.3 一步到位的解法：解析解

线性回归有一个特殊的福利：因为模型是线性的、损失是平方的，所以我们可以直接用数学公式一步算出最优权重，而不需要一步一步地迭代。

想象损失函数是一个碗的形状。碗的最低点就是最优解。对于碗来说，最低点的特征是"斜率为零"——无论往哪个方向走，高度都不再下降。数学上，这意味着"损失对每个权重的导数都等于零"。因为模型是线性的，这组方程可以直接解出来。

这就是解析解。它一步到位，不需要迭代，结果也是精确的。

但为什么深度学习不用解析解？因为这个"一步算出"的前提是模型足够简单。稍微复杂一点的模型（加一个非线性激活函数），那组方程就解不出来了。所以解析解是线性回归的"特权"，深度学习中几乎用不到。

### 1.4 走一步看一步的解法：梯度下降

既然复杂模型没有解析解，我们需要一个通用的方法：**梯度下降**。

核心直觉：想象你站在一座山上，眼睛被蒙住了，想走到谷底。你能做的唯一事情是用脚试探四周的坡度——哪个方向最陡就往哪里走。每一步都往最陡的下坡方向迈出一小步。这就是梯度下降。

**梯度**是什么？它是一个方向，告诉你"损失增长最快的方向"。既然我们想让损失减小，就往梯度的反方向走——损失下降最快的方向。

**学习率**是什么？是你每一步迈多大。步子太大，可能会从山的一边直接跨到另一边，在谷底附近来回震荡，永远到不了底部。步子太小，虽然一定能到谷底，但会走得极慢。学习率是深度学习中最重要的超参数之一。

但如果每走一步都要看完所有训练数据（计算全部样本的梯度），那数据量大的时候会非常慢。解决方案是**小批量随机梯度下降**（Mini-batch SGD）：

1. 每次随机抽一小批样本（比如 256 个）
2. 只用这一小批样本估算梯度
3. 按照这个近似梯度更新参数
4. 重复

为什么这样可行？因为随机抽样的平均值是全体数据的无偏估计——虽然每一步的方向不太准，但平均下来是朝着正确方向走的。而且每一步的计算量小了很多，总体速度反而更快。

**超参数总结：**

| 超参数 | 含义 | 太大了会怎样 | 太小了会怎样 |
|--------|------|-------------|-------------|
| 学习率 | 每步走多远 | 震荡不收敛 | 收敛太慢 |
| 批量大小 | 每次看多少样本 | 更新太慢（但梯度更准） | 梯度噪音太大 |
| 训练轮数 | 整个数据集看几遍 | 过拟合 | 还没学到规律 |

### 1.5 为什么选均方误差：概率的视角

前面说"损失函数选平方误差"似乎是拍脑袋决定的，但其实背后有严格的概率论支撑。

思路是这样的：真实世界的数据不完美。即使模型完全正确（知道真正的权重），观测到的数据也有噪声——测量误差、遗漏特征的影响等等。最常见的假设是这些噪声服从**正态分布**（也叫高斯分布）：大部分噪声很小，偶尔有较大的噪声，正负噪声出现的概率相同。

在这个假设下，"找到最好的权重"等价于"找到让观测数据出现概率最大的权重"——这就是**最大似然估计**。经过推导会发现，最大化似然函数和最小化均方误差是完全等价的。

所以 MSE 不是随便选的——它是"假设噪声服从正态分布"这个前提下的最优选择。如果噪声不是正态分布（比如有很多极端异常值），MSE 就不一定是最好的损失函数了。

### 1.6 矢量化：为什么不用循环

如果有 1000 个样本、100 个特征，最自然的写法是用两层循环：外层遍历样本，内层遍历特征做加权求和。但这样极其缓慢。

**矢量化**的意思是：把循环去掉，改成矩阵运算。一个矩阵乘法就能同时算出所有样本的所有预测值。速度差距可以达到几百倍——因为矩阵运算可以利用 CPU 的 SIMD 指令集和 GPU 的大规模并行能力，而循环只能一个一个地串行计算。

这是深度学习中的铁律：**能用矩阵运算就绝不用循环。**

---

### 场景走查：从头到尾走一遍房价预测

假设我们有一个简单的房价预测任务：根据面积和房龄预测房价。我们有 1000 套房子的历史数据。

**第一步：准备数据。** 收集每套房子的面积（比如 60-150 平米）和房龄（比如 0-30 年），以及对应的成交价。把数据分成 800 条训练、200 条验证。

**第二步：初始化模型。** 随机给面积的权重、房龄的权重和偏置各赋一个很小的初始值。此时模型的预测完全是瞎猜。

**第三步：开始训练。** 从 800 条训练数据中随机取出 32 条（一个 mini-batch）。用当前的权重和偏置计算这 32 套房子的预测价格。计算预测值和真实价格的均方误差。

**第四步：计算梯度。** 损失对每个参数求偏导数。面积权重的梯度告诉我们"面积权重应该往上调还是往下调、调多少"。房龄权重和偏置同理。

**第五步：更新参数。** 按照梯度的反方向，乘以学习率，更新三个参数。现在模型的预测应该比刚才准了一点点。

**第六步：重复。** 再取下一个 mini-batch，重复第三到第五步。当所有 800 条数据都看完一遍（一个 epoch），在 200 条验证数据上评估模型的表现。如果验证误差还在下降，就继续下一个 epoch。

**训练结束后：** 模型学到的权重大约反映了"面积每增加一平米，房价增加多少"和"房龄每增加一年，房价下降多少"。偏置反映了"面积和房龄都为零时的基础价"。

<img src="img/linear_regression_flow.png" alt="线性回归流程" style="max-width:640px; max-height:600px;">

---

## 二、泛化 <sub>([generalization.ipynb](generalization.ipynb))</sub>

### 2.1 训练误差与泛化误差

在训练集上表现好，不代表在新数据上也表现好。

想象一个学生准备考试：他把往年真题全背下来了，每一道都能做对。但考试出了一道新题，他完全不会。这就是**过拟合**——记住了训练数据的所有细节（包括噪声），但没有学到真正的规律。

**训练误差**是模型在训练数据上的表现，**泛化误差**是在从未见过的新数据上的表现。我们真正关心的是泛化误差，但我们只能直接观察到训练误差。两者之间的差距叫**泛化差距**。

关键假设：训练数据和测试数据来自同一个分布（叫**独立同分布**假设，简称 IID）。如果这个假设不成立（比如用夏天的数据预测冬天），那训练误差再低也没意义。

### 2.2 模型复杂度与容量

模型越复杂，越容易过拟合。但模型太简单，又可能**欠拟合**——连训练数据中的基本规律都学不到。

影响模型复杂度的因素：

- **参数数量**：参数越多，模型能表达的函数越复杂
- **参数取值范围**：允许参数取很大的值，模型更灵活但也更容易过拟合
- **数据维度**：特征越多，需要更多数据才能学好
- **数据量**：数据越多，泛化越好（几乎总是如此）

一个经典的例子是多项式拟合：用一条直线拟合数据可能太简单（欠拟合），用一个高次多项式可以完美穿过每个数据点但在数据点之间疯狂波动（过拟合），用一个恰当次数的多项式才能学到真正的趋势。

### 2.3 欠拟合与过拟合

| | 欠拟合 | 过拟合 |
|---|---|---|
| 训练误差 | 高 | 低（甚至接近零） |
| 验证误差 | 高 | 高 |
| 泛化差距 | 小 | 大 |
| 原因 | 模型太简单 | 模型太复杂或数据太少 |
| 解决方案 | 增大模型、增加特征 | 增加数据、正则化、减小模型 |

一个有意思的现象：深度学习中的模型往往参数量远超训练样本数（理论上严重过拟合），但实际泛化效果却很好。这是现代深度学习理论中尚未完全解释清楚的谜题之一。

### 2.4 模型选择与交叉验证

怎么选择"最好的模型"（比如学习率、层数、正则化强度等超参数）？

**三路拆分法**：把数据分成训练集、验证集和测试集三部分。训练集用来训练模型参数，验证集用来选超参数和模型架构，测试集只在最后用一次来报告最终性能。

为什么不能用测试集来选模型？因为一旦你根据测试集的表现调整了模型，测试集就不再是"未见过的新数据"了——测试误差就不再是泛化误差的无偏估计。

**交叉验证**：当数据量太少时，拿出一部分做验证集太奢侈。解决方案是**K 折交叉验证**：把数据分成 K 份，轮流用其中一份做验证，其余做训练，重复 K 次取平均。这样每个数据点都被用于训练和验证，充分利用了有限的数据。

---

## 三、权重衰减 <sub>([weight-decay.ipynb](weight-decay.ipynb))</sub>

### 3.1 为什么需要正则化

上一节说模型太复杂会过拟合。一个直接的想法是减少参数数量，但对深度学习来说这往往不太可行——网络结构是固定的，不方便随便删参数。

另一个思路是：**不减少参数数量，而是限制参数的大小。** 直觉上，如果权重都很小、接近零，那模型就接近一个常数函数（最简单的函数）。权重越大，模型越"剧烈"——对输入的微小变化产生过大的反应。所以限制权重大小是控制模型复杂度的一种"软"方式。

### 3.2 L2 正则化的直觉

具体做法是在原来的损失函数后面加一个"惩罚项"：**所有权重的平方和**。训练时，模型不仅要让预测准确（原来的损失小），还要让权重不要太大（惩罚项小）。这就迫使模型在"拟合好"和"保持简单"之间做权衡。

惩罚项前面有一个系数，用来控制"限制权重"这件事有多重要。系数为零时，等于没有正则化；系数很大时，权重被压得很小，模型接近常数函数。

为什么只惩罚权重，不惩罚偏置？偏置只是整体平移预测值，不影响模型对输入变化的敏感程度。限制偏置没有什么好处。

### 3.3 权重衰减的效果

加了正则化后，梯度更新的公式会多出一项：在每一步更新时，权重不仅要沿着梯度方向调整，还要被稍微"收缩"一点向零靠拢。这就是"权重衰减"这个名字的来历——每一步都让权重衰减一点点。

效果：

- **训练误差会变高**——因为模型被限制了，不能完美拟合训练数据了
- **验证误差会变低**——因为模型变"规矩"了，不再记住训练数据的噪声
- **权重的整体大小会变小**

这就是正则化的权衡：牺牲训练集上的表现，换取在新数据上的表现。

### L2 正则化 vs L1 正则化

| | L2 正则化（权重衰减） | L1 正则化 |
|---|---|---|
| 惩罚的是什么 | 权重的平方和 | 权重的绝对值之和 |
| 效果 | 让所有权重都变小 | 让很多权重直接变成零（稀疏） |
| 适用场景 | 一般场景，深度学习默认选择 | 特征选择（自动去掉不重要的特征） |
| 几何直觉 | 把权重限制在一个"球"里 | 把权重限制在一个"菱形"里 |

---

### 场景走查：权重衰减如何拯救过拟合

假设一个极端情况：200 个特征，但只有 20 个训练样本。不加正则化时：

- 模型有足够的自由度"背下"每一个训练样本
- 训练误差几乎降到零
- 但在验证集上的误差很高——因为学到的是噪声，不是规律

加上权重衰减后：

- 模型被限制不能给任何特征分配过大的权重
- 训练误差比之前高了一些——但这正是我们想要的
- 验证集上的误差大幅下降——模型终于学到了真正的规律
- 权重值变小了很多，那些不重要的特征的权重接近零

**下一章预告：** 这一章我们学了线性回归这个最简单的预测模型。下一章，我们将把同样的思路用到分类任务上——不是预测一个数字，而是预测一张图片属于哪个类别。这就引出了 Softmax 回归。

---
---

# 第二部分：数学推导篇

---

## 一、线性回归 <sub>([linear-regression.ipynb](linear-regression.ipynb))</sub>

### 1.1 模型

给定 $d$ 个特征，线性回归的预测为：

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b \tag{1.1}$$

> $x_1, x_2, \ldots, x_d$ 是输入特征（如面积、房龄），$w_1, w_2, \ldots, w_d$ 是对应的权重（每个特征的重要程度），$b$ 是偏置（基准值）。公式的含义：把每个特征乘以其权重，全部加起来，再加上偏置，得到预测值 $\hat{y}$。

对 $n$ 个样本的小批量矩阵形式：

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b \tag{1.2}$$

> $\mathbf{X}$ 是 $n \times d$ 的矩阵（每行是一个样本，每列是一个特征），$\mathbf{w}$ 是 $d \times 1$ 的权重向量。一次矩阵乘法就把所有样本的预测值算出来了，不需要循环。

这是一个**仿射变换** = 线性变换（乘权重）+ 平移（加偏置）。

### 1.2 损失函数：均方误差

单个样本的损失：

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2}\left(\hat{y}^{(i)} - y^{(i)}\right)^2 \tag{1.3}$$

> $\hat{y}^{(i)}$ 是第 $i$ 个样本的预测值，$y^{(i)}$ 是真实值。两者之差的平方衡量"预测离真实值有多远"。前面的 $\frac{1}{2}$ 纯粹为了求导方便——导数会出来一个 2，乘以 $\frac{1}{2}$ 刚好消掉。

整个数据集上的平均损失：

$$L(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^{n} l^{(i)} = \frac{1}{n}\sum_{i=1}^{n}\frac{1}{2}\left(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\right)^2 \tag{1.4}$$

> 对所有 $n$ 个样本的损失求平均。这就是我们要最小化的目标函数。

目标：

$$\mathbf{w}^*, b^* = \arg\min_{\mathbf{w},b} L(\mathbf{w}, b) \tag{1.5}$$

> $\arg\min$ 表示"使 $L$ 取最小值的那组 $\mathbf{w}$ 和 $b$"。

### 1.3 解析解

线性回归的特殊福利——令损失对参数的导数等于零，直接解方程得到闭式解：

$$\mathbf{w}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} \tag{1.6}$$

> $\mathbf{X}^\top\mathbf{X}$ 是 $d \times d$ 的方阵，$(\mathbf{X}^\top\mathbf{X})^{-1}$ 是其逆矩阵。要求 $\mathbf{X}^\top\mathbf{X}$ 可逆（即特征之间线性无关）。这个公式一步到位算出最优权重，不需要迭代。但只有线性回归能享受这个福利——复杂模型不存在闭式解。

### 1.4 小批量随机梯度下降 (Mini-batch SGD)

#### 前置概念：导数、偏导数与梯度

**导数（单变量函数）**

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} \tag{1.7}$$

> $h$ 是无穷小的增量。导数描述的是：$x$ 变化一点点时，$f$ 变化多少。几何上就是曲线在该点的切线斜率。例如 $f(x) = x^2$，则 $f'(x) = 2x$，在 $x=3$ 处斜率为 6。

**偏导数（多变量函数）**

$$\frac{\partial L}{\partial w_j} = \lim_{h \to 0} \frac{L(\ldots, w_j + h, \ldots) - L(\ldots, w_j, \ldots)}{h} \tag{1.8}$$

> 固定其他所有参数不动，只看 $L$ 随 $w_j$ 的变化速率。符号 $\partial$ 区别于单变量的 $d$，提醒你"还有其他变量，但我暂时不管它们"。

**梯度 = 所有偏导数打包成向量**

$$\nabla_{\mathbf{w}} L = \left(\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \ldots, \frac{\partial L}{\partial w_d}\right) \tag{1.9}$$

> 梯度是一个向量，每个分量告诉你"沿这个参数方向，损失变化多快"。**梯度指向损失增长最快的方向**，所以往**梯度的反方向**走一步，损失就会下降。

**自动微分 vs 数值微分：**

| 方式 | 怎么做 | 精确？ | 速度 | 实际用途 |
|---|---|---|---|---|
| 数值微分 | 取很小的 $h$，算 $\frac{f(x+h)-f(x)}{h}$ | 近似 | 极慢（每个参数算一次前向） | 仅用于 gradient check |
| 自动微分 (autograd) | 记录计算图 + 逐步套解析公式 + 链式法则 | 精确 | 快（一次反向传播算所有参数） | **实际训练用这个** |

---

#### SGD 更新公式

每次随机抽一小批样本 $\mathcal{B}$，按梯度反方向更新：

$$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \frac{\eta}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w}, b) \tag{1.10}$$

> $\mathcal{B}$ 是随机抽出的一小批样本，$|\mathcal{B}|$ 是批量大小，$\eta$ 是学习率。对小批量中的每个样本算梯度，取平均，然后让参数往梯度反方向走 $\eta$ 这么远。

展开后：

$$\mathbf{w} \leftarrow \mathbf{w} - \frac{\eta}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \mathbf{x}^{(i)}\left(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\right) \tag{1.11}$$

$$b \leftarrow b - \frac{\eta}{|\mathcal{B}|}\sum_{i \in \mathcal{B}}\left(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\right) \tag{1.12}$$

> 括号里的 $\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}$ 就是"预测值 - 真实值"（残差）。对 $\mathbf{w}$ 求导多乘了一个 $\mathbf{x}^{(i)}$（链式法则的结果），对 $b$ 求导就只剩残差本身。

### 1.5 概率视角：为什么用 MSE？

假设观测含高斯噪声：

$$y = \mathbf{w}^\top\mathbf{x} + b + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) \tag{1.13}$$

> $\epsilon$ 是随机噪声，服从均值为 0、方差为 $\sigma^2$ 的正态分布。

似然函数为：

$$P(y|\mathbf{x}) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y - \mathbf{w}^\top\mathbf{x} - b)^2}{2\sigma^2}\right) \tag{1.14}$$

> 给定输入 $\mathbf{x}$ 时观测到 $y$ 的概率密度。正态分布的中心在预测值 $\mathbf{w}^\top\mathbf{x} + b$ 处，预测越准，概率密度越大。

对所有样本取负对数似然：

$$-\log P(\mathbf{y}|\mathbf{X}) = \sum_{i=1}^{n} \frac{1}{2\sigma^2}\left(y^{(i)} - \mathbf{w}^\top\mathbf{x}^{(i)} - b\right)^2 + \text{常数} \tag{1.15}$$

> 取对数把连乘变成求和，加负号把"最大化概率"变成"最小化损失"。去掉常数项后，和 MSE（公式 1.4）形式完全一致。

**结论：最小化 MSE $\Leftrightarrow$ 高斯噪声假设下的最大似然估计。**

### 1.6 矢量化

循环实现 vs 矢量化实现的速度差异约 **数百倍**。原因：矢量化操作利用底层 BLAS 库和 GPU 并行计算，而循环只能串行执行。

---

## 二、泛化 <sub>([generalization.ipynb](generalization.ipynb))</sub>

### 2.1 训练误差与泛化误差

**训练误差（经验风险）**：

$$R_{\text{emp}}[f] = \frac{1}{n}\sum_{i=1}^{n} l\left(f(\mathbf{x}^{(i)}),\ y^{(i)}\right) \tag{2.1}$$

> 模型 $f$ 在 $n$ 个训练样本上的平均损失。这个值我们可以直接算出来。

**泛化误差（总体风险）**：

$$R[f] = E_{(\mathbf{x}, y) \sim P}\left[l\left(f(\mathbf{x}),\ y\right)\right] \tag{2.2}$$

> 模型 $f$ 在整个数据分布 $P$ 上的期望损失。这个值我们永远无法精确得到——只能用测试集近似。

**独立同分布假设（IID）**：训练集和测试集中的样本独立地从同一个分布 $P$ 中抽取。这是整个统计学习框架的基石。

### 2.2 模型复杂度：多项式拟合示例

用 $d$ 次多项式拟合数据：

$$\hat{y} = \sum_{i=0}^{d} w_i x^i \tag{2.3}$$

> 当 $d=1$ 是直线，$d=2$ 是抛物线，$d$ 越大函数越复杂。如果 $d \geq n$（多项式次数不小于样本数），可以完美拟合任何数据点——但这不代表模型好，可能只是在"背答案"。

### 2.3 过拟合与欠拟合的判断

| 指标 | 欠拟合 | 合适 | 过拟合 |
|------|--------|------|--------|
| 训练误差 | 高 | 适中 | 很低 |
| 验证误差 | 高 | 适中 | 高 |
| 泛化差距 | 小 | 小 | 大 |
| 处方 | 增加模型复杂度 | 维持 | 正则化 / 增加数据 |

### 2.4 模型选择

**三路拆分**：训练集 / 验证集 / 测试集

- 训练集：拟合模型参数
- 验证集：选择超参数和模型架构
- 测试集：最终评估，只用一次

**K 折交叉验证**：当数据稀缺时，将数据分成 $K$ 份，轮流以其中一份为验证集、其余为训练集，重复 $K$ 次取平均。每个样本恰好做一次验证。

---

## 三、权重衰减 <sub>([weight-decay.ipynb](weight-decay.ipynb))</sub>

### 3.1 L2 正则化

在原始损失后加上权重的 L2 惩罚：

$$L_{\text{reg}}(\mathbf{w}, b) = L(\mathbf{w}, b) + \frac{\lambda}{2}\|\mathbf{w}\|^2 \tag{3.1}$$

> $\|\mathbf{w}\|^2 = \sum_{j} w_j^2$ 是权重的 L2 范数平方。$\lambda \geq 0$ 是正则化系数：$\lambda = 0$ 时退化为无正则化；$\lambda$ 越大，权重被压得越小。除以 2 同样是为了求导方便。注意**偏置 $b$ 通常不加正则化**。

### 3.2 带权重衰减的 SGD 更新

对公式 (3.1) 求梯度后，$\mathbf{w}$ 的更新变为：

$$\mathbf{w} \leftarrow (1 - \eta\lambda)\,\mathbf{w} - \frac{\eta}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \mathbf{x}^{(i)}\left(\mathbf{w}^\top\mathbf{x}^{(i)} + b - y^{(i)}\right) \tag{3.2}$$

> 与公式 (1.11) 相比，多了一个 $(1 - \eta\lambda)$ 的系数乘在 $\mathbf{w}$ 前面。因为 $\eta\lambda > 0$，所以 $(1 - \eta\lambda) < 1$，每一步更新都会让 $\mathbf{w}$ 稍微"缩小"一点——这就是"权重衰减"。

### 3.3 L2 vs L1 正则化

$$\text{L2 惩罚：}\quad \frac{\lambda}{2}\|\mathbf{w}\|^2 = \frac{\lambda}{2}\sum_j w_j^2 \tag{3.3}$$

$$\text{L1 惩罚：}\quad \lambda\|\mathbf{w}\|_1 = \lambda\sum_j |w_j| \tag{3.4}$$

| | L2（Ridge） | L1（Lasso） |
|---|---|---|
| 惩罚项 | $\sum w_j^2$ | $\sum \|w_j\|$ |
| 梯度 | $2w_j$（平滑） | $\text{sign}(w_j)$（在零点不可导） |
| 效果 | 权重均匀缩小 | 权重变稀疏（很多直接变零） |
| 用途 | 深度学习中的默认选择 | 特征选择 |

### 3.4 概率视角

L2 正则化对应**高斯先验** $\mathbf{w} \sim \mathcal{N}(0, 1/\lambda)$ 下的最大后验估计（MAP）。L1 正则化对应**拉普拉斯先验**。

---

## 关键概念串联表

| 维度 | 线性回归 |
|------|---------|
| **模型** | $\hat{y} = \mathbf{w}^\top\mathbf{x} + b$ |
| **损失函数** | MSE：$\frac{1}{2}(\hat{y} - y)^2$ |
| **概率解释** | 高斯噪声假设 → MLE = 最小化 MSE |
| **解析解** | $\mathbf{w}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| **优化** | Mini-batch SGD |
| **梯度** | $\hat{y} - y$（乘特征向量） |
| **正则化** | L2 权重衰减：$(1-\eta\lambda)\mathbf{w}$ |
| **泛化** | 训练/验证/测试三路拆分 + 交叉验证 |

---

### 📌 数值走查示例：从数据到梯度

用一个最简单的例子：2 个特征（面积 $x_1$、房龄 $x_2$），1 个样本。

**已知：**
- 输入特征：$\mathbf{x} = (80, 10)$（80 平米，10 年房龄）
- 真实房价：$y = 300$（万元）
- 当前权重：$\mathbf{w} = (2.0, -1.0)$，偏置 $b = 150$
- 学习率：$\eta = 0.001$

**第 ① 步：前向传播 (1.1)**

$$\hat{y} = w_1 x_1 + w_2 x_2 + b = 2.0 \times 80 + (-1.0) \times 10 + 150 = 160 - 10 + 150 = 300$$

> 预测值 300，恰好等于真实值。如果真是这样，训练就完成了。但为了演示完整流程，我们把权重改成"不太对"的值。

**重新设定权重：** $\mathbf{w} = (1.5, -0.5)$，$b = 100$

$$\hat{y} = 1.5 \times 80 + (-0.5) \times 10 + 100 = 120 - 5 + 100 = 215$$

**第 ② 步：计算损失 (1.3)**

$$l = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(215 - 300)^2 = \frac{1}{2}(85)^2 = \frac{1}{2} \times 7225 = 3612.5$$

**第 ③ 步：计算梯度 (1.11, 1.12)**

残差：$\hat{y} - y = 215 - 300 = -85$

$$\frac{\partial l}{\partial w_1} = x_1 \cdot (\hat{y} - y) = 80 \times (-85) = -6800$$

$$\frac{\partial l}{\partial w_2} = x_2 \cdot (\hat{y} - y) = 10 \times (-85) = -850$$

$$\frac{\partial l}{\partial b} = \hat{y} - y = -85$$

> 梯度为负，说明这三个参数都应该增大。

**第 ④ 步：更新参数 (1.10)**

$$w_1 \leftarrow 1.5 - 0.001 \times (-6800) = 1.5 + 6.8 = 8.3$$

$$w_2 \leftarrow -0.5 - 0.001 \times (-850) = -0.5 + 0.85 = 0.35$$

$$b \leftarrow 100 - 0.001 \times (-85) = 100 + 0.085 = 100.085$$

> 更新后预测：$8.3 \times 80 + 0.35 \times 10 + 100.085 = 664 + 3.5 + 100.085 = 767.6$，过冲了！这是因为学习率太大了。实际训练中会用小批量平均来稳定梯度，并仔细调整学习率。

---
---

# 第三部分：代码实现篇

---

## 一、面向对象设计 <sub>([oo-design.ipynb](oo-design.ipynb))</sub>

本书定义了三个核心类，贯穿所有后续章节：

| 类 | 职责 | 关键方法 |
|---|---|---|
| `Module` | 模型（网络结构 + 训练/验证步骤） | `forward`, `loss`, `training_step`, `configure_optimizers` |
| `DataModule` | 数据（加载、预处理、分批） | `train_dataloader`, `val_dataloader` |
| `Trainer` | 编排训练循环 | `fit(model, data)`, `fit_epoch` |

### 1.1 核心设计模式

```python
# 1. add_to_class 装饰器：允许在类定义之外往类里添加方法
@d2l.add_to_class(A)
def do(self):
    print('Class A', self.x)

# 2. HyperParameters 基类：自动把 __init__ 的所有参数存为属性
class MyClass(d2l.HyperParameters):
    def __init__(self, a, b):
        self.save_hyperparameters()  # 自动创建 self.a = a, self.b = b
```

### 1.2 Trainer 训练循环（所有模型通用）

```python
@d2l.add_to_class(d2l.Trainer)
def fit_epoch(self):
    self.model.train()
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

> 这个训练循环从线性回归到 Transformer 都是一样的。变的只是 `model`（不同网络）和 `loss`（不同损失函数）。

---

## 二、合成回归数据 <sub>([synthetic-regression-data.ipynb](synthetic-regression-data.ipynb))</sub>

```python
class SyntheticRegressionData(d2l.DataModule):
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, len(w))           # 标准正态分布的随机特征
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise
```

> **要点：** 因为真实参数已知（$\mathbf{w}$ 和 $b$ 是我们自己设定的），所以可以在训练后检验模型是否学对了。噪声标准差 0.01 很小，正常训练后学到的参数应该非常接近真值。

### DataLoader：手写 vs 框架

```python
# 手写版（教学用）
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)

# 框架版（实际用）：内部使用 TensorDataset + DataLoader，
# 自动处理打乱、分批、多线程预读
```

---

## 三、线性回归从零实现 <sub>([linear-regression-scratch.ipynb](linear-regression-scratch.ipynb))</sub>

### 3.1 模型 + 损失

```python
class LinearRegressionScratch(d2l.Module):
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = torch.normal(0, sigma, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b  # 公式 (1.2)

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2                 # 公式 (1.3)
        return l.mean()
```

### 3.2 手写 SGD

```python
class SGD(d2l.HyperParameters):
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad         # 公式 (1.10)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

### 3.3 训练

```python
model = LinearRegressionScratch(2, lr=0.03)
data = SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

### 注意事项

- **权重初始化**：用小的随机数（标准差 0.01），不用零——全零初始化在对称网络中会导致所有神经元学到相同的东西
- **`requires_grad=True`**：告诉 PyTorch 跟踪该张量上的所有运算，以便 `backward()` 自动求梯度
- **`l.mean().backward()`**：损失必须是标量才能反向传播，所以先取平均
- **`grad.zero_()`**：每次更新后必须清零梯度，否则 PyTorch 会累加梯度

---

## 四、线性回归简洁实现 <sub>([linear-regression-concise.ipynb](linear-regression-concise.ipynb))</sub>

### 从零实现 vs 框架 API 对比

| 组件 | 从零实现 | PyTorch API |
|------|---------|------------|
| 模型 | `torch.matmul(X, w) + b` | `nn.LazyLinear(1)` |
| 损失 | 手写 `(y_hat - y)**2 / 2` | `nn.MSELoss()` |
| 优化器 | 手写 SGD | `torch.optim.SGD(net.parameters(), lr=...)` |
| 参数初始化 | `torch.normal(...)` | `net.weight.data.normal_(0, 0.01)` |

### 完整代码

```python
class LinearRegression(d2l.Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)                # 自动推断输入维度
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
```

### 关键 API

- **`nn.LazyLinear(out)`**：延迟初始化的全连接层，第一次 `forward` 时自动推断输入维度
- **`nn.MSELoss()`**：返回平均值（不带 $\frac{1}{2}$，但不影响优化）
- **`self.parameters()`**：返回所有可学习参数的迭代器
- **`_` 结尾方法**（`normal_`, `fill_`, `zero_`）：原地操作，不创建新张量

---

## 五、权重衰减实现 <sub>([weight-decay.ipynb](weight-decay.ipynb))</sub>

### 5.1 实验设置

```python
# 高维线性回归：200 个特征，但只有 20 个训练样本 → 严重过拟合
# 真实参数：w_i = 0.01, b = 0.05
data = d2l.SyntheticRegressionData(
    w=torch.ones((200, 1)) * 0.01, b=0.05,
    num_train=20, num_val=100)
```

### 5.2 从零实现

```python
def l2_penalty(w):
    return (w ** 2).sum() / 2

class WeightDecayScratch(d2l.LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)
        self.save_hyperparameters()

    def loss(self, y_hat, y):
        return (super().loss(y_hat, y) +       # 原始 MSE
                self.lambd * l2_penalty(self.w)) # + λ × L2 惩罚
```

### 5.3 框架实现

```python
class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},  # 权重有衰减
            {'params': self.net.bias}],                            # 偏置无衰减
            lr=self.lr)
```

### 5.4 实验结果

| | 无正则化 ($\lambda = 0$) | 有正则化 ($\lambda = 3$) |
|---|---|---|
| 训练误差 | 接近 0（记住了训练数据） | 较高（被限制了） |
| 验证误差 | 很高（过拟合） | 较低（泛化好） |
| 权重 L2 范数 | 较大 | 很小 |

### ⚠️ 常见坑

1. **偏置不要加正则化**：框架中需要用参数组分别设置 `weight_decay`
2. **`weight_decay` 参数直接传给优化器**：不需要手动在损失中加惩罚项
3. **$\lambda$ 需要调参**：太小没效果，太大欠拟合——用验证集选最佳值

<img src="img/training_loop.png" alt="训练循环" style="max-width:640px; max-height:800px;">
