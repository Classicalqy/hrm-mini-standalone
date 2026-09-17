# RT 与 Flow 求解动力学实验总结

## 要解决的问题

我们希望判断：**RT 是否因为没有预设路径限制，能够采用更自由、更曲折的推理轨迹；这种轨迹自由度是否帮助它完成复杂推理，或者避开错误稳定状态？**

## 观点 1：Flow 的直线路径来自 interpolant，而不是架构限制

### 实验 1：二维解析轨迹

### 方法 1

构造直线、半圆和 1/2/4 次弯折的正弦路径五类二维解析轨迹。训练集含 10,000 条轨迹，测试集含 2,000 条轨迹，每条轨迹离散为 64 个时间点，并使用 3 个随机种子。

在相同规模的网络上比较三种条件：离散 RT 逐步拟合完整目标轨迹；直线 Flow 学习线性 interpolant 的解析速度；曲线 Flow 学习真实曲线 interpolant 的解析速度。统一评估 endpoint error、完整路径 RMSE、tortuosity、平均转向角、方向自相关和更新有效秩，以下以 path RMSE 为主要指标。

### 结果 1

三种子平均 path RMSE：

| 目标轨迹 | RT | 直线 Flow | 曲线 Flow |
|---|---:|---:|---:|
| 直线 | 0.0253 | **0.0012** | 0.0053 |
| 半圆 | 0.0258 | 0.3517 | **0.0263** |
| sine-1 | 0.0236 | 0.2056 | **0.0109** |
| sine-2 | 0.0273 | 0.2051 | **0.0229** |
| sine-4 | **0.0257** | 0.2038 | 0.0469 |

### 结论 1

曲线 Flow 相比直线 Flow 将非直线路径误差降低约 77%–95%。因此，**Flow 架构本身可以表示复杂曲线；常见 Flow 轨迹接近直线，主要是所选 interpolant 和训练目标造成的。**

## 观点 2：RT 的实际求解轨迹比直线 Flow 更自由

### 实验 2：严格匹配的 Sudoku RT/Flow

### 方法 2

使用 Sudoku-Extreme 的 1,000 道训练题和 held-out `test_hard`，正式种子为 1/2/3。RT 与 Flow 共享 4 层、hidden size 512 的 Transformer、输入 embedding、初始状态、正弦时间 embedding、固定正交 digit codebook 和无参数 decoder；同时固定初始化、样本顺序、optimizer steps 和训练轮数。

RT 按 \(z_{k+1}=F_\theta(z_k+x+e(t_k))\) 递归更新，并每 7 步 detach carry。Flow 使用线性 interpolant \(z_t=(1-t)z_0+t z_y\)，学习速度 \(z_y-z_0\)，推理时用 Euler 法在 \(t\in[0,1]\) 上积分。两者每个 optimizer step 均调用 backbone 7 次；一条完整轨迹包含 16 个 outer blocks，主结果统一使用 112 NFE。Flow 的 56/224 NFE 只用于检查数值离散误差。

对同一道题记录逐步预测，并投影到共同的 logits/probability 空间，计算 exact match、tortuosity、方向自相关、prediction reversals 和更新协方差参与秩。

### 结果 2

在共同输出概率空间中的结果为：

| 条件 | Exact match | Tortuosity | 方向自相关 | Prediction reversals | 有效秩 |
|---|---:|---:|---:|---:|---:|
| RT | 61.80% | 12.63 | -0.018 | 116.5 | 8.09 |
| 直线 Flow V1 | 0% | 1.06 | 0.976 | 0 | 1.10 |

### 结论 2

RT 的轨迹明显更曲折，方向变化、预测反转和有效更新维度也更多。**RT 确实表现出了更复杂的迭代动力学。**

但 Flow 的 exact match 为 0，因此这个比较受到“成功模型与失败模型”的混杂。它只能作为描述性观察，不能证明 RT 的复杂轨迹带来了更强的求解能力。

## 观点 3：RT 的轨迹几何可以被主动控制

### 实验 3：RT 曲率正则 sweep

### 方法 3

在 RT 的 endpoint cross-entropy 上加入相邻更新方向的曲率惩罚：

\[
L=L_{\rm CE}+\lambda\sum_k\left\|
\frac{\Delta z_{k+1}}{\|\Delta z_{k+1}\|}
-\frac{\Delta z_k}{\|\Delta z_k\|}
\right\|^2.
\]

比较 \(\lambda\in\{0,0.01,0.1,1\}\)，其余网络、数据顺序、训练预算和推理步数均保持不变，每个条件运行种子 1/2/3。不同条件在同一道题上配对比较；准确率差异使用 seed/puzzle hierarchical bootstrap 计算 95% 置信区间。

### 结果 3

| \(\lambda\) | Exact match | Tortuosity | 方向自相关 | Prediction reversals | 有效秩 |
|---:|---:|---:|---:|---:|---:|
| 0 | 61.80% | 12.63 | -0.018 | 116.5 | 8.09 |
| 0.01 | 63.17% | 11.28 | 0.042 | 86.2 | 7.44 |
| 0.1 | 62.63% | 11.19 | 0.052 | 91.2 | 7.53 |
| 1 | 61.93% | 10.58 | 0.104 | 83.6 | 7.48 |

从 \(\lambda=0\) 增加到 1 后，tortuosity 降低约 16.2%，prediction reversals 降低约 28.3%，方向自相关升高，state residual 和 Sudoku constraint violations 也同时下降。

### 结论 3

**曲率正则可以因果性地让 RT 轨迹更平滑、更少翻转、更低维。**

但 exact match 没有显著或单调变化：

| \(\lambda\) | 相对 baseline 的 EM 差异 | 95% bootstrap CI |
|---:|---:|---:|
| 0.01 | +1.37 pp | [-3.10, +6.63] pp |
| 0.1 | +0.83 pp | [-1.70, +3.33] pp |
| 1 | +0.13 pp | [-2.17, +2.37] pp |

三个区间都跨 0。因此，**当前没有证据说明轨迹更曲折或更平滑会直接提高 RT 的准确率。**

## 观点 4：RT 可能会收敛到错误固定点

### 实验 4：固定点检测

### 方法 4

在每个 micro-step 记录固定随机投影后的 hidden trajectory、预测、exact match、Sudoku 约束违反数、normalized state residual、logit change 和 prediction flip count。

错误固定点定义为：连续 8 步 normalized state residual 小于 \(10^{-3}\)，平均 logit change 小于 \(10^{-4}\)，预测保持不变，并且 exact match 为 false。另用 residual 阈值 \(10^{-2}\) 和 \(10^{-4}\) 做敏感性分析，避免结论依赖单一阈值。

若发现错误固定点，预设的后续检验是在相对 state RMS 的不同噪声尺度下进行扰动恢复，并用 JVP/VJP 估计局部 Jacobian 最大奇异值；原生 RT 还计划延长到 448 步观察长期行为。

### 结果 4

我们在现有 112-step trace 上检测 wrong-attractor rate，并对 residual 阈值进行敏感性分析。

### 结论 4

主要阈值下，所有条件的 wrong-attractor rate 都为 0。放宽阈值后，Flow 的结果会随 NFE 大幅变化，说明当前判据受到单步步长影响。

因此，**目前没有证据证明 RT 会快速进入错误固定点，也没有证据证明曲折轨迹能够帮助它逃离错误固定点。** 原生 RT 的 448-step rollout、扰动恢复和 Jacobian 分析尚未形成正式结果。

## 观点 5：需要先让 Flow 在 Sudoku 上工作，才能公平比较

### 实验 5：Flow V1/V2 调试

### 方法 5

Flow V1 使用标准 conditional flow matching：采样时间 \(t\)，构造 \(z_t=(1-t)z_0+t z_y\)，监督 \(v_\theta(z_t,t,x)\) 拟合 \(z_y-z_0\)，同时用预测终点计算 CE。每个 optimizer step 采样 7 个时间点，并在三个正式种子及 56/112/224 NFE 下检查 rollout。

针对 V1 的退化，Flow V2 将速度损失改为相对向量误差

\[
L_{\rm FM}=\mathbb E\frac{\|v_\theta-u\|_2^2}{\operatorname{stopgrad}(\|u\|_2^2)+10^{-6}},
\]

并加入 endpoint CE、endpoint cosine loss、修正后的弯曲路径终点公式和与 RT 一致的 chunked carry。`teacher` 条件始终使用解析 interpolant state；`on-policy` 条件在训练前 20% 使用 teacher state，20%–60% 线性切换，后 40% 完全使用模型生成的 state，同时仍保持每步 7 次 backbone 调用。

先在关闭 augmentation 的 32 道固定题上检查能否过拟合，再用非正式 seed 0 和固定 hash 的 20% dev/80% test 划分检查泛化。诊断指标包括速度相对误差与 cosine、预测/目标速度范数、自由 rollout endpoint accuracy、cell accuracy、constraint violations 和 \(t=0\) endpoint accuracy。

### 结果 5

Flow V1 在三个种子和 56/112/224 NFE 下 exact match 均为 0。分析发现：

- 512 维逐元素 MSE 使零速度预测的损失只有约 \(1/512\)；
- teacher-forced \(z_t\) 含有真实答案分量，使训练 CE 很低，但自由 rollout 失败。

Flow V2 随后加入相对速度损失、修正后的 endpoint 公式、chunked on-policy rollout、endpoint loss 和独立 dev/test 划分。

V2 可以在 32 道固定训练题上达到 100% free-rollout exact match，但在 seed-0 dev 集上：

| 条件 | Dev exact match | Dev cell accuracy | Dev \(t=0\) endpoint EM |
|---|---:|---:|---:|
| Teacher Flow V2 | 0% | 约 39%–41% | 0% |
| On-policy Flow V2 | 约 1% | 最高约 65% | 0% |

### 结论 5

V2 能记忆少量训练题，但不能泛化到新题。直线 CFM 在 \(t=0\) 要求：

\[
v_\theta(z_0,x,0)\approx z_y-z_0,
\]

这近似等价于要求一次 4-layer backbone 调用直接指出完整 Sudoku 答案方向。RT 可以利用递归状态逐步计算，而直线 CFM 的监督没有提供同样的计算过程。

因此，**当前 Flow 不是有效的 Sudoku baseline，不能用它与 RT 的轨迹差异解释性能差异。** Flow 的失败可以保留为方法限制或 negative control，暂时不继续投入正式三种子训练。

## 总结

1. Flow 能表示曲线路径，轨迹限制主要来自 interpolant。
2. RT 在 Sudoku 上表现出更曲折、更高维的实际求解轨迹。
3. RT 的轨迹几何可以被曲率正则稳定改变。
4. 大幅改变轨迹几何没有显著改变 exact match，因此几何与性能之间的因果关系尚未建立。
5. 当前没有检测到错误固定点，也无法证明 RT 通过曲折路线逃离错误状态。
6. Sudoku Flow 尚未达到可比较性能，因此跨模型结果只能作为描述性证据。

对最初问题的阶段性回答是：

> RT 的轨迹自由度确实更高，但现有实验尚未证明这种自由度是其推理性能的来源。标准直线 Flow Matching 在 Sudoku 上难以形成公平 baseline，原因可能是它要求过强的一步终点方向预测，而没有提供与递归推理对应的逐步计算路径。
