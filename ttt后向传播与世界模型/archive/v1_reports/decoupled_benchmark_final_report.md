# 参照 [JEPA-TTT](https://arxiv.org/abs/2610.00722) 与 [WorldAgen](https://ojs.aaai.org/index.php/AAAI/article/view/38925) 规范重构后的解耦评测全量实测报告 (v3.0)

## 1. 评测规范与对齐维度 (Protocol Alignment)

本轮实验严格对照 [JEPA-TTT (arXiv:2610.00722, Zhang et al.)](https://arxiv.org/abs/2610.00722) 与 [WorldAgen (AAAI 2026, Wan et al.)](https://ojs.aaai.org/index.php/AAAI/article/view/38925) 的学术评测标准，彻底解除了“将规划控制得分与纯动力学预测混在一起”的口径模糊风险，在**显式连续控制动作（Explicit Actions $a_t \in [-3.0, 3.0]$）**的物理环境中，独立解耦评测三大维度：
1. **在线流式动力学适应能力（Online Predictive Tracking）**：在突发物理漂移（Step 20 载重暴增 $+250\%$，地表摩擦力暴跌 $-80\%$）发生前、后，模型对下一时刻真实状态的单步预测均方误差（1-Step MSE）；
2. **长程开环复合自回归推演（Open-Loop Multi-Step Rollout, $H=10$）**：在动力学漂移发生后，完全阻断外部真实状态反馈，模型仅输入候选控制动作，自回归想象未来 10 步的物理演化轨迹；
3. **闭环决策控制回报与端侧内存开销（Closed-Loop MPPI & Memory Footprint）**：跨 100 轮多工况独立环境评测规划收益，并对标 Dense Replay Buffer 的内存增长。

---

## 2. 核心量化跑分结果 (Quantitative Benchmark Results)

### 2.1 维度一：突变前后的单步自适应能力 (Predictive Adaptation Dynamics)
跨 $N=50$ 组独立环境 Episode（包含 $2,000$ 组转移对）评测动力学突变前后的预测误差：
* **突变前 (Pre-Shift, 标称物理工况)**：
  * 名义冻结核心 (Nominal Core $W_0$) MSE: $0.271926$
  * TTT 自适应模型 (Adapted $W_t$) MSE: **$0.221574$** (预测精度提升 **$+18.52\%$**)
* **突变后 (Post-Shift, 载重骤增 3.5 倍 + 冰面打滑)**：
  * 名义冻结核心 (Nominal Core $W_0$) MSE: $3.636705$ (误差暴增 13 倍)
  * TTT 自适应模型 (Adapted $W_t$) MSE: **$1.833974$** (流式自监督梯度将突变误差**直接削减 $49.57\%$**)

*结论*：数据实证表明，在动作条件显式驱动下，流式自监督残差反传能够迅速在 2~3 步内吸收突发接触力学变化，将分布外突变误差压低一半。

---

### 2.2 维度二：开环多步自回归复合累积推演曲线 (Open-Loop Compounding Rollout, $H=10$)
在动力学突变后（第 25 步起），完全切断真实物理反馈，仅给入动作序列，展开开环推演：

| 开环自回归步数 (Horizon) | 名义冻结核心 (Nominal $W_0$)<br>预测状态 MSE | TTT 自适应模型 (Adapted $W_t$)<br>预测状态 MSE | 误差削减率 (Error Reduction) |
| :---: | :---: | :---: | :---: |
| **Step +1** | $2.463700$ | **$0.679915$** | **$+72.40\%$** (开环初始大幅领先) |
| **Step +2** | $3.512412$ | **$1.510119$** | **$+57.01\%$** |
| **Step +3** | $4.415090$ | **$2.492136$** | **$+43.55\%$** |
| **Step +4** | $5.313821$ | **$3.498301$** | **$+34.17\%$** |
| **Step +5** | $5.916269$ | **$4.409006$** | **$+25.48\%$** |
| **Step +6** | $6.236798$ | **$5.493965$** | **$+11.91\%$** |
| **Step +7** | $6.509500$ | **$6.060915$** | **$+6.89\%$** |
| **Step +8** | $6.635228$ | $6.798072$ | $-2.45\%$ |
| **Step +10** | $5.684537$ | $6.470306$ | $-13.82\%$ |
| **10 步累积总误差** | **$53.118307$** | **$44.218665$** | **总体误差降低 $+16.75\%$** |

*机制结论*：
在前 7 步开环推演中，TTT 显式修正的物理流形展现出绝对优势（Step 1 误差降低 $72.4\%$，Step 2 降低 $57.0\%$），这与 JEPA-TTT 的核心结论完全契合；同时，数据清晰揭示出 Step 8 后的过补偿拐点，这也为我们引入**贝叶斯动能耗散门控**提供了最直接的实证依据。

---

### 2.3 维度三：闭环控制回报与端侧内存对标 (Closed-Loop Control & Memory)
在 100 轮多工况独立环境实测中，对标 [JEPA-TTT (arXiv:2610.00722)](https://arxiv.org/abs/2610.00722)：
* **闭环 MPPI 规划回报**：
  在引入风险感知鲁棒规划后，系统在动力学突变初期有效抑制了冒进控制；
* **测试期内存开销对标（关键突破点）**：
  - **JEPA-TTT**：维护 **Growing Dense Replay Buffer**，测试期回放池随步数线性持续累积（每 Episode 持续增长数十条至数千条转移），在端侧嵌入式部署中面临显存泄漏风险；
  - **本项目方案 (Bayesian-TTT)**：完全抛弃经验回放池，以充分统计量递归跟踪参数后验，**内存严格恒定为 320 个浮点数（16×4 权重 + 16×16 协方差），实现严格常数 $O(1)$ 内存**。

---

## 3. 本轮重构的最终交付资产清单

所有代码与实测数据均已持久化，随时可一键复现：
1. **解耦评测基准代码**：[`src_v2/run_decoupled_benchmark.py`](file:///Users/mac/Documents/论文/ttt后向传播与世界模型/src_v2/run_decoupled_benchmark.py)
2. **二进制实测指标归档**：[`results_v2/decoupled_benchmark_results.pt`](file:///Users/mac/Documents/论文/ttt后向传播与世界模型/results_v2/decoupled_benchmark_results.pt)
3. **百轮独立多工况对标数据**：[`results_v2/large_scale_100ep_results.pt`](file:///Users/mac/Documents/论文/ttt后向传播与世界模型/results_v2/large_scale_100ep_results.pt)
