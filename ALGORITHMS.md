# PID 迭代整定与智能诊断系统：核心算法深度解析与有效性论证

## 1. 导论
本白皮书旨在为控制系统评审专家提供本程序底层逻辑的详尽推导与数学论证。系统通过“模型驱动”结合“数据校验”，解决了工业现场 PID 整定中长期存在的“盲目试错”与“鲁棒性不足”问题。

---

## 2. 过程辨识：基于非线性最小二乘与数值模拟的耦合算法

### 2.1 物理模型描述
受控对象被抽象为含时滞的一阶线性系统 (FOPDT)：
$$ \tau \frac{dy(t)}{dt} + y(t) = K u(t - \theta) + y_0 $$
其中参数空间 $\mathcal{P} = \{K, \tau, \theta, y_0\}$ 必须满足物理约束：$\tau > 0, \theta \ge 0$。

### 2.2 离散化与数值稳定性 (Numerical Stability)
程序采用前向欧拉法进行时域模拟。为防止在寻优过程中由于 $\tau$ 过小导致的数值振荡，系统引入了**时间常数下限保护机制**。
离散化递推公式：
$$ y_{pred}[k] = y_{pred}[k-1] + \frac{\Delta t}{\max(\tau, \Delta t)} \left( K \cdot (u[k - 1 - \lfloor\theta/\Delta t\rfloor] - u_0) - (y_{pred}[k-1] - y_0) \right) $$

### 2.3 优化目标与非凸性处理
损失函数定义为误差二范数：
$$ f(K, \tau, \theta, y_0) = \sum_{k=1}^N \| y_{meas}[k] - y_{pred}[k] \|^2 $$
**算法策略论证**：由于 $e^{-\theta s}$ 导致 $f$ 对 $\theta$ 是高度非凸且可能存在多个局部极小值的。本系统采用 **Global Grid Scan + Local L-BFGS-B** 的两段式策略：
1.  **全局网格扫描**：在 $\theta \in [0, 0.4 \cdot T_{total}]$ 范围内进行 15 个采样点的遍历，粗定位 $\theta$ 的最优区间。
2.  **约束 L-BFGS-B**：在选定的网格点附近，利用二阶梯度信息加速收敛。

---

## 3. 整定理论：Skogestad SIMC 鲁棒整定策略

### 3.1 法则推导
SIMC 本质上是一种简化版的内模型控制 (IMC)。对于 FOPDT 模型，其标准形式下目标闭环传递函数为 $G_{cl}(s) = \frac{e^{-\theta s}}{\tau_c s + 1}$。
通过代数化简，推导出控制器增益：
$$ K_c = \frac{1}{K} \frac{\tau}{\tau_c + \theta} $$

### 3.2 鲁棒性控制指标 ($M_s$)
在控制工程中，鲁棒性通过最大灵敏度函数 $M_s = \max |(1 + G K)^{-1}|$ 衡量。
*   **本系统安全标准**：
    *   **保守模式** ($\tau_c = 10\theta$): 对应 $M_s \approx 1.3$，系统具备极高的幅值/相角裕度，适合传感器噪声大、对象非线性强的场景。
    *   **适中模式** ($\tau_c = 3\theta$): 对应 $M_s \approx 1.6$，这是工业界公认的最佳平衡点。
    *   **激进模式** ($\tau_c = \theta$): 对应 $M_s \approx 1.9 \sim 2.2$，接近稳定极限，仅建议在纯惯性对象且无死区时使用。

---

## 4. 性能指标：数学定义与评价准则

评审人员可根据以下公式复核程序输出的 KPI 指标：

### 4.1 控制精度：IAE (Integral Absolute Error)
$$ IAE = \int_{0}^{T} |SP(t) - PV(t)| dt \approx \sum |SP_k - PV_k| \Delta t $$
**有效性论证**：相比 MSE，IAE 对小幅持续偏差更敏感，更能反映工业生产的经济性损耗。

### 4.2 执行器负载：TV (Total Variation)
$$ TV = \sum_{k=1}^{N} |OP_k - OP_{k-1}| $$
**物理意义**：TV 量化了执行器（如调节阀）的动作强度。高 TV 意味着高机械磨损。本系统将 TV 与 $K_p$ 挂钩，通过限制 $K_p$ 调整步长来保护硬件。

### 4.3 控制攻击性 (Aggressiveness Index)
$$ Aggr = \frac{\Delta OP / OP_{range}}{\Delta Error / SP_{range}} $$
该指标反映了控制器对单位误差的即时爆发力。若 $Aggr > 5.0$，系统将提示“控制过于激进，易放大高频噪声”。

---

## 5. 诊断逻辑：非线性故障模式识别

### 5.1 零点交叉震荡分析算例 (Numerical Example)
**原始信号**：$e = [1.2, 0.5, -0.3, -0.8, -0.2, 0.4, 0.9, 0.1, -0.5]$
1.  **检测交叉点**：符号变化发生在索引 2, 5, 8 处。
2.  **计算半周期**：$P_1 = 3$ 采样点, $P_2 = 3$ 采样点。
3.  **计算峰值 IAE**：
    *   正半周峰值 = 1.2
    *   负半周峰值 = 0.8
    *   下一正半周 = 0.9
4.  **衰减率计算**：$R = 0.9 / 1.2 = 0.75$。
5.  **判定结论**：由于衰减率 $< 1.0$ 且 $> 0.25$，判定为“收敛性震荡（稳定性尚可但阻尼不足）”。

### 5.2 阀门粘滞 (Valve Stiction) 的判定矩阵
系统基于 **Hagglund 交叉检测** 结合 **OP 速率变化**：
*   **特征 A**：$OP$ 呈现锯齿波或阶跃状变化（$\frac{dOP}{dt}$ 不连续）。
*   **特征 B**：$PV$ 在 $OP$ 越过某个阈值前保持静止，随后产生突跳。
*   **判定公式**：若 $\text{std}(PV_{flat}) < \text{NoiseFloor}$ 且 $\Delta OP > \text{Threshold}$，则标记 Stiction 风险。

---

## 6. 安全性论证：分步迭代的收敛性
程序强制执行 $20\%$ 的参数调整限制，其数学本质是**阻尼牛顿法 (Damped Newton Method)**。
在模型不确定性为 $\Delta$ 的情况下，小步长更新确保了：
$$ \| K_{p, new} - K_{p, optimal} \| < \epsilon \cdot \| K_{p, old} - K_{p, optimal} \| $$
只要 $\epsilon < 1$，系统整定过程即具有**渐进稳定性**。

---

## 7. 权威参考文献
[1] **Skogestad, S. (2003)**. "Simple analytic rules for model reduction and PID control". *Journal of Process Control*. (提供了 SIMC 规则的完整解析推导)  
[2] **Åström, K. J., & Hägglund, T. (2006)**. *Advanced PID Control*. ISA - The Instrumentation, Systems, and Automation Society. (PID 领域的圣经，支撑了本程序的诊断与整定框架)  
[3] **Bauer, M., et al. (2016)**. "A review of control loop monitoring, maintenance and optimization". *Annual Reviews in Control*. (支撑了 TV 与 IAE 的性能评价准则)  
[4] **Nocedal, J., & Wright, S. (2006)**. *Numerical Optimization*. Springer. (论证了 L-BFGS-B 的收敛性与内存效率)

---
**核准工程师**: ____________________  
**整定系统版本**: v0.1.0-Embedded-Runtime
