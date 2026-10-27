# KV Cache 原理详解：自回归解码中的键值复用机制

&emsp;&emsp;大语言模型的推理过程是逐 token 串行生成的。这一特性使得朴素实现在每步解码时都要对完整序列重新执行一次前向计算，产生大量冗余。**KV Cache** 是消除该冗余最基础、也最关键的工程手段。

---

## 1. 自回归解码的基本流程

&emsp;&emsp;给定初始 Prompt，模型按如下循环生成文本：

1. 将当前序列输入模型，得到下一个 token 的概率分布；
2. 采样或取 argmax，得到新 token；
3. 将新 token 追加至序列末尾，重复步骤 1，直至遇到结束符或达到长度上限。

&emsp;&emsp;每一步仅输出**一个** token，这是 Decode 阶段串行性的来源。

<svg width="920" height="400" viewBox="0 0 920 400" xmlns="http://www.w3.org/2000/svg" font-family="'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <linearGradient id="f1_model" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#4338ca"/></linearGradient>
    <linearGradient id="f1_tok" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#eff6ff"/><stop offset="100%" stop-color="#bfdbfe"/></linearGradient>
    <linearGradient id="f1_out" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#fef3c7"/><stop offset="100%" stop-color="#fbbf24"/></linearGradient>
    <filter id="f1_sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.15"/></filter>
    <marker id="f1_arr" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto"><path d="M0,0 L9,3 L0,6 Z" fill="#94a3b8"/></marker>
  </defs>
  <text x="20" y="76" font-size="13" fill="#64748b" font-weight="600">Step 1</text>
  <rect x="90" y="48" width="76" height="44" rx="10" fill="url(#f1_tok)" stroke="#93c5fd" filter="url(#f1_sh)"/><text x="128" y="76" font-size="17" text-anchor="middle" fill="#1e3a8a" font-weight="600">今天</text>
  <rect x="176" y="48" width="76" height="44" rx="10" fill="url(#f1_tok)" stroke="#93c5fd" filter="url(#f1_sh)"/><text x="214" y="76" font-size="17" text-anchor="middle" fill="#1e3a8a" font-weight="600">天气</text>
  <line x1="256" y1="70" x2="560" y2="70" stroke="#94a3b8" stroke-width="2" marker-end="url(#f1_arr)"/>
  <rect x="576" y="32" width="150" height="76" rx="14" fill="url(#f1_model)" filter="url(#f1_sh)"/><text x="651" y="76" font-size="16" text-anchor="middle" fill="#fff" font-weight="700">Transformer</text>
  <line x1="730" y1="70" x2="790" y2="70" stroke="#f59e0b" stroke-width="2" marker-end="url(#f1_arr)"/>
  <rect x="800" y="48" width="76" height="44" rx="10" fill="url(#f1_out)" stroke="#f59e0b" filter="url(#f1_sh)"/><text x="838" y="76" font-size="17" text-anchor="middle" fill="#78350f" font-weight="700">真</text>
  <text x="20" y="206" font-size="13" fill="#64748b" font-weight="600">Step 2</text>
  <rect x="90" y="178" width="76" height="44" rx="10" fill="url(#f1_tok)" stroke="#93c5fd" filter="url(#f1_sh)"/><text x="128" y="206" font-size="17" text-anchor="middle" fill="#1e3a8a" font-weight="600">今天</text>
  <rect x="176" y="178" width="76" height="44" rx="10" fill="url(#f1_tok)" stroke="#93c5fd" filter="url(#f1_sh)"/><text x="214" y="206" font-size="17" text-anchor="middle" fill="#1e3a8a" font-weight="600">天气</text>
  <rect x="262" y="178" width="76" height="44" rx="10" fill="url(#f1_out)" stroke="#f59e0b" filter="url(#f1_sh)"/><text x="300" y="206" font-size="17" text-anchor="middle" fill="#78350f" font-weight="700">真</text>
  <line x1="342" y1="200" x2="560" y2="200" stroke="#94a3b8" stroke-width="2" marker-end="url(#f1_arr)"/>
  <rect x="576" y="162" width="150" height="76" rx="14" fill="url(#f1_model)" filter="url(#f1_sh)"/><text x="651" y="206" font-size="16" text-anchor="middle" fill="#fff" font-weight="700">Transformer</text>
  <line x1="730" y1="200" x2="790" y2="200" stroke="#f59e0b" stroke-width="2" marker-end="url(#f1_arr)"/>
  <rect x="800" y="178" width="76" height="44" rx="10" fill="url(#f1_out)" stroke="#f59e0b" filter="url(#f1_sh)"/><text x="838" y="206" font-size="17" text-anchor="middle" fill="#78350f" font-weight="700">好</text>
  <text x="20" y="336" font-size="13" fill="#64748b" font-weight="600">Step 3</text>
  <rect x="90" y="308" width="76" height="44" rx="10" fill="url(#f1_tok)" stroke="#93c5fd" filter="url(#f1_sh)"/><text x="128" y="336" font-size="17" text-anchor="middle" fill="#1e3a8a" font-weight="600">今天</text>
  <rect x="176" y="308" width="76" height="44" rx="10" fill="url(#f1_tok)" stroke="#93c5fd" filter="url(#f1_sh)"/><text x="214" y="336" font-size="17" text-anchor="middle" fill="#1e3a8a" font-weight="600">天气</text>
  <rect x="262" y="308" width="76" height="44" rx="10" fill="url(#f1_out)" stroke="#f59e0b" filter="url(#f1_sh)"/><text x="300" y="336" font-size="17" text-anchor="middle" fill="#78350f" font-weight="700">真</text>
  <rect x="348" y="308" width="76" height="44" rx="10" fill="url(#f1_out)" stroke="#f59e0b" filter="url(#f1_sh)"/><text x="386" y="336" font-size="17" text-anchor="middle" fill="#78350f" font-weight="700">好</text>
  <line x1="428" y1="330" x2="560" y2="330" stroke="#94a3b8" stroke-width="2" marker-end="url(#f1_arr)"/>
  <rect x="576" y="292" width="150" height="76" rx="14" fill="url(#f1_model)" filter="url(#f1_sh)"/><text x="651" y="336" font-size="16" text-anchor="middle" fill="#fff" font-weight="700">Transformer</text>
  <line x1="730" y1="330" x2="790" y2="330" stroke="#f59e0b" stroke-width="2" marker-end="url(#f1_arr)"/>
  <rect x="800" y="308" width="76" height="44" rx="10" fill="url(#f1_out)" stroke="#f59e0b" filter="url(#f1_sh)"/><text x="838" y="336" font-size="17" text-anchor="middle" fill="#78350f" font-weight="700">！</text>
</svg>

**图 1：自回归生成流程。蓝色为原始 Prompt token，橙色为逐步生成的 token。**

&emsp;&emsp;设序列长度在第 $t$ 步为 $t$。若不加任何缓存，第 $t$ 步前向计算的输入长度为 $t$，且其中前 $t-1$ 个 token 与上一步完全相同。

---

## 2. 朴素解码的计算冗余

&emsp;&emsp;以生成 3 个 token 为例，朴素实现的计算分布如下：

| 步 | 输入序列 | 参与计算的 token 数 |
|---|---|---|
| 1 | $[t_1]$ | 1 |
| 2 | $[t_1,t_2]$ | 2 |
| 3 | $[t_1,t_2,t_3]$ | 3 |

&emsp;&emsp;$t_1$ 被计算 3 次，$t_2$ 被计算 2 次。总 token 级计算量为 $1+2+3=6$，而实际只有 3 个 token。

<svg width="920" height="480" viewBox="0 0 920 480" xmlns="http://www.w3.org/2000/svg" font-family="'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <filter id="f2_sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-opacity="0.12"/></filter>
  </defs>
  <text x="170" y="52" font-size="15" text-anchor="middle" fill="#475569" font-weight="700">Step 1</text>
  <text x="460" y="52" font-size="15" text-anchor="middle" fill="#475569" font-weight="700">Step 2</text>
  <text x="750" y="52" font-size="15" text-anchor="middle" fill="#475569" font-weight="700">Step 3</text>
  <rect x="105" y="298" width="130" height="42" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5" filter="url(#f2_sh)"/><text x="170" y="325" font-size="15" text-anchor="middle" fill="#14532d" font-weight="600">t₁</text>
  <rect x="395" y="298" width="130" height="42" rx="9" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5" filter="url(#f2_sh)"/><text x="460" y="325" font-size="14" text-anchor="middle" fill="#7f1d1d" font-weight="600">t₁（重算）</text>
  <rect x="395" y="248" width="130" height="42" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5" filter="url(#f2_sh)"/><text x="460" y="275" font-size="15" text-anchor="middle" fill="#14532d" font-weight="600">t₂</text>
  <rect x="685" y="298" width="130" height="42" rx="9" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5" filter="url(#f2_sh)"/><text x="750" y="325" font-size="14" text-anchor="middle" fill="#7f1d1d" font-weight="600">t₁（重算）</text>
  <rect x="685" y="248" width="130" height="42" rx="9" fill="#fee2e2" stroke="#dc2626" stroke-width="1.5" filter="url(#f2_sh)"/><text x="750" y="275" font-size="14" text-anchor="middle" fill="#7f1d1d" font-weight="600">t₂（重算）</text>
  <rect x="685" y="198" width="130" height="42" rx="9" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5" filter="url(#f2_sh)"/><text x="750" y="225" font-size="15" text-anchor="middle" fill="#14532d" font-weight="600">t₃</text>
  <text x="170" y="382" font-size="13" text-anchor="middle" fill="#64748b">计算 1 个 token</text>
  <text x="460" y="382" font-size="13" text-anchor="middle" fill="#64748b">计算 2 个 token</text>
  <text x="750" y="382" font-size="13" text-anchor="middle" fill="#64748b">计算 3 个 token</text>
  <rect x="150" y="415" width="620" height="46" rx="12" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>
  <text x="460" y="444" font-size="14" text-anchor="middle" fill="#9a3412" font-weight="600">实际只有 3 个 token，却做了 1 + 2 + 3 = 6 次 token 级计算</text>
</svg>

**图 2：朴素实现的重复计算。红色块为被重复计算的历史 token。**

&emsp;&emsp;在真实模型中，每个 token 需经过 $L$ 层 Transformer，每层包含一次完整注意力与前馈网络。冗余计算量随序列长度线性累积，不可忽略。

---

## 3. Self-Attention 的计算结构与可复用性

&emsp;&emsp;设输入隐状态矩阵 $X \in \mathbb{R}^{n \times d}$，其中 $n$ 为序列长度、$d$ 为隐藏维度。每层注意力通过三个线性投影得到：

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V
$$

&emsp;&emsp;其中 $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$。三者的语义分别为：

- **Q（Query）**：当前 token 用于检索的查询向量；
- **K（Key）**：各 token 作为被检索对象的索引向量；
- **V（Value）**：各 token 携带的实际内容向量。

&emsp;&emsp;注意力输出为 $\mathrm{softmax}(QK^\top/\sqrt{d_h})\,V$，即用 Q 与全部 K 的相似度对 V 加权聚合。

<svg width="920" height="480" viewBox="0 0 920 480" xmlns="http://www.w3.org/2000/svg" font-family="'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <linearGradient id="f3_wq" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#60a5fa"/><stop offset="100%" stop-color="#2563eb"/></linearGradient>
    <linearGradient id="f3_wk" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#7c3aed"/></linearGradient>
    <linearGradient id="f3_wv" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#34d399"/><stop offset="100%" stop-color="#059669"/></linearGradient>
    <marker id="f3_arr" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker>
  </defs>
  <text x="155" y="66" font-size="13" text-anchor="middle" fill="#64748b" font-weight="700">X (输入隐状态)</text>
  <rect x="100" y="85" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="100" y="121" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="100" y="157" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/>
  <line x1="215" y1="100" x2="283" y2="100" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="215" y1="136" x2="283" y2="136" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="215" y1="172" x2="283" y2="172" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/>
  <rect x="290" y="70" width="90" height="102" rx="8" fill="url(#f3_wq)"/><text x="335" y="128" font-size="18" text-anchor="middle" fill="#fff" font-weight="700">W_Q</text>
  <line x1="385" y1="100" x2="463" y2="100" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="385" y1="136" x2="463" y2="136" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="385" y1="172" x2="463" y2="172" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/>
  <rect x="470" y="85" width="110" height="30" rx="5" fill="#dbeafe" stroke="#60a5fa"/><rect x="470" y="121" width="110" height="30" rx="5" fill="#dbeafe" stroke="#60a5fa"/><rect x="470" y="157" width="110" height="30" rx="5" fill="#dbeafe" stroke="#60a5fa"/>
  <text x="525" y="106" font-size="13" text-anchor="middle" fill="#1e40af" font-weight="600">q₁</text><text x="525" y="142" font-size="13" text-anchor="middle" fill="#1e40af" font-weight="600">q₂</text><text x="525" y="178" font-size="13" text-anchor="middle" fill="#1e40af" font-weight="600">q₃</text>
  <text x="610" y="125" font-size="14" fill="#1e40af" font-weight="600">Q = X · W_Q</text><text x="610" y="148" font-size="12" fill="#64748b">每个 token 对应一行 query</text>
  <rect x="100" y="215" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="100" y="251" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="100" y="287" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/>
  <line x1="215" y1="230" x2="283" y2="230" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="215" y1="266" x2="283" y2="266" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="215" y1="302" x2="283" y2="302" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/>
  <rect x="290" y="200" width="90" height="102" rx="8" fill="url(#f3_wk)"/><text x="335" y="258" font-size="18" text-anchor="middle" fill="#fff" font-weight="700">W_K</text>
  <line x1="385" y1="230" x2="463" y2="230" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="385" y1="266" x2="463" y2="266" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="385" y1="302" x2="463" y2="302" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/>
  <rect x="470" y="215" width="110" height="30" rx="5" fill="#ede9fe" stroke="#a78bfa"/><rect x="470" y="251" width="110" height="30" rx="5" fill="#ede9fe" stroke="#a78bfa"/><rect x="470" y="287" width="110" height="30" rx="5" fill="#ede9fe" stroke="#a78bfa"/>
  <text x="525" y="236" font-size="13" text-anchor="middle" fill="#5b21b6" font-weight="600">k₁</text><text x="525" y="272" font-size="13" text-anchor="middle" fill="#5b21b6" font-weight="600">k₂</text><text x="525" y="308" font-size="13" text-anchor="middle" fill="#5b21b6" font-weight="600">k₃</text>
  <text x="610" y="255" font-size="14" fill="#5b21b6" font-weight="600">K = X · W_K</text><text x="610" y="278" font-size="12" fill="#64748b">每个 token 对应一行 key</text>
  <rect x="100" y="345" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="100" y="381" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="100" y="417" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/>
  <line x1="215" y1="360" x2="283" y2="360" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="215" y1="396" x2="283" y2="396" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="215" y1="432" x2="283" y2="432" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/>
  <rect x="290" y="330" width="90" height="102" rx="8" fill="url(#f3_wv)"/><text x="335" y="388" font-size="18" text-anchor="middle" fill="#fff" font-weight="700">W_V</text>
  <line x1="385" y1="360" x2="463" y2="360" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="385" y1="396" x2="463" y2="396" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/><line x1="385" y1="432" x2="463" y2="432" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f3_arr)"/>
  <rect x="470" y="345" width="110" height="30" rx="5" fill="#d1fae5" stroke="#34d399"/><rect x="470" y="381" width="110" height="30" rx="5" fill="#d1fae5" stroke="#34d399"/><rect x="470" y="417" width="110" height="30" rx="5" fill="#d1fae5" stroke="#34d399"/>
  <text x="525" y="366" font-size="13" text-anchor="middle" fill="#065f46" font-weight="600">v₁</text><text x="525" y="402" font-size="13" text-anchor="middle" fill="#065f46" font-weight="600">v₂</text><text x="525" y="438" font-size="13" text-anchor="middle" fill="#065f46" font-weight="600">v₃</text>
  <text x="610" y="385" font-size="14" fill="#065f46" font-weight="600">V = X · W_V</text><text x="610" y="408" font-size="12" fill="#64748b">每个 token 对应一行 value</text>
</svg>

**图 3：自注意力中的 Q、K、V 投影。X 的每一行对应一个 token，分别经三个权重矩阵投影。**

&emsp;&emsp;**引理（K/V 的位置稳定性）**：在因果自回归生成中，第 $t$ 步输入序列的前 $t-1$ 个 token 与第 $t-1$ 步完全一致，且因果掩码保证这些位置的隐状态不受后续 token 影响。因此，历史 token 的 $k$ 与 $v$ 在后续所有步中保持不变。据此得到两条结论：

1. **K、V 可缓存**：历史 token 的 $k_i, v_i$ 只需计算一次，之后直接复用；
2. **Q 不可缓存**：每步新增 token 都需要一个全新的 query 去检索历史信息，$Q$ 必须实时计算。

KV Cache 的全部内容即源于此：缓存历史 token 的 K 与 V，避免重复投影。

---

## 4. KV Cache 的机制

&emsp;&emsp;核心策略是以显存换取计算量。每生成一个新 token：

1. 仅对当前 token 计算 $q_t, k_t, v_t$；
2. 将 $k_t, v_t$ 追加至缓存末尾；
3. 以 $q_t$ 与缓存中全部 $K_{1:t}$ 做注意力，对全部 $V_{1:t}$ 加权求和。

<svg width="920" height="420" viewBox="0 0 920 420" xmlns="http://www.w3.org/2000/svg" font-family="'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <filter id="f4_sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-opacity="0.12"/></filter>
    <marker id="f4_arr" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#cbd5e1"/></marker>
    <marker id="f4_arr2" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#f97316"/></marker>
  </defs>
  <rect x="105" y="60" width="90" height="50" rx="10" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" filter="url(#f4_sh)"/><text x="150" y="91" font-size="16" text-anchor="middle" fill="#475569" font-weight="600">t₁</text>
  <rect x="215" y="60" width="90" height="50" rx="10" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" filter="url(#f4_sh)"/><text x="260" y="91" font-size="16" text-anchor="middle" fill="#475569" font-weight="600">t₂</text>
  <rect x="325" y="60" width="90" height="50" rx="10" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" filter="url(#f4_sh)"/><text x="370" y="91" font-size="16" text-anchor="middle" fill="#475569" font-weight="600">t₃</text>
  <rect x="435" y="60" width="90" height="50" rx="10" fill="#ffedd5" stroke="#f97316" stroke-width="2" filter="url(#f4_sh)"/><text x="480" y="91" font-size="16" text-anchor="middle" fill="#9a3412" font-weight="700">t₄</text>
  <text x="570" y="82" font-size="13" fill="#64748b">← 历史 token（已缓存）</text>
  <text x="570" y="102" font-size="13" fill="#f97316" font-weight="600">← 新 token</text>
  <line x1="150" y1="115" x2="150" y2="218" stroke="#cbd5e1" stroke-width="1.8" marker-end="url(#f4_arr)"/>
  <line x1="260" y1="115" x2="260" y2="218" stroke="#cbd5e1" stroke-width="1.8" marker-end="url(#f4_arr)"/>
  <line x1="370" y1="115" x2="370" y2="218" stroke="#cbd5e1" stroke-width="1.8" marker-end="url(#f4_arr)"/>
  <line x1="480" y1="115" x2="480" y2="218" stroke="#f97316" stroke-width="2" marker-end="url(#f4_arr2)"/>
  <rect x="85" y="222" width="470" height="160" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="2"/>
  <text x="105" y="248" font-size="14" fill="#334155" font-weight="700">KV Cache</text>
  <text x="65" y="288" font-size="14" fill="#7c3aed" font-weight="700" text-anchor="end">K</text>
  <rect x="105" y="265" width="90" height="45" rx="6" fill="#ede9fe" stroke="#a78bfa"/><text x="150" y="293" font-size="14" text-anchor="middle" fill="#5b21b6" font-weight="600">k₁</text>
  <rect x="215" y="265" width="90" height="45" rx="6" fill="#ede9fe" stroke="#a78bfa"/><text x="260" y="293" font-size="14" text-anchor="middle" fill="#5b21b6" font-weight="600">k₂</text>
  <rect x="325" y="265" width="90" height="45" rx="6" fill="#ede9fe" stroke="#a78bfa"/><text x="370" y="293" font-size="14" text-anchor="middle" fill="#5b21b6" font-weight="600">k₃</text>
  <rect x="435" y="265" width="90" height="45" rx="6" fill="#ffedd5" stroke="#f97316" stroke-width="2"/><text x="480" y="293" font-size="14" text-anchor="middle" fill="#9a3412" font-weight="700">k₄</text>
  <text x="65" y="353" font-size="14" fill="#059669" font-weight="700" text-anchor="end">V</text>
  <rect x="105" y="330" width="90" height="45" rx="6" fill="#d1fae5" stroke="#34d399"/><text x="150" y="358" font-size="14" text-anchor="middle" fill="#065f46" font-weight="600">v₁</text>
  <rect x="215" y="330" width="90" height="45" rx="6" fill="#d1fae5" stroke="#34d399"/><text x="260" y="358" font-size="14" text-anchor="middle" fill="#065f46" font-weight="600">v₂</text>
  <rect x="325" y="330" width="90" height="45" rx="6" fill="#d1fae5" stroke="#34d399"/><text x="370" y="358" font-size="14" text-anchor="middle" fill="#065f46" font-weight="600">v₃</text>
  <rect x="435" y="330" width="90" height="45" rx="6" fill="#ffedd5" stroke="#f97316" stroke-width="2"/><text x="480" y="358" font-size="14" text-anchor="middle" fill="#9a3412" font-weight="700">v₄</text>
  <text x="600" y="265" font-size="14" fill="#334155" font-weight="600">历史 token 的 K、V</text>
  <text x="600" y="288" font-size="14" fill="#334155" font-weight="600">仅计算一次，跨步复用</text>
  <text x="600" y="340" font-size="14" fill="#f97316" font-weight="600">新 token 的 k、v</text>
  <text x="600" y="363" font-size="14" fill="#f97316" font-weight="600">计算后追加至缓存尾部</text>
</svg>

**图 4：KV Cache 的存储布局。K 与 V 按序列位置排列，新 token 的 k、v 追加至末尾。**

&emsp;&emsp;值得注意的是，缓存中不包含 Q。query 仅在当前步的注意力计算中使用一次，不具备跨步复用价值。

---

## 5. Prefill 阶段：Prompt 的并行处理

&emsp;&emsp;实际推理分为两个性质迥异的阶段：**Prefill** 与 **Decode**。

&emsp;&emsp;Prefill 发生在收到完整 Prompt 后、开始生成之前。此时模型持有全部 Prompt token，可对其并行计算 $Q$、$K$、$V$，属于计算密集型（compute-bound）操作。

<svg width="920" height="500" viewBox="0 0 920 500" xmlns="http://www.w3.org/2000/svg" font-family="'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <linearGradient id="f5_wq" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#60a5fa"/><stop offset="100%" stop-color="#2563eb"/></linearGradient>
    <linearGradient id="f5_wk" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#7c3aed"/></linearGradient>
    <linearGradient id="f5_wv" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#34d399"/><stop offset="100%" stop-color="#059669"/></linearGradient>
    <marker id="f5_arr" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker>
    <marker id="f5_arr2" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#a78bfa"/></marker>
    <marker id="f5_arr3" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#34d399"/></marker>
  </defs>
  <text x="80" y="66" font-size="13" fill="#64748b" font-weight="700">X (Prompt 隐状态)</text>
  <rect x="80" y="85" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="80" y="121" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/><rect x="80" y="157" width="110" height="30" rx="5" fill="#e2e8f0" stroke="#cbd5e1"/>
  <line x1="195" y1="100" x2="263" y2="100" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="195" y1="136" x2="263" y2="136" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="195" y1="172" x2="263" y2="172" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/>
  <rect x="270" y="70" width="90" height="102" rx="8" fill="url(#f5_wq)"/><text x="315" y="128" font-size="18" text-anchor="middle" fill="#fff" font-weight="700">W_Q</text>
  <rect x="270" y="200" width="90" height="102" rx="8" fill="url(#f5_wk)"/><text x="315" y="258" font-size="18" text-anchor="middle" fill="#fff" font-weight="700">W_K</text>
  <rect x="270" y="330" width="90" height="102" rx="8" fill="url(#f5_wv)"/><text x="315" y="388" font-size="18" text-anchor="middle" fill="#fff" font-weight="700">W_V</text>
  <line x1="365" y1="100" x2="443" y2="100" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="365" y1="136" x2="443" y2="136" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="365" y1="172" x2="443" y2="172" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/>
  <rect x="450" y="85" width="110" height="30" rx="5" fill="#dbeafe" stroke="#60a5fa"/><rect x="450" y="121" width="110" height="30" rx="5" fill="#dbeafe" stroke="#60a5fa"/><rect x="450" y="157" width="110" height="30" rx="5" fill="#dbeafe" stroke="#60a5fa"/>
  <text x="505" y="106" font-size="13" text-anchor="middle" fill="#1e40af" font-weight="600">q₁</text><text x="505" y="142" font-size="13" text-anchor="middle" fill="#1e40af" font-weight="600">q₂</text><text x="505" y="178" font-size="13" text-anchor="middle" fill="#1e40af" font-weight="600">q₃</text>
  <line x1="365" y1="230" x2="443" y2="230" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="365" y1="266" x2="443" y2="266" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="365" y1="302" x2="443" y2="302" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/>
  <rect x="450" y="215" width="110" height="30" rx="5" fill="#ede9fe" stroke="#a78bfa"/><rect x="450" y="251" width="110" height="30" rx="5" fill="#ede9fe" stroke="#a78bfa"/><rect x="450" y="287" width="110" height="30" rx="5" fill="#ede9fe" stroke="#a78bfa"/>
  <text x="505" y="236" font-size="13" text-anchor="middle" fill="#5b21b6" font-weight="600">k₁</text><text x="505" y="272" font-size="13" text-anchor="middle" fill="#5b21b6" font-weight="600">k₂</text><text x="505" y="308" font-size="13" text-anchor="middle" fill="#5b21b6" font-weight="600">k₃</text>
  <line x1="365" y1="360" x2="443" y2="360" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="365" y1="396" x2="443" y2="396" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/><line x1="365" y1="432" x2="443" y2="432" stroke="#94a3b8" stroke-width="1.8" marker-end="url(#f5_arr)"/>
  <rect x="450" y="345" width="110" height="30" rx="5" fill="#d1fae5" stroke="#34d399"/><rect x="450" y="381" width="110" height="30" rx="5" fill="#d1fae5" stroke="#34d399"/><rect x="450" y="417" width="110" height="30" rx="5" fill="#d1fae5" stroke="#34d399"/>
  <text x="505" y="366" font-size="13" text-anchor="middle" fill="#065f46" font-weight="600">v₁</text><text x="505" y="402" font-size="13" text-anchor="middle" fill="#065f46" font-weight="600">v₂</text><text x="505" y="438" font-size="13" text-anchor="middle" fill="#065f46" font-weight="600">v₃</text>
  <rect x="640" y="170" width="220" height="260" rx="14" fill="#fff" stroke="#e2e8f0" stroke-width="2"/>
  <text x="750" y="198" font-size="14" text-anchor="middle" fill="#334155" font-weight="700">KV Cache</text>
  <line x1="565" y1="266" x2="633" y2="240" stroke="#a78bfa" stroke-width="2" marker-end="url(#f5_arr2)"/>
  <line x1="565" y1="396" x2="633" y2="380" stroke="#34d399" stroke-width="2" marker-end="url(#f5_arr3)"/>
  <text x="680" y="228" font-size="12" fill="#7c3aed" font-weight="700">K</text>
  <rect x="660" y="238" width="56" height="38" rx="5" fill="#ede9fe" stroke="#a78bfa"/><text x="688" y="262" font-size="12" text-anchor="middle" fill="#5b21b6">k₁</text>
  <rect x="724" y="238" width="56" height="38" rx="5" fill="#ede9fe" stroke="#a78bfa"/><text x="752" y="262" font-size="12" text-anchor="middle" fill="#5b21b6">k₂</text>
  <rect x="788" y="238" width="56" height="38" rx="5" fill="#ede9fe" stroke="#a78bfa"/><text x="816" y="262" font-size="12" text-anchor="middle" fill="#5b21b6">k₃</text>
  <text x="680" y="358" font-size="12" fill="#059669" font-weight="700">V</text>
  <rect x="660" y="368" width="56" height="38" rx="5" fill="#d1fae5" stroke="#34d399"/><text x="688" y="392" font-size="12" text-anchor="middle" fill="#065f46">v₁</text>
  <rect x="724" y="368" width="56" height="38" rx="5" fill="#d1fae5" stroke="#34d399"/><text x="752" y="392" font-size="12" text-anchor="middle" fill="#065f46">v₂</text>
  <rect x="788" y="368" width="56" height="38" rx="5" fill="#d1fae5" stroke="#34d399"/><text x="816" y="392" font-size="12" text-anchor="middle" fill="#065f46">v₃</text>
  <text x="640" y="455" font-size="12" fill="#64748b">全部历史 K、V 一次性写入</text>
</svg>

**图 5：Prefill 阶段。Prompt 整体并行前向，Q、K、V 同时得出，K 与 V 写入缓存。**

&emsp;&emsp;该阶段并行度高、算术强度大，GPU 计算单元利用率充分，通常是耗时的主要部分之一（与 Prompt 长度成正比）。其输出除首 token 外，还包括一份已填充的 KV Cache，供后续 Decode 复用。

---

## 6. Decode 阶段：单 token 增量计算

&emsp;&emsp;Prefill 完成后进入 Decode 阶段。此后模型每步仅处理一个新 token，属于访存密集型（memory-bound）操作。

&emsp;&emsp;设缓存中已有 $k_{1:3}, v_{1:3}$，当前待处理 token 为 $t_4$：

1. 对 $t_4$ 计算 $q_4, k_4, v_4$；
2. 将 $k_4, v_4$ 追加至缓存，得到 $K_{1:4}, V_{1:4}$；
3. 计算 $\mathrm{softmax}(q_4K_{1:4}^\top/\sqrt{d_h})\,V_{1:4}$，得到输出并预测 $t_5$。

<svg width="920" height="630" viewBox="0 0 920 630" xmlns="http://www.w3.org/2000/svg" font-family="'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <linearGradient id="f6_model" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#4338ca"/></linearGradient>
    <linearGradient id="f6_out" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#fef3c7"/><stop offset="100%" stop-color="#fbbf24"/></linearGradient>
    <filter id="f6_sh" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.15"/></filter>
    <marker id="f6_arr" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#94a3b8"/></marker>
    <marker id="f6_arr2" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#f97316"/></marker>
  </defs>
  <text x="480" y="28" font-size="12" text-anchor="middle" fill="#f97316" font-weight="700">当前 token</text>
  <rect x="440" y="38" width="80" height="46" rx="10" fill="#ffedd5" stroke="#f97316" stroke-width="2" filter="url(#f6_sh)"/><text x="480" y="68" font-size="16" text-anchor="middle" fill="#9a3412" font-weight="700">t₄</text>
  <line x1="480" y1="84" x2="480" y2="115" stroke="#f97316" stroke-width="2" marker-end="url(#f6_arr2)"/>
  <rect x="390" y="120" width="180" height="50" rx="12" fill="url(#f6_model)" filter="url(#f6_sh)"/><text x="480" y="151" font-size="14" text-anchor="middle" fill="#fff" font-weight="700">QKV 投影</text>
  <path d="M480,170 C480,205 150,195 150,226" fill="none" stroke="#60a5fa" stroke-width="2" marker-end="url(#f6_arr)"/>
  <rect x="110" y="230" width="80" height="44" rx="9" fill="#dbeafe" stroke="#60a5fa" stroke-width="1.5" filter="url(#f6_sh)"/><text x="150" y="258" font-size="15" text-anchor="middle" fill="#1e40af" font-weight="700">q₄</text>
  <text x="150" y="296" font-size="12" text-anchor="middle" fill="#64748b">仅 Q 需实时计算</text>
  <path d="M480,170 C480,230 730,235 730,305" fill="none" stroke="#f97316" stroke-width="2" marker-end="url(#f6_arr2)"/>
  <text x="580" y="248" font-size="12" fill="#f97316" font-weight="600">k₄、v₄ 追加至缓存</text>
  <rect x="260" y="270" width="540" height="150" rx="14" fill="#fff" stroke="#e2e8f0" stroke-width="2"/>
  <text x="280" y="295" font-size="14" fill="#334155" font-weight="700">KV Cache</text>
  <rect x="290" y="310" width="100" height="45" rx="5" fill="#ede9fe" stroke="#a78bfa"/><text x="340" y="338" font-size="13" text-anchor="middle" fill="#5b21b6">k₁</text>
  <rect x="290" y="355" width="100" height="45" rx="5" fill="#d1fae5" stroke="#34d399"/><text x="340" y="383" font-size="13" text-anchor="middle" fill="#065f46">v₁</text>
  <rect x="420" y="310" width="100" height="45" rx="5" fill="#ede9fe" stroke="#a78bfa"/><text x="470" y="338" font-size="13" text-anchor="middle" fill="#5b21b6">k₂</text>
  <rect x="420" y="355" width="100" height="45" rx="5" fill="#d1fae5" stroke="#34d399"/><text x="470" y="383" font-size="13" text-anchor="middle" fill="#065f46">v₂</text>
  <rect x="550" y="310" width="100" height="45" rx="5" fill="#ede9fe" stroke="#a78bfa"/><text x="600" y="338" font-size="13" text-anchor="middle" fill="#5b21b6">k₃</text>
  <rect x="550" y="355" width="100" height="45" rx="5" fill="#d1fae5" stroke="#34d399"/><text x="600" y="383" font-size="13" text-anchor="middle" fill="#065f46">v₃</text>
  <rect x="680" y="310" width="100" height="45" rx="5" fill="#ffedd5" stroke="#f97316" stroke-width="2"/><text x="730" y="338" font-size="13" text-anchor="middle" fill="#9a3412" font-weight="700">k₄</text>
  <rect x="680" y="355" width="100" height="45" rx="5" fill="#ffedd5" stroke="#f97316" stroke-width="2"/><text x="730" y="383" font-size="13" text-anchor="middle" fill="#9a3412" font-weight="700">v₄</text>
  <text x="730" y="298" font-size="11" text-anchor="middle" fill="#f97316" font-weight="700">新追加</text>
  <line x1="150" y1="274" x2="150" y2="465" stroke="#60a5fa" stroke-width="2" marker-end="url(#f6_arr)"/>
  <line x1="530" y1="420" x2="530" y2="465" stroke="#94a3b8" stroke-width="2" marker-end="url(#f6_arr)"/>
  <rect x="110" y="470" width="660" height="60" rx="14" fill="#f8fafc" stroke="#cbd5e1" stroke-width="2"/>
  <text x="440" y="497" font-size="14" text-anchor="middle" fill="#334155" font-weight="700">Attention：softmax(q₄ · K₁:₄ᵀ) · V₁:₄</text>
  <text x="440" y="517" font-size="12" text-anchor="middle" fill="#64748b">q₄ 与全部 4 个 K 计算权重，再对 4 个 V 加权求和</text>
  <line x1="480" y1="530" x2="480" y2="558" stroke="#f59e0b" stroke-width="2" marker-end="url(#f6_arr)"/>
  <rect x="440" y="562" width="80" height="46" rx="10" fill="url(#f6_out)" stroke="#f59e0b" stroke-width="2" filter="url(#f6_sh)"/><text x="480" y="592" font-size="16" text-anchor="middle" fill="#78350f" font-weight="700">t₅</text>
</svg>

**图 6：Decode 阶段。每步仅计算单个 token 的 Q、K、V，K/V 追加进缓存，Q 与全部历史 K 做注意力。**

&emsp;&emsp;需要强调的是，Decode 每步的注意力仍需读取全部历史 K、V，因此其开销随序列长度线性增长，但增长的是**访存**而非**计算**——这正是 Decode 阶段受显存带宽制约的根源。

---

## 7. 复杂度与显存开销

### 7.1 计算复杂度

&emsp;&emsp;记序列长度为 $n$、隐藏维度为 $d$、层数为 $L$。单层单步的主要开销来自线性投影与注意力：

| 项目 | 无 KV Cache（第 $t$ 步） | 有 KV Cache（第 $t$ 步） |
|---|---|---|
| QKV 投影 | $O(t \cdot d^2)$ | $O(d^2)$ |
| 注意力打分与聚合 | $O(t^2 \cdot d)$ | $O(t \cdot d)$ |
| 输出投影 / FFN | $O(t \cdot d^2)$ | $O(d^2)$ |

&emsp;&emsp;对 $t = 1 \dots n$ 求和，完整生成 $n$ 个 token 的总开销为：

$$
\text{无缓存}: O(n^2 d^2 + n^3 d)
\qquad\Longrightarrow\qquad
\text{有缓存}: O(n\,d^2 + n^2 d)
$$

&emsp;&emsp;当 $n \gg d$ 时，注意力项占主导，规模从 $O(n^3)$ 降至 $O(n^2)$；当 $n \ll d$ 时，线性层项占主导，规模从 $O(n^2d^2)$ 降至 $O(nd^2)$。两种情形下 KV Cache 均带来数量级上的改善。

<svg width="920" height="420" viewBox="0 0 920 420" xmlns="http://www.w3.org/2000/svg" font-family="'PingFang SC','Microsoft YaHei',sans-serif">
  <defs>
    <linearGradient id="f7_red" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#f87171"/><stop offset="100%" stop-color="#dc2626"/></linearGradient>
    <linearGradient id="f7_green" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#4ade80"/><stop offset="100%" stop-color="#16a34a"/></linearGradient>
  </defs>
  <line x1="100" y1="360" x2="880" y2="360" stroke="#cbd5e1" stroke-width="1.5"/>
  <line x1="100" y1="60" x2="100" y2="360" stroke="#cbd5e1" stroke-width="1.5"/>
  <line x1="100" y1="300" x2="880" y2="300" stroke="#e2e8f0" stroke-dasharray="4,4"/>
  <line x1="100" y1="240" x2="880" y2="240" stroke="#e2e8f0" stroke-dasharray="4,4"/>
  <line x1="100" y1="180" x2="880" y2="180" stroke="#e2e8f0" stroke-dasharray="4,4"/>
  <line x1="100" y1="120" x2="880" y2="120" stroke="#e2e8f0" stroke-dasharray="4,4"/>
  <line x1="100" y1="60" x2="880" y2="60" stroke="#e2e8f0" stroke-dasharray="4,4"/>
  <text x="88" y="364" font-size="11" text-anchor="end" fill="#94a3b8">0</text>
  <text x="88" y="304" font-size="11" text-anchor="end" fill="#94a3b8">20</text>
  <text x="88" y="244" font-size="11" text-anchor="end" fill="#94a3b8">40</text>
  <text x="88" y="184" font-size="11" text-anchor="end" fill="#94a3b8">60</text>
  <text x="88" y="124" font-size="11" text-anchor="end" fill="#94a3b8">80</text>
  <text x="88" y="64" font-size="11" text-anchor="end" fill="#94a3b8">100</text>
  <text x="55" y="210" font-size="12" fill="#64748b" transform="rotate(-90 55 210)" text-anchor="middle">累计 token 级计算次数</text>
  <rect x="116" y="357" width="45" height="3" rx="2" fill="url(#f7_red)"/>
  <rect x="169" y="357" width="45" height="3" rx="2" fill="url(#f7_green)"/>
  <text x="165" y="382" font-size="12" text-anchor="middle" fill="#475569" font-weight="600">1</text>
  <rect x="246" y="345" width="45" height="15" rx="2" fill="url(#f7_red)"/>
  <rect x="299" y="351" width="45" height="9" rx="2" fill="url(#f7_green)"/>
  <text x="295" y="382" font-size="12" text-anchor="middle" fill="#475569" font-weight="600">2</text>
  <rect x="376" y="318" width="45" height="42" rx="2" fill="url(#f7_red)"/>
  <rect x="429" y="342" width="45" height="18" rx="2" fill="url(#f7_green)"/>
  <text x="425" y="382" font-size="12" text-anchor="middle" fill="#475569" font-weight="600">3</text>
  <rect x="506" y="270" width="45" height="90" rx="2" fill="url(#f7_red)"/>
  <rect x="559" y="330" width="45" height="30" rx="2" fill="url(#f7_green)"/>
  <text x="555" y="382" font-size="12" text-anchor="middle" fill="#475569" font-weight="600">4</text>
  <rect x="636" y="195" width="45" height="165" rx="2" fill="url(#f7_red)"/>
  <rect x="689" y="315" width="45" height="45" rx="2" fill="url(#f7_green)"/>
  <text x="685" y="382" font-size="12" text-anchor="middle" fill="#475569" font-weight="600">5</text>
  <rect x="766" y="87" width="45" height="273" rx="2" fill="url(#f7_red)"/>
  <rect x="819" y="297" width="45" height="63" rx="2" fill="url(#f7_green)"/>
  <text x="815" y="382" font-size="12" text-anchor="middle" fill="#475569" font-weight="600">6</text>
  <text x="465" y="408" font-size="12" text-anchor="middle" fill="#64748b">生成的 token 数 n</text>
  <rect x="270" y="20" width="16" height="16" rx="3" fill="url(#f7_red)"/>
  <text x="292" y="33" font-size="13" fill="#475569" font-weight="600">无 Cache：每步重算全序列</text>
  <rect x="540" y="20" width="16" height="16" rx="3" fill="url(#f7_green)"/>
  <text x="562" y="33" font-size="13" fill="#475569" font-weight="600">有 Cache：每步仅计算新 token</text>
</svg>

**图 7：累计 token 级计算次数示意。无缓存时呈二次增长，有缓存时呈线性增长；真实 FLOPs 还需乘上每 token 的投影与注意力开销。**

### 7.2 显存开销

&emsp;&emsp;KV Cache 的代价体现在显存占用上。设层数为 $L$、注意力头数为 $n_h$、每头维度 $d_h$、序列长度为 $S$、每元素字节数为 $b$，则缓存大小为：

$$
M_{\text{cache}} = 2 \times L \times n_h \times S \times d_h \times b
$$

&emsp;&emsp;其中系数 $2$ 对应 K 与 V 各一份。以 7B 级模型（$L=32$、$n_h=32$、$d_h=128$、FP16 即 $b=2$）、序列长度 $S=4096$ 为例：

$$
2 \times 32 \times 32 \times 4096 \times 128 \times 2 \approx 2\ \text{GiB}
$$

&emsp;&emsp;该数值为**单条请求**的缓存占用。并发服务时，缓存总量随并发数与序列长度线性叠加，往往成为显存容量的主要消耗项，也是限制 batch size 与上下文长度的关键因素。

&emsp;&emsp;这一矛盾直接催生了后续一系列优化方向：以 MQA / GQA 缩减 KV 头数以压缩缓存体积，以 PagedAttention 消除分页式预分配带来的内部碎片。相关内容不在本文范围内。

---

## 8. 小结

1. 大模型推理为自回归逐 token 生成，朴素实现在每步重复计算整个序列；
2. 因果掩码保证了历史 token 的隐状态不受后续 token 影响，因此其 $K$、$V$ 在后续步骤中保持不变；
3. KV Cache 缓存历史 $K$、$V$，每步仅计算新 token 的 $q, k, v$，并将 $k, v$ 追加至缓存；
4. 推理分为 Prefill（并行处理 Prompt，计算密集）与 Decode（逐 token 生成，访存密集）两个阶段；
5. 计算开销由 $O(n^2d^2 + n^3d)$ 降至 $O(nd^2 + n^2d)$；
6. 代价是显存占用随序列长度与并发数线性增长。

KV Cache 是 LLM 推理系统中最基础的优化之一，也是理解后续调度、分页与压缩技术的必要前提。
