"""
为深度学习训练内存分析讲义生成可视化图片。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).parent / "figures"
OUTPUT.mkdir(exist_ok=True)

import matplotlib.font_manager as fm
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(_FONT_PATH).exists():
    _prop = fm.FontProperties(fname=_FONT_PATH)
    _FONT_NAME = _prop.get_name()
    # 强制指定 SC 变体
    if "CJK" in _FONT_NAME:
        _FONT_NAME = "Noto Sans CJK SC"
    fm.fontManager.addfont(_FONT_PATH)
else:
    _FONT_NAME = "sans-serif"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_FONT_NAME, "Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": 12,
    "axes.unicode_minus": False,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
})

# ── 配色 ──
C_FWD = "#3B82F6"      # 蓝 — 前向
C_BWD = "#EF4444"      # 红 — 反向
C_ACT = "#10B981"      # 绿 — activation
C_PARAM = "#8B5CF6"    # 紫 — 参数
C_GRAD = "#F59E0B"     # 橙 — 梯度
C_OPT = "#EC4899"      # 粉 — 优化器
C_TEMP = "#6B7280"     # 灰 — 临时
C_MICRO = "#06B6D4"    # 青 — micro batch


def fig_computation_graph():
    """
    计算图: 前向产生 activation, 反向消耗 activation 产生梯度。
    用一个简单 3 层网络展示。
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 5.5),
                                         gridspec_kw={"height_ratios": [1, 1]})

    layers = ["Layer 1", "Layer 2", "Layer 3"]
    n = len(layers)
    x_pos = np.linspace(1, n * 2 - 1, n)

    # ── 上半: 前向 ──
    ax_top.set_title("前向传播 (Forward Pass)", fontsize=14, fontweight="bold", pad=8)
    ax_top.set_xlim(0, n * 2)
    ax_top.set_ylim(0, 3)
    ax_top.axis("off")

    # Input
    ax_top.text(0, 1.5, "x", fontsize=13, ha="center", va="center",
                bbox=dict(boxstyle="circle", fc="#DBEAFE", ec=C_FWD))
    ax_top.annotate("", xy=(0.6, 1.5), xytext=(0.25, 1.5),
                    arrowprops=dict(arrowstyle="->", color=C_FWD, lw=2))

    # Layers + activations
    for i, (name, x) in enumerate(zip(layers, x_pos)):
        # layer box
        ax_top.add_patch(mpatches.FancyBboxPatch(
            (x - 0.35, 1.0), 0.7, 1.0,
            boxstyle="round,pad=0.1", fc="#EFF6FF", ec=C_FWD, lw=2))
        ax_top.text(x, 1.5, name, fontsize=10, ha="center", va="center", color=C_FWD)

        # activation circle below
        label = f"a{i+1}"
        ax_top.text(x, 0.25, label, fontsize=11, ha="center", va="center",
                    bbox=dict(boxstyle="circle", fc="#D1FAE5", ec=C_ACT))

        # arrow from layer to activation
        ax_top.annotate("", xy=(x, 0.7), xytext=(x, 1.0),
                        arrowprops=dict(arrowstyle="->", color=C_ACT, lw=1.5))

        # right arrow to next
        if i < n - 1:
            ax_top.annotate("", xy=(x + 1, 1.5), xytext=(x + 0.35, 1.5),
                            arrowprops=dict(arrowstyle="->", color=C_FWD, lw=2))
        else:
            # loss
            ax_top.annotate("", xy=(x + 0.7, 1.5), xytext=(x + 0.35, 1.5),
                            arrowprops=dict(arrowstyle="->", color=C_FWD, lw=2))
            ax_top.text(x + 0.7, 1.5, "loss", fontsize=11, ha="center", va="center",
                        bbox=dict(boxstyle="circle", fc="#FEE2E2", ec=C_BWD))

    # annotation
    ax_top.annotate("", xy=(x_pos[-1] + 1.3, 2.5), xytext=(x_pos[-1] + 1.3, 0.25),
                    arrowprops=dict(arrowstyle="<->", color=C_ACT, lw=1.5, linestyle="dashed"))
    ax_top.text(x_pos[-1] + 1.3, 1.4, "需要保存\nactivation", fontsize=8, color=C_ACT,
                ha="center", va="center")

    # ── 下半: 反向 ──
    ax_bot.set_title("反向传播 (Backward Pass)", fontsize=14, fontweight="bold", pad=8)
    ax_bot.set_xlim(0, n * 2)
    ax_bot.set_ylim(0, 3)
    ax_bot.axis("off")

    # Grad from loss
    ax_bot.text(n * 2 - 0.7, 1.5, "∂ℓ/∂a₃\n(梯度)", fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="#FEE2E2", ec=C_BWD))

    for i, (name, x) in reversed(list(enumerate(zip(layers, x_pos)))):
        # backward arrow
        if i < n - 1:
            ax_bot.annotate("", xy=(x + 0.35, 1.5), xytext=(x + 1, 1.5),
                            arrowprops=dict(arrowstyle="->", color=C_BWD, lw=2))

        # layer box
        ax_bot.add_patch(mpatches.FancyBboxPatch(
            (x - 0.35, 1.0), 0.7, 1.0,
            boxstyle="round,pad=0.1", fc="#FEF2F2", ec=C_BWD, lw=2))
        ax_bot.text(x, 1.5, name, fontsize=10, ha="center", va="center", color=C_BWD)

        # 使用 activation (从上面借)
        ax_bot.annotate("", xy=(x, 0.25), xytext=(x, 0.7),
                        arrowprops=dict(arrowstyle="->", color=C_ACT, lw=1.5, linestyle="dotted"))
        ax_bot.text(x, 0.25, f"a{i+1}\n(from fwd)", fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round", fc="#D1FAE5", ec=C_ACT, alpha=0.7))

    ax_bot.annotate("", xy=(0.6, 1.5), xytext=(1, 1.5),
                    arrowprops=dict(arrowstyle="->", color=C_BWD, lw=2))
    ax_bot.text(0, 1.5, "∂ℓ/∂W₁\n(梯度)", fontsize=9, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="#FEF3C7", ec=C_GRAD))

    # legend
    patches = [
        mpatches.Patch(color=C_FWD, label="前向传播"),
        mpatches.Patch(color=C_BWD, label="反向传播"),
        mpatches.Patch(color=C_ACT, label="Activation (需保留)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    path = OUTPUT / "fig_computation_graph.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def fig_memory_timeline():
    """
    显存随时间变化: 前向积累 activation → 反向逐步释放 → optimizer step。
    分别画 micro_bs=1 和 micro_bs=4 的对比。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # 模拟时间步
    t = np.linspace(0, 100, 500)

    def make_profile(t, act_peak, act_layers, has_accum=False):
        """
        时间线: [0-40] fwd 阶段, [40-80] bwd 阶段, [80-100] update.
        act_peak: activation 峰值
        act_layers: 层数
        """
        # param (constant)
        param = np.full_like(t, 20)
        # grad (appears during bwd, kept until update)
        grad = np.where((t >= 40) & (t < 80), 20, 0)
        if has_accum:
            # in accum mode, grad stays between micro-batches
            grad = np.where((t >= 40) & (t < 85), 20, 0)
        # optimizer state (constant after init)
        opt = np.full_like(t, 40)
        # activation
        act = np.zeros_like(t)
        idx_fwd = t < 40
        idx_bwd = (t >= 40) & (t < 80)
        # fwd: linearly accumulate activations layer by layer
        act[idx_fwd] = act_peak * (t[idx_fwd] / 40)
        # bwd: release activations layer by layer (reverse)
        progress_bwd = (t[idx_bwd] - 40) / 40
        act[idx_bwd] = act_peak * (1 - progress_bwd)
        return param, grad, opt, act

    # ── 左: micro_bs=1 (peak act = 5) ──
    param, grad, opt, act = make_profile(t, act_peak=5, act_layers=32)
    total = param + grad + opt + act

    ax1.fill_between(t, 0, param, label="参数 (Weights)", color=C_PARAM, alpha=0.85)
    ax1.fill_between(t, param, param + grad, label="梯度 (Gradients)", color=C_GRAD, alpha=0.85)
    ax1.fill_between(t, param + grad, param + grad + opt,
                     label="优化器状态 (Adam)", color=C_OPT, alpha=0.85)
    ax1.fill_between(t, param + grad + opt, total,
                     label="Activation", color=C_ACT, alpha=0.85)
    ax1.plot(t, total, color="black", lw=1.5, label="峰值显存")

    ax1.set_title("Micro Batch = 1\n(小 activation)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("时间 (一个训练 step)")
    ax1.set_ylabel("显存 (GB)")
    ax1.set_xticks([0, 40, 80, 100])
    ax1.set_xticklabels(["开始", "前向结束", "反向结束", "更新"])
    ax1.axvline(40, color="gray", ls="--", lw=0.8)
    ax1.axvline(80, color="gray", ls="--", lw=0.8)

    # annotation
    ax1.text(20, total[:500].max() * 0.6, "前向\naccumulate\nactivation",
             ha="center", fontsize=8, color=C_ACT)
    ax1.text(60, total[:500].max() * 0.6, "反向\nrelease\nactivation",
             ha="center", fontsize=8, color=C_BWD)

    # ── 右: micro_bs=4 (peak act = 20) ──
    param, grad, opt, act = make_profile(t, act_peak=20, act_layers=32)
    total2 = param + grad + opt + act

    ax2.fill_between(t, 0, param, label="参数", color=C_PARAM, alpha=0.85)
    ax2.fill_between(t, param, param + grad, label="梯度", color=C_GRAD, alpha=0.85)
    ax2.fill_between(t, param + grad, param + grad + opt, label="优化器状态", color=C_OPT, alpha=0.85)
    ax2.fill_between(t, param + grad + opt, total2, label="Activation", color=C_ACT, alpha=0.85)
    ax2.plot(t, total2, color="black", lw=1.5, label="峰值显存")

    ax2.set_title("Micro Batch = 4\n(大 activation)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("时间 (一个训练 step)")
    ax2.set_xticks([0, 40, 80, 100])
    ax2.set_xticklabels(["开始", "前向结束", "反向结束", "更新"])
    ax2.axvline(40, color="gray", ls="--", lw=0.8)
    ax2.axvline(80, color="gray", ls="--", lw=0.8)

    ax2.text(20, total2[:500].max() * 0.5, "activation\n4x 更大",
             ha="center", fontsize=8, color=C_ACT)
    ax2.text(60, total2[:500].max() * 0.5, "显存峰值\n更高",
             ha="center", fontsize=8, color=C_BWD)

    # annotate peak
    peak1 = total[:250].max()
    peak2 = total2[:250].max()
    ax1.annotate(f"峰值 = {peak1:.0f} GB", xy=(40, peak1),
                 xytext=(25, peak1 + 5), fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color="black"))
    ax2.annotate(f"峰值 = {peak2:.0f} GB", xy=(40, peak2),
                 xytext=(25, peak2 + 5), fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color="black"))

    fig.legend(handles=[
        mpatches.Patch(color=C_PARAM, label="参数"),
        mpatches.Patch(color=C_GRAD, label="梯度"),
        mpatches.Patch(color=C_OPT, label="优化器 (Adam)"),
        mpatches.Patch(color=C_ACT, label="Activation"),
    ], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.12))

    plt.tight_layout()
    path = OUTPUT / "fig_memory_timeline.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def fig_gradient_accumulation():
    """
    Gradient Accumulation 示意图:
    多个 micro batch 依次 fwd+bwd, 累加梯度, 最后一步更新参数.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    n_micro = 4
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_micro))

    ax.set_xlim(0, n_micro * 3 + 1)
    ax.set_ylim(0, 4)
    ax.axis("off")

    for i in range(n_micro):
        x_start = i * 3 + 0.3
        # FWD block
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_start, 1.5), 1.0, 1.5,
            boxstyle="round,pad=0.05", fc="#DBEAFE", ec=C_FWD))
        ax.text(x_start + 0.5, 2.25, f"Fwd {i+1}",
                ha="center", fontsize=10, color=C_FWD, fontweight="bold")

        # BWD block
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_start + 1.2, 1.5), 1.0, 1.5,
            boxstyle="round,pad=0.05", fc="#FEE2E2", ec=C_BWD))
        ax.text(x_start + 1.7, 2.25, f"Bwd {i+1}",
                ha="center", fontsize=10, color=C_BWD, fontweight="bold")

        # arrow
        if i < n_micro - 1:
            ax.annotate("", xy=(x_start + 2.8, 2.25),
                        xytext=(x_start + 2.2, 2.25),
                        arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1))

    # gradient accumulation annotation
    y_line = 1.0
    for i in range(n_micro):
        x_c = i * 3 + 1.8
        if i == 0:
            ax.text(x_c, y_line, "grad₁", fontsize=8, ha="center", color=C_GRAD)
        elif i == n_micro - 1:
            ax.text(x_c, y_line,
                    f"grad₁+₂+₃+₄\n(4×)", fontsize=8, ha="center", color=C_GRAD)
        else:
            ax.text(x_c, y_line,
                    f"grad₁+...+{i+1}", fontsize=7, ha="center", color=C_GRAD)

    # optimizer step
    x_opt = n_micro * 3 + 0.3
    ax.add_patch(mpatches.FancyBboxPatch(
        (x_opt, 1.5), 1.2, 1.5,
        boxstyle="round,pad=0.05", fc="#EDE9FE", ec=C_PARAM))
    ax.text(x_opt + 0.6, 2.25, "Update\n(一步)",
            ha="center", fontsize=10, color=C_PARAM, fontweight="bold")

    ax.annotate("", xy=(x_opt, 2.25), xytext=(n_micro * 3 - 0.2, 2.25),
                arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1))

    # Accumulation indicator
    ax.annotate("", xy=(n_micro * 3 - 0.5, 1.0), xytext=(0.5, 1.0),
                arrowprops=dict(arrowstyle="<->", color=C_GRAD, lw=1.5, linestyle="dashed"))

    # Time axis
    ax.text(0, 0.3, "时间 →", fontsize=11, ha="center", color="black")
    ax.annotate("", xy=(0.3, 0.3), xytext=(n_micro * 3 + 1.2, 0.3),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Activation memory annotation
    for i in range(n_micro):
        x_c = i * 3 + 0.8
        # activation bubble (small, micro batch sized)
        act_h = 0.3 if i == 0 else 0.25
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_c - 0.3, 3.1), 0.6, act_h,
            boxstyle="round,pad=0.02", fc="#D1FAE5", ec=C_ACT, alpha=0.7))
    ax.text(n_micro * 1.2, 3.5, "Activation: 每步只用 micro_batch 大小, 用完即释放",
            ha="center", fontsize=9, color=C_ACT)

    # Grad annotation
    ax.text(n_micro * 1.2, 0.6, "梯度: 累加 n_micro 次后再更新",
            ha="center", fontsize=9, color=C_GRAD)

    ax.set_title("Gradient Accumulation: 显存 vs 等效 Batch Size",
                 fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    path = OUTPUT / "fig_gradient_accumulation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def fig_memory_pie():
    """
    显存构成饼图: LLaMA-7B, micro_bs=1 vs micro_bs=4.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    params_gb = 14
    grads_gb = 14
    optim_gb = 56

    # micro_bs=1: act = 32 layers * 1*2048*4096*2B*3 ≈ 1.6 GB
    act1_gb = 1.6
    total1 = params_gb + grads_gb + optim_gb + act1_gb

    sizes1 = [params_gb, grads_gb, optim_gb, act1_gb]
    labels1 = [
        f"参数\n{params_gb} GB ({params_gb/total1*100:.0f}%)",
        f"梯度\n{grads_gb} GB ({grads_gb/total1*100:.0f}%)",
        f"优化器\n{optim_gb} GB ({optim_gb/total1*100:.0f}%)",
        f"Activation\n{act1_gb} GB ({act1_gb/total1*100:.0f}%)",
    ]
    colors_pie = [C_PARAM, C_GRAD, C_OPT, C_ACT]

    wedges1, texts1 = ax1.pie(
        sizes1, labels=labels1, colors=colors_pie, startangle=90,
        textprops={"fontsize": 9})
    ax1.set_title(f"Micro BS=1: 总计 {total1:.0f} GB", fontsize=12, fontweight="bold")

    # micro_bs=4: act = 4x = 6.4 GB
    act4_gb = 6.4
    total4 = params_gb + grads_gb + optim_gb + act4_gb
    sizes4 = [params_gb, grads_gb, optim_gb, act4_gb]
    labels4 = [
        f"参数\n{params_gb} GB ({params_gb/total4*100:.0f}%)",
        f"梯度\n{grads_gb} GB ({grads_gb/total4*100:.0f}%)",
        f"优化器\n{optim_gb} GB ({optim_gb/total4*100:.0f}%)",
        f"Activation\n{act4_gb} GB ({act4_gb/total4*100:.0f}%)",
    ]

    wedges2, texts2 = ax2.pie(
        sizes4, labels=labels4, colors=colors_pie, startangle=90,
        textprops={"fontsize": 9})
    ax2.set_title(f"Micro BS=4: 总计 {total4:.0f} GB", fontsize=12, fontweight="bold")

    fig.suptitle("LLaMA-7B 显存构成 (fp16, Adam, seq=2048)",
                 fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    path = OUTPUT / "fig_memory_pie.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def fig_backprop_detail():
    """
    详细反向传播: 放大单层的计算图。
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ═══ 前向 (上半) ═══
    # Input
    ax.text(1, 5.2, "xᵢ\n(输入)", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#DBEAFE", ec=C_FWD))

    # Weight
    ax.text(3, 5.2, "W\n(参数)", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#EDE9FE", ec=C_PARAM))

    # Multiply
    ax.annotate("", xy=(2.2, 5.2), xytext=(1.6, 5.2),
                arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1))
    ax.annotate("", xy=(3.8, 5.2), xytext=(3.4, 5.2),
                arrowprops=dict(arrowstyle="->", color=C_TEMP, lw=1))

    # output
    ax.text(5, 5.2, "W·xᵢ\n(线性)", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#FEF3C7", ec=C_GRAD))

    ax.annotate("", xy=(5.8, 5.2), xytext=(5.4, 5.2),
                arrowprops=dict(arrowstyle="->", color=C_FWD, lw=1.5))

    # activation function
    ax.text(7, 5.2, "σ(·)\n(激活)", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#EFF6FF", ec=C_FWD))

    ax.annotate("", xy=(7.8, 5.2), xytext=(7.4, 5.2),
                arrowprops=dict(arrowstyle="->", color=C_FWD, lw=1.5))

    # Output activation
    ax.text(9, 5.2, "xᵢ₊₁\n= aᵢ", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="circle", fc="#D1FAE5", ec=C_ACT))

    # ── 公式: xᵢ₊₁ = σ(W·xᵢ) ──
    ax.text(5, 4.2, "前向:  xᵢ₊₁ = σ(W · xᵢ)     ← 存 xᵢ (给反向用)",
            fontsize=11, ha="center", color=C_FWD,
            bbox=dict(boxstyle="round", fc="#EFF6FF", ec=C_FWD, alpha=0.7))

    # ═══ 反向 (下半) ═══
    # grad from above
    ax.text(9, 2.5, "∂ℓ/∂xᵢ₊₁\n(上层梯度)", fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#FEE2E2", ec=C_BWD))

    # Chain rule arrow
    ax.annotate("", xy=(7.8, 2.5), xytext=(9.5, 2.5),
                arrowprops=dict(arrowstyle="->", color=C_BWD, lw=1.5))

    # activation backward
    ax.text(7, 2.5, "σ'(·)\n×", fontsize=10, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#FEF2F2", ec=C_BWD))

    # gradient w.r.t. linear output
    ax.text(5, 2.5, "∂ℓ/∂z\n(中间)", fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#FEF3C7", ec=C_GRAD))

    ax.annotate("", xy=(5.8, 2.5), xytext=(6.4, 2.5),
                arrowprops=dict(arrowstyle="->", color=C_BWD, lw=1.5))

    # split: toward W and toward x
    # toward W: ∂ℓ/∂W = ∂ℓ/∂z · xᵢ
    ax.annotate("", xy=(3.5, 1.2), xytext=(5, 2.0),
                arrowprops=dict(arrowstyle="->", color=C_GRAD, lw=1.5, connectionstyle="arc3,rad=0.3"))
    ax.text(3, 0.7, "∂ℓ/∂W =\n ∂ℓ/∂z · xᵢ\n(梯度, 存下来更新参数)",
            fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#FEF3C7", ec=C_GRAD, alpha=0.8))

    # toward x: ∂ℓ/∂xᵢ = ∂ℓ/∂z · Wᵀ
    ax.annotate("", xy=(1, 1.2), xytext=(4.2, 2.0),
                arrowprops=dict(arrowstyle="->", color=C_BWD, lw=1.5, connectionstyle="arc3,rad=-0.3"))
    ax.text(1, 0.7, "∂ℓ/∂xᵢ =\n ∂ℓ/∂z · Wᵀ\n(传到下一层)",
            fontsize=9, ha="center", va="center",
            bbox=dict(boxstyle="round", fc="#FEE2E2", ec=C_BWD, alpha=0.8))

    # Dotted lines connecting xᵢ to backward
    ax.annotate("", xy=(1, 1.2), xytext=(1, 4.0),
                arrowprops=dict(arrowstyle="<->", color=C_ACT, lw=1, linestyle="dotted"))
    ax.text(0.5, 2.8, "xᵢ\n(前向\n存的!)", fontsize=8, ha="center", color=C_ACT)

    # Formula
    ax.text(5, -0.2,
            "反向:  ∂ℓ/∂W = ∂ℓ/∂z · xᵢ    需要 xᵢ (前向存)!\n"
            "       ∂ℓ/∂xᵢ = Wᵀ · ∂ℓ/∂z   传到下一层继续反向",
            fontsize=11, ha="center", color=C_BWD,
            bbox=dict(boxstyle="round", fc="#FEF2F2", ec=C_BWD, alpha=0.7))

    ax.set_title("单层计算图: 前向产生 activation, 反向消耗 + 产生梯度",
                 fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    path = OUTPUT / "fig_backprop_detail.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path.name}")
    return path


def main():
    print("生成讲义可视化图片...")
    fig_computation_graph()
    fig_memory_timeline()
    fig_gradient_accumulation()
    fig_memory_pie()
    fig_backprop_detail()
    print(f"\n全部完成! 图片保存在 {OUTPUT}/")
    print(f"共 {len(list(OUTPUT.glob('*.png')))} 张图片")


if __name__ == "__main__":
    main()
