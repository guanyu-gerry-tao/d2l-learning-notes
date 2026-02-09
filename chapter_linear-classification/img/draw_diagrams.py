"""
为第4章 notes-zh.md 生成流程图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# 全局样式
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = ['DejaVu Sans', 'Heiti TC', 'STHeiti', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
C_INPUT = '#4ECDC4'    # 青绿
C_PROCESS = '#556FB5'  # 蓝
C_OUTPUT = '#FF6B6B'   # 红
C_LOSS = '#FFE66D'     # 黄
C_ARROW = '#333333'
C_BG = '#FAFAFA'


def draw_box(ax, xy, w, h, text, color, fontsize=13, bold=False):
    box = mpatches.FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.15",
        facecolor=color, edgecolor='#333333', linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(xy[0] + w/2, xy[1] + h/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color='white' if color in [C_PROCESS, C_OUTPUT] else '#333333')


def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->', mutation_scale=20,
        color=C_ARROW, linewidth=2
    )
    ax.add_patch(arrow)


# ==============================
# 图1：Softmax 回归流程
# ==============================
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 3))
ax1.set_xlim(-0.5, 14)
ax1.set_ylim(-0.3, 2.5)
ax1.axis('off')
ax1.set_facecolor(C_BG)
fig1.patch.set_facecolor(C_BG)

ax1.set_title('Softmax Regression Data Flow', fontsize=16, fontweight='bold', pad=15)

draw_box(ax1, (0, 0.5), 1.8, 1.2, 'Input Image\n28x28', C_INPUT)
draw_arrow(ax1, (1.9, 1.1), (2.5, 1.1))
draw_box(ax1, (2.5, 0.5), 1.8, 1.2, 'Flatten\n-> 784', C_INPUT)
draw_arrow(ax1, (4.4, 1.1), (5.0, 1.1))
draw_box(ax1, (5.0, 0.5), 2.2, 1.2, 'Linear Layer\nO = XW + b', C_PROCESS)
draw_arrow(ax1, (7.3, 1.1), (7.9, 1.1))
draw_box(ax1, (7.9, 0.5), 1.8, 1.2, 'Softmax\n-> Prob', C_PROCESS)
draw_arrow(ax1, (9.8, 1.1), (10.4, 1.1))
draw_box(ax1, (10.4, 0.5), 1.6, 1.2, 'argmax\nPredict', C_OUTPUT)
draw_arrow(ax1, (9.8, 0.5), (12.6, 0.2))
draw_box(ax1, (12.1, 0.5), 1.8, 1.2, 'Cross-Entropy\nLoss', C_LOSS, fontsize=12)

fig1.tight_layout()
fig1.savefig('/home/user/d2l-learning-notes/chapter_linear-classification/img/softmax_flow.png',
             dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close(fig1)

# ==============================
# 图2：三种分布偏移
# ==============================
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
ax2.set_xlim(-0.5, 12)
ax2.set_ylim(-0.5, 6)
ax2.axis('off')
ax2.set_facecolor(C_BG)
fig2.patch.set_facecolor(C_BG)

ax2.set_title('Three Types of Distribution Shift', fontsize=16, fontweight='bold', pad=15)

headers = ['Type', 'What Changed', 'What Stays', 'Solution']
rows = [
    ['Covariate\nShift', 'P(x) changed', 'P(y|x) same', 'Importance\nWeighting'],
    ['Label\nShift', 'P(y) changed', 'P(x|y) same', 'Confusion\nMatrix'],
    ['Concept\nShift', 'P(y|x) changed', '—', 'Retrain'],
]

col_x = [0.2, 2.8, 5.8, 8.8]
col_w = [2.2, 2.6, 2.6, 2.8]
row_h = 1.0
start_y = 4.5

# 画表头
for j, (x, w) in enumerate(zip(col_x, col_w)):
    box = mpatches.FancyBboxPatch((x, start_y), w, row_h,
                                   boxstyle="round,pad=0.05",
                                   facecolor=C_PROCESS, edgecolor='white', linewidth=1)
    ax2.add_patch(box)
    ax2.text(x + w/2, start_y + row_h/2, headers[j],
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# 画数据行
row_colors = [C_INPUT + '40', C_LOSS + '40', C_OUTPUT + '40']
for i, row in enumerate(rows):
    y = start_y - (i+1) * row_h
    colors = [row_colors[i]] * 4
    colors[0] = '#E8E8E8'
    for j, (x, w) in enumerate(zip(col_x, col_w)):
        box = mpatches.FancyBboxPatch((x, y), w, row_h,
                                       boxstyle="round,pad=0.05",
                                       facecolor=colors[j], edgecolor='#CCCCCC', linewidth=0.5)
        ax2.add_patch(box)
        ax2.text(x + w/2, y + row_h/2, row[j],
                 ha='center', va='center', fontsize=11, color='#333333')

fig2.tight_layout()
fig2.savefig('/home/user/d2l-learning-notes/chapter_linear-classification/img/distribution_shift.png',
             dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close(fig2)

print("All diagrams generated!")
