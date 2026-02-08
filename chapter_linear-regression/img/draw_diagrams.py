"""
为第2章 notes.md 生成流程图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# 全局样式
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = ['Heiti TC', 'STHeiti', 'Songti SC', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 颜色方案
C_INPUT = '#4ECDC4'    # 青绿
C_PROCESS = '#556FB5'  # 蓝
C_OUTPUT = '#FF6B6B'   # 红
C_LOSS = '#FFE66D'     # 黄
C_ARROW = '#333333'
C_BG = '#FAFAFA'


def draw_box(ax, xy, w, h, text, color, fontsize=13, bold=False):
    """画一个圆角矩形 + 居中文字"""
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
    """画箭头"""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->', mutation_scale=20,
        color=C_ARROW, linewidth=2
    )
    ax.add_patch(arrow)


# ==============================
# 图1：线性回归流程
# ==============================
fig1, ax1 = plt.subplots(1, 1, figsize=(12, 2.5))
ax1.set_xlim(-0.5, 11.5)
ax1.set_ylim(-0.3, 2.3)
ax1.axis('off')
ax1.set_facecolor(C_BG)
fig1.patch.set_facecolor(C_BG)

ax1.set_title('线性回归：数据流程', fontsize=16, fontweight='bold', pad=15)

draw_box(ax1, (0, 0.5), 2.2, 1.2, '输入特征\n(面积, 房龄)', C_INPUT)
draw_arrow(ax1, (2.3, 1.1), (3.0, 1.1))
draw_box(ax1, (3.0, 0.5), 2.8, 1.2, '加权求和 + 偏置\ny = w·x + b', C_PROCESS)
draw_arrow(ax1, (5.9, 1.1), (6.6, 1.1))
draw_box(ax1, (6.6, 0.5), 2.0, 1.2, '预测值\n(房价)', C_OUTPUT)
draw_arrow(ax1, (8.7, 1.1), (9.4, 1.1))
draw_box(ax1, (9.4, 0.5), 2.0, 1.2, 'MSE 损失\n|预测-真实|²', C_LOSS)

fig1.tight_layout()
fig1.savefig('/Users/guanyutao/developers/learning-projects/d2l-zh/pytorch/chapter_linear-networks/img/linear_regression_flow.png',
             dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close(fig1)


# ==============================
# 图2：Softmax 回归流程
# ==============================
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 3))
ax2.set_xlim(-0.5, 14)
ax2.set_ylim(-0.3, 2.5)
ax2.axis('off')
ax2.set_facecolor(C_BG)
fig2.patch.set_facecolor(C_BG)

ax2.set_title('Softmax 回归：数据流程', fontsize=16, fontweight='bold', pad=15)

draw_box(ax2, (0, 0.5), 1.8, 1.2, '输入图片\n28×28', C_INPUT)
draw_arrow(ax2, (1.9, 1.1), (2.5, 1.1))
draw_box(ax2, (2.5, 0.5), 1.8, 1.2, 'Flatten\n展平为784', C_INPUT)
draw_arrow(ax2, (4.4, 1.1), (5.0, 1.1))
draw_box(ax2, (5.0, 0.5), 2.2, 1.2, '全连接层\nO = XW + b', C_PROCESS)
draw_arrow(ax2, (7.3, 1.1), (7.9, 1.1))
draw_box(ax2, (7.9, 0.5), 1.8, 1.2, 'Softmax\n→ 概率', C_PROCESS)
draw_arrow(ax2, (9.8, 1.1), (10.4, 1.1))
draw_box(ax2, (10.4, 0.5), 1.6, 1.2, 'argmax\n预测类别', C_OUTPUT)
draw_arrow(ax2, (9.8, 0.5), (12.6, 0.2))
draw_box(ax2, (12.1, 0.5), 1.8, 1.2, '交叉熵\n损失', C_LOSS, fontsize=12)

fig2.tight_layout()
fig2.savefig('/Users/guanyutao/developers/learning-projects/d2l-zh/pytorch/chapter_linear-networks/img/softmax_regression_flow.png',
             dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close(fig2)


# ==============================
# 图3：训练循环
# ==============================
fig3, ax3 = plt.subplots(1, 1, figsize=(8, 9))
ax3.set_xlim(-1, 9)
ax3.set_ylim(-0.5, 10.5)
ax3.axis('off')
ax3.set_facecolor(C_BG)
fig3.patch.set_facecolor(C_BG)

ax3.set_title('通用训练循环', fontsize=16, fontweight='bold', pad=15)

# 步骤从上往下
steps = [
    (9.5, '初始化权重\n(随机)', C_INPUT),
    (8.3, '取一个 mini-batch', C_INPUT),
    (7.1, '前向传播\n计算预测值', C_PROCESS),
    (5.9, '计算损失\n预测 vs 真实', C_LOSS),
    (4.7, '反向传播\n计算梯度', C_PROCESS),
    (3.5, '更新权重\nw = w - lr × 梯度', C_PROCESS),
    (2.3, '清零梯度', '#AAAAAA'),
    (1.0, '评估测试集精度', C_OUTPUT),
]

bw, bh = 3.0, 0.9
bx = 3.0

for y, text, color in steps:
    draw_box(ax3, (bx, y), bw, bh, text, color, fontsize=11)

# 箭头
for i in range(len(steps) - 1):
    y_from = steps[i][0]
    y_to = steps[i+1][0]
    draw_arrow(ax3, (bx + bw/2, y_from), (bx + bw/2, y_to + bh))

# 循环箭头 (从清零梯度回到取mini-batch)
ax3.annotate('', xy=(bx - 0.3, steps[1][0] + bh/2), xytext=(bx - 0.3, steps[6][0] + bh/2),
             arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2.5))
ax3.text(bx - 1.0, (steps[1][0] + steps[6][0]) / 2 + bh/2, '遍历所有\nmini-batch',
         fontsize=10, color='#E74C3C', ha='center', va='center', fontweight='bold')

# 外层循环箭头 (从评估回到取mini-batch)
ax3.annotate('', xy=(bx + bw + 0.3, steps[1][0] + bh/2), xytext=(bx + bw + 0.3, steps[7][0] + bh/2),
             arrowprops=dict(arrowstyle='->', color='#3498DB', lw=2.5))
ax3.text(bx + bw + 1.1, (steps[1][0] + steps[7][0]) / 2 + bh/2, '重复\n多个 epoch',
         fontsize=10, color='#3498DB', ha='center', va='center', fontweight='bold')

fig3.tight_layout()
fig3.savefig('/Users/guanyutao/developers/learning-projects/d2l-zh/pytorch/chapter_linear-networks/img/training_loop.png',
             dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close(fig3)


# ==============================
# 图4：两种任务对比
# ==============================
fig4, ax4 = plt.subplots(1, 1, figsize=(12, 5))
ax4.set_xlim(-0.5, 12)
ax4.set_ylim(-0.5, 6)
ax4.axis('off')
ax4.set_facecolor(C_BG)
fig4.patch.set_facecolor(C_BG)

ax4.set_title('线性回归 vs Softmax 回归', fontsize=16, fontweight='bold', pad=15)

# 表格数据
headers = ['', '线性回归', 'Softmax 回归']
rows = [
    ['任务',     '预测一个数字',         '预测属于哪个类别'],
    ['输出',     '1 个数',              '每类的概率 (q 个)'],
    ['最后一步', '直接输出',             'softmax → 概率'],
    ['损失函数', 'MSE (均方误差)',       '交叉熵'],
    ['梯度',     'ŷ - y',              'ŷ - y  (一样！)'],
]

col_x = [0.5, 3.5, 7.5]
col_w = [2.5, 3.5, 4.0]
row_h = 0.8
start_y = 4.5

# 画表头
for j, (x, w, h) in enumerate(zip(col_x, col_w, [2.5, 3.5, 4.0])):
    box = mpatches.FancyBboxPatch((x, start_y), w, row_h,
                                   boxstyle="round,pad=0.05",
                                   facecolor=C_PROCESS, edgecolor='white', linewidth=1)
    ax4.add_patch(box)
    ax4.text(x + w/2, start_y + row_h/2, headers[j],
             ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# 画数据行
for i, row in enumerate(rows):
    y = start_y - (i+1) * row_h
    colors = ['#E8E8E8', C_INPUT + '40', C_OUTPUT + '40']
    for j, (x, w) in enumerate(zip(col_x, col_w)):
        box = mpatches.FancyBboxPatch((x, y), w, row_h,
                                       boxstyle="round,pad=0.05",
                                       facecolor=colors[j], edgecolor='#CCCCCC', linewidth=0.5)
        ax4.add_patch(box)
        ax4.text(x + w/2, y + row_h/2, row[j],
                 ha='center', va='center', fontsize=11, color='#333333')

fig4.tight_layout()
fig4.savefig('/Users/guanyutao/developers/learning-projects/d2l-zh/pytorch/chapter_linear-networks/img/comparison.png',
             dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.close(fig4)

print("✅ 4 张图全部生成完毕！")
