import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# -------------------------- 数据准备 --------------------------
data_line1 = [16, 22, 24, 33.3333333333333, 34.3333333333333, 32.6666666666667, 33, 27.6666666666667,
              31.6666666666667, 30.6666666666667, 40.3333333333333, 63.3333333333333, 79.3333333333333,
              94, 84.3333333333333, 83.6666666666667, 68.3333333333333, 69.6666666666667, 52, 49.6666666666667,
              40.3333333333333, 40, 34.6666666666667, 26, 19.3333333333333, 14.3333333333333, 11.6666666666667,
              15, 15.6666666666667, 13, 8.66666666666667, 8, 9.33333333333333, 7, 4.33333333333333, 1.33333333333333,
              1, 1.33333333333333, 2, 1.66666666666667, 1]

data_raw_line2 = [70.1727258193047,	88.1331821650300,	108.757784123991,	131.734281543839,	156.611971697599,	182.778878778860,
                  209.454344482400,	235.722888626591,	260.584845177191,	283.019839465277,	302.057449150463,	316.848099919765,
                  326.726846239023,	331.263279463901,	330.292324070310,	323.922905526305,	312.524063394450,	296.690651728240,
                  277.192945819693,	254.915972708733,	230.795042070479,	205.753757066402,	180.649846608865,	156.232694631677,
                  133.114712684139,	111.756974871488,	92.4680328525633,	75.4137074752976,	60.6349855428010,	48.0709309675604,
                  37.5836841435809,	28.9830685693453,	22.0489320973057,	16.5500103932628,	12.2587217295971,	8.96182291080628,
                  6.46724328566021,	4.60766071459319,	3.24150275263179,	2.25207286736600,	1.54544493514479]

data_raw_line_up = [69.1320491368076,	87.4427762914709,	109.250420109407,	133.964088487886,	161.206021744617,	190.409864102634,	220.800944752782,
                    251.421832215194,	281.178548532815,	308.904604749834,	333.437737199188,	353.702213297743,	368.788424753196,	378.021424004558,
                    381.011134963761,	377.679038939138,	368.258891737229,	353.272050337054,	333.480828145983,	309.825557205957,	283.352430647849,
                    255.139602405210,	226.228470919889,	197.565747352459,	169.960079577017,	144.054981858694,	120.317899893210,	99.0436537793117,
                    80.3693921730169,	64.2976102842068,	50.7237001865271,	39.4648205542370,	30.2874668792629,	22.9318574930314,	17.1320043310373,
                    12.6310172121036,	9.19173717891338,	6.60318190130000,	4.68351544117555,	3.28034631015570,	2.26914299265707]

data_raw_line_down = [67.5952375440126,	80.1563720445003,	94.0131775627538,	108.304075053944,	122.529402476046,	136.163168582209,	148.657783515098,
                      159.480837082155,	168.153421163576,	174.285830606376,	177.606877425553,	177.983724373985,	175.430232677653,	170.103163166323,
                      162.286967438570,	152.369151855998,	140.809122236885,	128.103912891126,	114.754237128174,	101.233908385032,	87.9649715319908,
                      75.2999880852569,	63.5119811261709,	52.7916942588841,	43.2511491335245,	34.9320507172332,	27.8173987608010,	21.8446921790555,
                      16.9193103087619,	12.9269595824229,	9.74442451989414,	7.24820554071534,	5.32092462036781,	3.85561084222346,	2.75813371991036,
                      1.94813667799200,	1.35884786598603,	0.936125934878493,	0.637050493762622,	0.428304939556148,	0.284534455578546]

data_raw_line_change = [70.3520678709719,	87.9593463720889,	107.939727314970,	130.019854052904,	153.721782607817,	178.420944601713,	203.342652802824,
                        227.598922724115,	250.240222568011,	270.317813918270,	286.950902491749,	299.391845559022,	307.082575066982,	309.696241183891,
                        307.159748710817,	299.655099367589,	287.599925804678,	271.609939840278,	252.447887864899,	230.964774430296,	208.039472218819,
                        184.522409153256,	161.187954166092,	138.698634170114,	117.582660752877,	98.2246676161123,	80.8682508566149,	65.6279846406646,
                        52.5080984602636,	41.4249240262494,	32.2304741814618,	24.7349987785773,	18.7269624514870,	13.9895069468084,	10.3130192135186,
                        7.50387656806161,	5.38975993152148,	3.82211598069672,	2.67642567439871,	1.85092558702075,	1.26435815666694]

factor = 0.2247

data_line2 = np.array(data_raw_line2) * factor
data_line_up = np.array(data_raw_line_up) * factor
data_line_down = np.array(data_raw_line_down) * factor
data_line_change = np.array(data_raw_line_change) * factor

time_points = np.arange(1, len(data_line1) + 1)

data_line1_cumsum = np.cumsum(data_line1)
data_line2_cumsum = np.cumsum(data_line2)
data_line_up_cumsum = np.cumsum(data_line_up)
data_line_down_cumsum = np.cumsum(data_line_down)
data_line_change_cumsum = np.cumsum(data_line_change)

# -------------------------- 中文显示配置 --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 1.0

# -------------------------- 创建并排组图 --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=False)

# -------------------------- 第一个图：原始趋势图 --------------------------
# 关键修改：用 y=-0.12 把标题移到子图下方，pad=10 控制标题与图的间距
ax1.set_title('(a)', fontsize=18, fontweight='bold', pad=10, y=-0.18)

# Real data 仅显示散点，无连线
ax1.scatter(time_points, data_line1, color='#1f77b4', s=36, alpha=0.8,
            edgecolors='white', linewidth=0.5, label='Data', zorder=5)
# Best fit model 虚线
ax1.plot(time_points, data_line2, linestyle='--', color='#2ca02c', linewidth=5,
         label='Model estimation', zorder=4)

# 添加上下限之间的阴影填充
ax1.fill_between(time_points, data_line_down, data_line_up,
                 color='#87CEEB', alpha=0.2, zorder=1, label='95% Credible interval')

# 上下限曲线设为白色
ax1.plot(time_points, data_line_up, linestyle='--', color='white', linewidth=3, zorder=2)
ax1.plot(time_points, data_line_down, linestyle='--', color='white', linewidth=3, zorder=2)

# 第一个图美化
ax1.set_xlabel('t', fontsize=14, fontweight='bold')
ax1.set_ylabel('New weekly cases', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12, loc='upper right', frameon=True, fancybox=True, shadow=True)
ax1.set_xlim(0, len(time_points) + 1)
ax1.set_ylim(bottom=0)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# 移除右侧和上方的边框
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# -------------------------- 第二个图：累加趋势图 --------------------------
# 关键修改：同上，标题移到下方
ax2.set_title('(b)', fontsize=18, fontweight='bold', pad=10, y=-0.18)

ax2.scatter(time_points, data_line1_cumsum, color='#1f77b4', s=36, alpha=0.8,
            edgecolors='white', linewidth=0.5, label='Data', zorder=5)
ax2.plot(time_points, data_line_change_cumsum, linestyle='--', color='#2ca02c', linewidth=5,
         label='Model estimation', zorder=4)

# 添加上下限之间的阴影填充
ax2.fill_between(time_points, data_line_down_cumsum, data_line_up_cumsum,
                 color='#87CEEB', alpha=0.2, zorder=1, label='95% Credible interval')

# 上下限曲线设为白色
ax2.plot(time_points, data_line_up_cumsum, linestyle='--', color='white', linewidth=3, zorder=2)
ax2.plot(time_points, data_line_down_cumsum, linestyle='--', color='white', linewidth=3, zorder=2)

# 第二个图美化
ax2.set_xlabel('t', fontsize=14, fontweight='bold')
ax2.set_ylabel('Cumulative weekly cases', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12, loc='lower right', frameon=True, fancybox=True, shadow=True)
ax2.set_xlim(0, len(time_points) + 1)
ax2.set_ylim(bottom=0)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

# 移除右侧和上方的边框
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 关键修改：调整底部边距（bottom从默认0.1改为0.15），避免标题被截断
plt.tight_layout()
plt.subplots_adjust(top=0.88, wspace=0.15, bottom=0.15)

# 保存高质量图片
plt.savefig('trend_figure_sci.pdf', dpi=300, bbox_inches='tight', format='pdf')
plt.savefig('trend_figure_sci.png', dpi=300, bbox_inches='tight', format='png')

# 显示图形
plt.show()
