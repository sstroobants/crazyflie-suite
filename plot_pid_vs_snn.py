import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

rc_fonts = {
            "font.family": "serif",
            # "font.serif": "libertine",
            # "font.size": 20,
            # 'figure.figsize': (5, 3),
            "text.usetex": True,
            # # 'text.latex.preview': True,
            'text.latex.preamble': [r"usepackage{libertine}"],
}
font_serif = {"family": "serif"}
plt.rcParams.update(rc_fonts)

snn_datasets = []
for file in os.listdir("data/tests_snn_pos"):
    if file.endswith("hover.csv"):
        snn_datasets.append(pd.read_csv(os.path.join("data/tests_snn_pos", file), skipinitialspace=True))
        print(os.path.join("data/tests_snn_pos", file))

pid_datasets = []
for file in os.listdir("data/tests_pid_pos"):
    if file.endswith("hover.csv"):
        pid_datasets.append(pd.read_csv(os.path.join("data/tests_pid_pos", file), skipinitialspace=True))
        print(os.path.join("data/tests_pid_pos", file))

fig = plt.figure(constrained_layout=True)
# fig.suptitle('Comparison of position control for SNN vs. PID', fontsize=15)

# create 3x1 subfigs
subfigs = fig.subfigures(nrows=2, ncols=1)

# Plot snn results
top_row = subfigs[0].subplots(1, 2, sharex=True)
top_row_axin = top_row[0].inset_axes([0.20, 0.25, 0.35, 0.3])
for data in snn_datasets:
    for i in range(len(data)):
        if np.abs(data.target_x[i] - 0.5) < 0.01:
            data.timeTick = data.timeTick - data.timeTick[i] + 2000
            break 
    top_row[0].plot(data.timeTick / 1000, data.otX, alpha=0.6, color="#2c7bb6")
    top_row_axin.plot(data.timeTick / 1000, data.otX, alpha=0.6, color="#2c7bb6")
    top_row[1].plot(data.timeTick / 1000, data.otY, alpha=0.6, color="#2c7bb6")
    # plt.plot(data.timeTick, data.otZ, '.')

    top_row[0].plot(data.timeTick / 1000, data.target_x, alpha=0.6, color="#d7191c")
    top_row_axin.plot(data.timeTick / 1000, data.target_x, alpha=0.6, color="#d7191c")
    top_row[1].plot(data.timeTick / 1000, data.target_y, alpha=0.6, color="#d7191c")
subfigs[0].suptitle("SNN controller results", fontweight='bold')
top_row[0].set_title("x position", fontsize=10)
top_row[1].set_title("y position", fontsize=10)
top_row[0].set_xlim([0, 13.5])
top_row[0].set_ylim([-0.1, 0.58])
top_row[1].set_ylim([-0.05, 0.05])

top_row_axin.set_xlim(5, 8.5)
top_row_axin.set_ylim(0.48, 0.52)
top_row_axin.set_xticks([])
top_row_axin.set_yticks([])
top_row[0].indicate_inset_zoom(top_row_axin, alpha=0.7, linewidth=1.2, edgecolor='0.3')


# Plot pid results
bottom_row = subfigs[1].subplots(1, 2, sharex=True)
bottom_row_axin = bottom_row[0].inset_axes([0.20, 0.25, 0.35, 0.3])
for data in pid_datasets:
    for i in range(len(data)):
        if np.abs(data.target_x[i] - 0.5) < 0.01:
            data.timeTick = data.timeTick - data.timeTick[i] + 2000
            break 
    bottom_row[0].plot(data.timeTick / 1000, data.otX, alpha=0.6, color="#2c7bb6")
    bottom_row_axin.plot(data.timeTick / 1000, data.otX, alpha=0.6, color="#2c7bb6")
    bottom_row[1].plot(data.timeTick / 1000, data.otY, alpha=0.6, color="#2c7bb6")
    # plt.plot(data.timeTick, data.otZ, '.')

    bottom_row[0].plot(data.timeTick / 1000, data.target_x, alpha=0.6, color="#d7191c")
    bottom_row_axin.plot(data.timeTick / 1000, data.target_x, alpha=0.6, color="#d7191c")
    bottom_row[1].plot(data.timeTick / 1000, data.target_y, alpha=0.6, color="#d7191c")
subfigs[1].suptitle("Conventional PID results", fontweight='bold')
bottom_row[0].set_xlim([0, 13.5])
bottom_row[0].set_ylim([-0.1, 0.58])
bottom_row[1].set_ylim([-0.05, 0.05])
bottom_row_axin.set_xlim(5, 8.5)
bottom_row_axin.set_ylim(0.48, 0.52)
bottom_row_axin.set_xticks([])
bottom_row_axin.set_yticks([])
bottom_row[0].indicate_inset_zoom(bottom_row_axin, alpha=0.7, linewidth=1.2, edgecolor='0.3')

for ax in bottom_row:
    ax.set_xlabel("T [s]", loc='left')
    ax.xaxis.set_label_coords(0.47, -0.025)

for ax in [top_row[0], bottom_row[0]]:
    ax.set_ylabel("pos. [m]", loc='top')
    ax.yaxis.set_label_coords(-0.02, 0.65)


top_row[1].tick_params(axis="y", right=True, left=False, labelright=True, labelleft=False)
ticklabels = ["" if round(i, 2) not in [-0.04, 0.00, 0.04] else str(round(i, 2)) for i in top_row[1].get_yticks()]
top_row[1].set_yticks(top_row[1].get_yticks())
top_row[1].set_yticklabels(ticklabels)

bottom_row[1].tick_params(axis="y", right=True, left=False, labelright=True, labelleft=False)
ticklabels = ["" if round(i, 2) not in [-0.04, 0.00, 0.04] else str(round(i, 2)) for i in bottom_row[1].get_yticks()]
bottom_row[1].set_yticks(bottom_row[1].get_yticks())
bottom_row[1].set_yticklabels(ticklabels)

top_row[0].set_yticks(np.arange(-0.1, 0.6, 0.1))
ticklabels = ["" if round(i, 1) not in [0, 0.5] else str(round(i, 1)) for i in top_row[0].get_yticks()]
top_row[0].set_yticklabels(ticklabels)

bottom_row[0].set_yticks(np.arange(-0.1, 0.6, 0.1))
ticklabels = ["" if round(i, 1) not in [0, 0.5] else str(round(i, 1)) for i in bottom_row[0].get_yticks()]
bottom_row[0].set_yticklabels(ticklabels)


bottom_row[1].tick_params(axis="y", right=True, left=False, labelright=True, labelleft=False)

for ax in [top_row[0], top_row[1], bottom_row[0], bottom_row[1]]:
    ax.set_facecolor('#EBEBEB')
    ax.grid()
    ax.tick_params(labelsize='x-small', direction="in")
    for tick in ax.get_xticklabels():
        tick.set_fontname(font_serif["family"])



bottom_row[1].legend(['Measured', 'Target'], fancybox=True, shadow=True, loc='lower right')
plt.savefig("pid_vs_snn_x_setpoint.png", dpi=200)
plt.show()