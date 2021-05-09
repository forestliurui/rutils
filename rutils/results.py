import sys
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(filename, metric, line_condition=None, skip_line=0, last_line=None):
    """
    Parse metric from all lines from filename that contain the string line_condition

    For each satisfying line, extract the first numerical substring (i.e., containing only digit and '.') appearing after metric   
    """
    line_cond = line_condition
    f = open(filename, "r")
    line_idx = -1
    val_list = []

    for line in f:
        line_idx += 1 
        if line_idx < skip_line:
            # any line before skip_line will be ignored
            continue
        if last_line is not None and line_idx > last_line:
            # any line beyond last_line will be ignored
            continue

        if line_cond is None or line_cond in line:
            idx = line.find(metric)
            if idx < 0:
                continue
            idx += len(metric)
            val = [float(s) for s in re.findall(r'-?\d+\.?\d*', line[idx:])][0]
            val_list.append(val)
    return val_list


def plot_2d_curve(filename, data, x_label, y_label, colors, markers):
    """
    plot 2d curve from data, using colors and markers
    save the figure to pdf file with name as filename

    @data: a dictionary mapping from curve_name to a dict {'x': [], 'y': []}
    @colors: a dictionary mapping from curve_name to color
    @markers: a dictionary mapping form curve_name to marker
    
    Examples:
    data = {"gspar": {'x': [1,2,3,4], 'y':[0.1, 0.3, 0.5, 0.6]}, 
            "bandit": {'x': [1,2,3,4,5], 'y': [-2, -4, -6, -8, -10]}
           }
    x_label = 'Number of Iterations'
    y_label = 'Loss Value'
    colors = {"gspar": "mediumslateblue", "bandit": "tomato", "terngrad": "darkmagenta", 'qsgd': "silver", "atomo_svd": "c"}
    markers = {"bandit": "o", "gspar": "v", "terngrad": "s", "qsgd": "P", "atomo_svd": "^"}
    """

    plt.figure(figsize = (20, 13))
    curve_names = list(data.keys())
    
    for name in curve_names:
        x_axis = data[name]['x']
        y_axis = data[name]['y']
        plt.plot(x_axis, y_axis, '-', color=colors[name], marker=markers[name], linewidth=10, markersize=20)

    plt.xlabel(x_label, fontsize=50)
    plt.ylabel(y_label, fontsize=50)

    plt.xticks(fontsize=40, rotation=20)
    plt.yticks(fontsize=40)
    plt.legend(curve_names, fontsize=45)
    plt.tight_layout()
    plt.savefig(filename)

def plot_bar(filename, data, group_ticks, x_label, y_label, colors):
    """
    plot the bar chart from data

    @data: a dictionary mapping from bar_name to a list of values
    @group_ticks: a list consisting of labels for each group
    @x_label: a string for label along x-axis
    @y_label: a string for label along y-axis
    @colors: a dictionary mapping from bar_name to color

    Examples:
    data = {'gspar': [0.0043, 0.0085, 0.0237, 0.066],
            'bandit': [0.0039, 0.00838, 0.0235, 0.0687],
           }
    group_ticks = [2, 4, 8, 16]
    x_label = 'Number of Workers'
    y_label = 'Communication Time'
    colors = {"gspar": "mediumslateblue", "bandit": "tomato", "terngrad": "darkmagenta", 'qsgd': "silver", "atomo_svd": "c"}
    """
    num_groups = len(group_ticks)
    num_bar = len(data)
    bar_names = list(data.keys())

    width = 1.0/(num_bar+1)
    ind = np.arange(1, num_groups+1)
    plt.figure(figsize = (20, 13))

    p = []
    left_loc = -(num_bar/2.0 - 0.5)
    idx = 0
    for name in bar_names:
        p.append(plt.bar(ind + (left_loc + idx)*width, data[name], width*0.9, color=colors[name]))
        idx += 1

    plt.xlabel(x_label, fontsize=50)
    plt.ylabel(y_label, fontsize=50)
    plt.xticks(ind, group_ticks, fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(p, bar_names, fontsize=45, loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)

