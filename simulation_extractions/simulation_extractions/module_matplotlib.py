import matplotlib.pyplot as plt
import matplotlib


def set_matplotlib_style(is_use_latex: bool = True):
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 20

    BIG_TITLE_SIZE = 20

    font = {'size'   : 22}

    matplotlib.rc('font', **font)
    matplotlib.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    if is_use_latex:
        matplotlib.rcParams['text.usetex'] = True
