from typing import Any, Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d  # noqa
import numpy as np
import pandas as pd

from .data import Motion, Sequence
from .metrics import MetricsContainer


CM2INCH = 0.393701
DEFAULT_WIDTH = 15
DEFAULT_HEIGHT = 12
DEFAULT_DPI = 300


def _new_figure(is_3d: bool = False, width: float = DEFAULT_WIDTH, height: float = DEFAULT_HEIGHT,
                dpi: int = DEFAULT_DPI, **kwargs: Any) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig = plt.figure(figsize=(width * CM2INCH, height * CM2INCH), dpi=dpi, facecolor='w', edgecolor='w', **kwargs)
    if is_3d:
        ax = plt.axes(projection='3d')
    else:
        ax = fig.gca()
    return fig, ax


def _new_subplots(nrows: int, ncols: int, width: float = DEFAULT_WIDTH, height: float = DEFAULT_HEIGHT,
                  dpi: int = DEFAULT_DPI, **kwargs: Any) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(nrows, ncols, figsize=(width * CM2INCH, height * CM2INCH), dpi=dpi, facecolor='w',
                           edgecolor='w', **kwargs)
    return fig, ax


def plot_path(path: np.ndarray, **kwargs: Any) -> mpl.figure.Figure:
    """Plot 3D path."""
    fig, ax = _new_figure(is_3d=True, **kwargs)

    axis_min = np.min(path, axis=0)
    axis_max = np.max(path, axis=0)
    axis_center = (axis_max + axis_min) / 2
    axis_len = np.max((axis_max - axis_min) / 2)

    ax.plot3D(path[:, 0], path[:, 1], path[:, 2], 'r-')
    ax.plot3D([path[-1, 0]], [path[-1, 1]], [path[-1, 2]], 'ro')
    ax.plot3D([path[1, 0]], [path[1, 1]], [path[1, 2]], 'go')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(axis_center[0] - axis_len, axis_center[0] + axis_len)
    ax.set_ylim(axis_center[1] - axis_len, axis_center[1] + axis_len)
    ax.set_zlim(axis_center[2] - axis_len, axis_center[2] + axis_len)
    ax.autoscale(enable=True, axis='x', tight=True)

    return fig


def plot_motion(motion: Motion, **kwargs: Any) -> mpl.figure.Figure:
    """Plot 3D path from motion."""
    return plot_path(motion.get_path(), **kwargs)


def plot_paths(path_pred: np.ndarray, path_gt: np.ndarray, **kwargs: Any) -> mpl.figure.Figure:
    """Plot 3D paths."""
    fig, ax = _new_figure(is_3d=True, **kwargs)

    pos_min = min(np.min(path_pred), np.min(path_gt))
    pos_max = max(np.max(path_pred), np.max(path_gt))

    ax.plot3D(path_gt[:, 0], path_gt[:, 1], path_gt[:, 2], 'g-')
    ax.plot3D(path_pred[:, 0], path_pred[:, 1], path_pred[:, 2], 'r-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(pos_min, pos_max)
    ax.set_ylim(pos_min, pos_max)
    ax.set_zlim(pos_min, pos_max)
    ax.legend(['Ground Truth', 'Prediction'])

    return fig


def plot_sequence(sequence: Sequence, **kwargs: Any) -> mpl.figure.Figure:
    """Plot 3D paths from sequence."""
    return plot_paths(sequence.prediction.get_path(), sequence.ground_truth.get_path(), **kwargs)


def plot_paths_2d(path_pred: np.ndarray, path_gt: np.ndarray, **kwargs: Any)\
        -> mpl.figure.Figure:
    """Plot 2D paths."""
    fig, ax = _new_figure(is_3d=False, **kwargs)

    x_range = np.max(path_gt[:, 0]) - np.min(path_gt[:, 0])
    y_range = np.max(path_gt[:, 1]) - np.min(path_gt[:, 1])
    x_center = np.min(path_gt[:, 0]) + x_range / 2
    y_center = np.min(path_gt[:, 1]) + y_range / 2
    val_range = max(x_range, y_range) / 2 + 5

    ax.plot(path_gt[:, 0], path_gt[:, 1], '-', color=[0, 0.4470, 0.7410])
    ax.plot(path_pred[:, 0], path_pred[:, 1], '--', color=[0.8500, 0.3250, 0.0980])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_center - val_range, x_center + val_range)
    ax.set_ylim(y_center - val_range, y_center + val_range)
    ax.legend(['Ground Truth', 'Prediction'])

    return fig


def plot_sequence_2d(sequence: Sequence, **kwargs: Any) -> mpl.figure.Figure:
    """Plot 2D paths from sequence."""
    return plot_paths_2d(sequence.prediction.get_path(), sequence.ground_truth.get_path(), **kwargs)


def plot_error_over_time(errors: MetricsContainer, **kwargs: Any) -> mpl.figure.Figure:
    """Plot translation and rotation error over time."""

    if 'width' not in kwargs:
        kwargs['width'] = 2 * DEFAULT_WIDTH
    if 'height' not in kwargs:
        kwargs['height'] = 2 * DEFAULT_HEIGHT

    fig, (ax1, ax2) = _new_subplots(2, 1, **kwargs)

    data = [[e.translation.kitti, np.rad2deg(e.rotation.kitti)] for e in errors]
    df = pd.DataFrame(data, columns=['t_err', 'r_err'])

    ax1.plot(df['t_err'])
    ax1.set_title('Translation Error')
    ax1.set_ylabel('e_t')

    ax2.plot(np.rad2deg(df['r_err']))
    ax2.set_title('Rotation Error')
    ax2.set_ylabel('e_r [deg]')

    return fig


def plot_segment_error_bars(segment_errors: Dict[str, MetricsContainer], **kwargs: Any) -> mpl.figure.Figure:
    """Plot translation and rotation error bars for multiple metrics containers."""

    t_means = [e.mean.translation.kitti * 100 for e in segment_errors.values()]
    t_std = [e.std.translation.kitti * 100 for e in segment_errors.values()]
    r_means = [np.rad2deg(e.mean.rotation.kitti) for e in segment_errors.values()]
    r_std = [np.rad2deg(e.std.rotation.kitti) for e in segment_errors.values()]

    fig, ax1 = _new_subplots(1, 1, **kwargs)
    ax2 = ax1.twinx()

    ind = np.arange(len(segment_errors))  # the x locations for the groups
    width = 0.35  # the width of the bars
    color1 = 'tab:blue'
    color2 = 'tab:orange'

    ax1.bar(ind, t_means, width, bottom=0, yerr=t_std, color=color1)
    ax2.bar(ind + width, r_means, width, bottom=0, yerr=r_std, color=color2)

    ax1.set_title('Errors by Dataset')
    ax1.set_xticks(ind + width / 2)
    ax1.set_xticklabels(segment_errors.keys())

    ax1.set_ylabel('Translation [%]', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2.set_ylabel('Rotation [deg/m]', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.autoscale_view()

    return fig


def plot_kitti_errors(segment_metrics: MetricsContainer, **kwargs: Any) -> mpl.figure.Figure:
    """Plot errors as proposed by the KITTI Odometry evaluation."""

    if len(segment_metrics) == 0:
        f = plt.figure()
        return f

    if 'width' not in kwargs:
        kwargs['width'] = 2 * DEFAULT_WIDTH
    if 'height' not in kwargs:
        kwargs['height'] = 2 * DEFAULT_HEIGHT

    data = [[e.segment_length, e.speed, e.translation.kitti, e.rotation.kitti] for e in segment_metrics]
    df = pd.DataFrame(data, columns=['seg_len', 'speed', 't_err', 'r_err'])

    # errors by segment length
    df_seg_len = df.groupby('seg_len')[['t_err', 'r_err']].mean()

    # errors by speed
    df_cut = pd.cut(df['speed'], np.linspace(np.min(df['speed']), np.max(df['speed']), 12), duplicates='drop')
    df_speed = df.groupby(df_cut)[['t_err', 'r_err']].mean()
    df_speed['speed'] = [interval.mid for interval in df_speed.index]

    # plot
    fig, ((ax1, ax2), (ax3, ax4)) = _new_subplots(2, 2, sharex='row', sharey='col', **kwargs)

    h1, = ax1.plot(df_seg_len.index, df_seg_len['t_err'] * 100, 'o-')
    ax1.set_xlabel('Path Length [m]')
    ax1.set_ylabel('Translation Error [%]')
    ax1.legend([h1], ['Translation Error'])

    h2, = ax2.plot(df_seg_len.index, np.rad2deg(df_seg_len['r_err']), 'o-')
    ax2.set_xlabel('Path Length [m]')
    ax2.set_ylabel('Rotation Error [deg/m]')
    ax2.legend([h2], ['Rotation Error'])

    h3, = ax3.plot(df_speed['speed'] * 3.6, df_speed['t_err'] * 100, 'o-')
    ax3.set_xlabel('Speed [km/h]')
    ax3.set_ylabel('Translation Error [%]')
    ax3.legend([h3], ['Translation Error'])

    h4, = ax4.plot(df_speed['speed'] * 3.6, np.rad2deg(df_speed['r_err']), 'o-')
    ax4.set_xlabel('Speed [km/h]')
    ax4.set_ylabel('Rotation Error [deg/m]')
    ax4.legend([h4], ['Rotation Error'])

    fig.set_size_inches(10, 8)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    return fig
