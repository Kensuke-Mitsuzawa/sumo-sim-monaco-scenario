import typing as ty
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

import pandas as pd
import seaborn as sns

import numpy as np

from simulation_extractions.module_matplotlib import set_matplotlib_style

import logzero
logger = logzero.logger


class MetricLabelConverter:
    @staticmethod
    def get_label(name_metric: str) -> str:
        if name_metric == 'edge_waiting_time':
            return 'Waiting time'
        elif name_metric == 'edge_count':
            return 'Vehicle count'
        elif name_metric == 'edge_density':
            return 'Density'
        else:
            raise ValueError(f'Unknown name_metric={name_metric}')


def load_array_npz(path_array: Path) -> ty.Dict[str, np.ndarray]:
    logger.info(f'Loading {path_array}')
    array = np.load(path_array)
    
    assert 'array' in array, f'{path_array} does not contain "array"'
    assert 'edge_id' in array, f'{path_array} does not contain "edge_id"'
    
    return array


def _get_sequence_values(array: np.ndarray,
                         vec_edge_id: np.ndarray,
                         edge_id: str,
                         bucket_size: int,
                         bucket_id: ty.Optional[int] = -1) -> np.ndarray:
    """
    Get the values of a specific edge in a specific bucket.
    """
    assert edge_id in vec_edge_id, f'{edge_id} not in vec_edge_id'
    assert len(vec_edge_id) == array.shape[0], (len(vec_edge_id), array.shape[0])
    
    index_edge_id = np.where(vec_edge_id == edge_id)[0][0]
    
    if bucket_id == -1:
        vec_egde = array[index_edge_id, :]
    else:
        assert isinstance(bucket_id, int) and bucket_id >= 0, bucket_id
        t_bucket_start = bucket_id * bucket_size
        t_bucket_end = (bucket_id + 1) * bucket_size        
        vec_egde = array[index_edge_id, t_bucket_start:t_bucket_end]
    # end if
    
    return vec_egde


def main(
    path_output_dir: Path,
    path_array_x_dir: Path,
    path_array_y_dir: Path,
    name_metric: str,
    edge_id: str,
    bucket_id: int,
    bucket_size: int):
    assert path_array_x_dir.exists(), f'{path_array_x_dir} does not exist'
    assert path_array_y_dir.exists(), f'{path_array_y_dir} does not exist'	
    
    assert name_metric in ('edge_waiting_time', 'edge_count', 'edge_density'), name_metric
    
    path_array_x = path_array_x_dir / f'{name_metric}.npz'
    path_array_y = path_array_y_dir / f'{name_metric}.npz'
    
    obj_array_x = load_array_npz(path_array_x)
    obj_array_y = load_array_npz(path_array_y)    
    
    vec_target_edge_x = _get_sequence_values(
        array=obj_array_x['array'],
        vec_edge_id=obj_array_x['edge_id'],
        edge_id=edge_id,
        bucket_id=bucket_id,
        bucket_size=bucket_size)
    vec_target_edge_y = _get_sequence_values(
        array=obj_array_y['array'],
        vec_edge_id=obj_array_y['edge_id'],
        edge_id=edge_id,
        bucket_id=bucket_id,
        bucket_size=bucket_size)
    logger.debug(f'sum(x)={vec_target_edge_x.sum()}, sum(y)={vec_target_edge_y.sum()}')
    
    assert len(vec_target_edge_x) == len(vec_target_edge_y), (len(vec_target_edge_x), len(vec_target_edge_y))
    
    # making a dataframe for the histogram
    df = pd.DataFrame({'x': vec_target_edge_x, 'y': vec_target_edge_y})
    df_melt = pd.melt(df)
    
    # generating the histogram
    path_png = path_output_dir / f'{name_metric}_{edge_id}_{bucket_id}.png'
    
    fig, ax = plt.subplots()
    
    # ------------------------------------    
    avg_x = df['x'].mean()
    avg_y = df['y'].mean()
    
    std_x = df['x'].std()
    std_y = df['y'].std()
    
    _label_plot = MetricLabelConverter.get_label(name_metric)
    
    sns.histplot(data=df_melt, ax=ax, hue='variable', x='value', alpha=0.5, bins=50)
    # ------------------------------------
    # x, y, title labels
    title_text = f'{_label_plot} at {edge_id}, bucket={bucket_id}\nX:AVG={avg_x:.2f} STD={std_x:.2f}\nY:AVG={avg_y:.2f} STD={std_y:.2f}'
    ax.set_title(title_text, fontsize=FONTSIZE_LABEL)
    ax.set_xlabel(_label_plot, fontsize=FONTSIZE_LABEL)
    # ax.set_ylabel('Frequency (log)', fontsize=FONTSIZE_LABEL)
    ax.set_ylabel('')
    ax.set_yscale('log')
    # ------------------------------------
    # ticks label
    ax.tick_params(axis='x', labelsize=FONTSIZE_TICKS)
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICKS)
    # ------------------------------------
    # ------------------------------------
    # saving figures
    fig.savefig(path_png, bbox_inches='tight')
    logger.debug(f'Saved {path_png}')


def test_one_road():
    path_output_dir = Path("/tmp/path_test_output_dir")
    path_array_x_dir = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/x")
    path_array_y_dir = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/y")
    name_metric = "edge_count"
    edge_id = '152763'
    # edge_id = '152777'
    bucket_id = 3
    bucket_size = 500

    path_output_dir.mkdir(parents=True, exist_ok=True)

    main(
        path_output_dir=path_output_dir,
        path_array_x_dir=path_array_x_dir,
        path_array_y_dir=path_array_y_dir,
        name_metric=name_metric,
        edge_id=edge_id,
        bucket_id=bucket_id,
        bucket_size=bucket_size
    )


if __name__ == "__main__":
    FONTSIZE_LABEL = 25
    FONTSIZE_TICKS = 25
    font = FontProperties()
    
    set_matplotlib_style()
    test_one_road()
