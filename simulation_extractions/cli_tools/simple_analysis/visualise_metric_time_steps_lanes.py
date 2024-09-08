import xml.etree.ElementTree as ET
import typing as ty
from pathlib import Path
import logzero

import tqdm

import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from simulation_extractions.module_matplotlib import set_matplotlib_style

import pandas as pd


logzero.loglevel(logzero.INFO)
# logzero.loglevel(logzero.DEBUG)
logger = logzero.logger


class LaneObservationContainer(ty.NamedTuple):
    lane_id: str
    edge_id: str
    time_begin: float
    time_end: float
    count_vehicles: float
    density: float


def parse_lane_observation(path_lane_observation: Path
                           ) -> ty.List[LaneObservationContainer]:
    """Parse the lane-observation file.
    """
    seq_lane_observations = []
    
    iter_elems = ET.iterparse(path_lane_observation.as_posix())
    for event, _elem in tqdm.tqdm(iter_elems):
        # _iter_interval_node = _elem.findall('interval')
        # for _node_interval in _iter_interval_node:
        if _elem.tag == 'interval':
            _time_step_begin = float(_elem.attrib['begin'])
            _time_step_end = float(_elem.attrib['end'])
            # extracting lane-nodes
            _iter_lane_node = _elem.findall('*/lane')
            for _node_lane in _iter_lane_node:
                _lane_id = _node_lane.attrib['id']
                if '_' in _lane_id:
                    _edge_id = _lane_id.split('_')[0]
                else:
                    _edge_id = ''
                # end if
                
                # collecting metrics
                if 'sampledSeconds' in _node_lane.attrib:
                    _count = float(_node_lane.attrib['sampledSeconds'])
                else:
                    _count = 0.0
                # end if
                if 'laneDensity' in _node_lane.attrib:
                    _density = float(_node_lane.attrib['laneDensity'])
                else:
                    _density = 0.0
                # end if

                _obs_container = LaneObservationContainer(
                    lane_id=_lane_id,
                    edge_id=_edge_id,
                    time_begin=_time_step_begin,
                    time_end=_time_step_end,
                    count_vehicles=_count,
                    density=_density)
                seq_lane_observations.append(_obs_container)
            # end for
        # end for
    # end for
    return seq_lane_observations


def parse_edge_observation(path_edge_observation: Path
                           ) -> ty.List[LaneObservationContainer]:
    """Parse the lane-observation file.
    """
    seq_lane_observations = []
    
    iter_elems = ET.iterparse(path_edge_observation.as_posix())
    for event, _elem in tqdm.tqdm(iter_elems):
        # _iter_interval_node = _elem.findall('interval')
        # for _node_interval in _iter_interval_node:
        if _elem.tag == 'interval':
            _time_step_begin = float(_elem.attrib['begin'])
            _time_step_end = float(_elem.attrib['end'])
            # extracting lane-nodes
            _iter_lane_node = _elem.findall('edge')
            for _node_lane in _iter_lane_node:
                _edge_id = _node_lane.attrib['id']
                
                # collecting metrics
                if 'sampledSeconds' in _node_lane.attrib:
                    _count = float(_node_lane.attrib['sampledSeconds'])
                else:
                    _count = 0.0
                # end if
                if 'laneDensity' in _node_lane.attrib:
                    _density = float(_node_lane.attrib['laneDensity'])
                else:
                    _density = 0.0
                # end if

                _obs_container = LaneObservationContainer(
                    lane_id='',
                    edge_id=_edge_id,
                    time_begin=_time_step_begin,
                    time_end=_time_step_end,
                    count_vehicles=_count,
                    density=_density)
                seq_lane_observations.append(_obs_container)
            # end for
        # end for
    # end for
    return seq_lane_observations


from datetime import datetime, timedelta
def __get_real_time(t: int,
                    default_date: str = '2021-01-01') -> datetime:
    # Parse the default date
    base_datetime = datetime.strptime(default_date, '%Y-%m-%d')
    
    # Add the simulation time (in seconds) to the base datetime
    real_time = base_datetime + timedelta(seconds=t)
    return real_time


import matplotlib.dates as mdates

def plot_lane_observation(path_save_png: Path, 
                          lane_id: str, 
                          seq_lane_observations_x: ty.List[LaneObservationContainer],
                          seq_lane_observations_y: ty.List[LaneObservationContainer],
                          size_bucket: int = 5000,
                          t_start: int = 14400,
                          t_end: int = 50400,
                          is_x_axis_real_time: bool = True):
    # sorting by time
    seq_lane_observations_x.sort(key=lambda x: x.time_begin)
    seq_lane_observations_y.sort(key=lambda x: x.time_begin)

    # plotting. Use Pandas for this
    dict_count_x = [dict(count=o.count_vehicles, time=o.time_begin, density=o.density) for o in seq_lane_observations_x]
    dict_count_y = [dict(count=o.count_vehicles, time=o.time_begin, density=o.density) for o in seq_lane_observations_y]

    df_x = pd.DataFrame(dict_count_x)
    df_y = pd.DataFrame(dict_count_y)
    
    # merging into a single dataframe
    if len(df_y) > 0:
        df_data = pd.merge(df_x, df_y, on='time', suffixes=('_x', '_y'), how='outer')
        df_count = pd.melt(df_data, id_vars=['time'], value_vars=['count_x', 'count_y'])
    else:
        df_x.rename(columns={'count': 'count_x', 'density': 'density_x'}, inplace=True)
        df_data = df_x
        df_count = pd.melt(df_data, id_vars=['time'], value_vars=['count_x'])        
    # end if
    
    # adding synthetic rows the starting and edning time.
    if t_start not in df_count.time:
        synthetic_row = pd.DataFrame([{'time': t_start, 'variable': 0, 'value': 0}])
        df_count = pd.concat([df_count, synthetic_row], ignore_index=True)
    if t_end not in df_count.time:
        synthetic_row = pd.DataFrame([{'time': t_end, 'variable': 0, 'value': 0}])
        df_count = pd.concat([df_count, synthetic_row], ignore_index=True)
    # end if
    
    df_count['real_time'] = df_count['time'].apply(lambda x: __get_real_time(x))
    df_count.fillna(0, inplace=True)

    # path of the count png
    path_png_count = path_save_png / f'{lane_id}.png'
    # plot count
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    if is_x_axis_real_time:
        _label_x = 'real_time'
    else:
        _label_x = 'time'
    # end if
    
    sns.lineplot(x=_label_x, 
                 y='value', 
                 hue='variable', 
                 data=df_count, 
                 ax=ax, 
                 alpha=0.6,
                 legend=False,
                 drawstyle='steps-post')
    
    for _t in range(t_start, t_end, size_bucket):
        if is_x_axis_real_time:
            ax.axvline(x=__get_real_time(_t), color='green', alpha=0.5)
            # Set the x-axis date format
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H-%M'))    
            # Rotate the x-axis labels for better readability
            plt.xticks(rotation=45)
        else:
            ax.axvline(x=_t, color='green', alpha=0.5)
            plt.xticks(rotation=90)
        # end if
    # end for
    

    ax.set_title(f'Count vehicles: {lane_id}')
    ax.set_xlabel('')
    ax.set_ylabel('Count vehicles')
    # Set y-axis limit to start at 0
    ax.set_ylim(bottom=0)    
    f.savefig(path_png_count.as_posix(), bbox_inches='tight')
    logger.info(f'Saved: {path_png_count}')


def main(path_lane_observation_x: Path,
         path_lane_observation_y: Path,
         path_dir_output: Path,
         vis_by_lane: bool = True,
         label_type_x_axis: str = 'real_time'):
    assert label_type_x_axis in ['simulation_time', 'real_time'], f'Invalid label_type_x_axis: {label_type_x_axis}'
    
    if vis_by_lane:
        seq_lane_observations_x = parse_lane_observation(path_lane_observation_x)
        seq_lane_observations_y = parse_lane_observation(path_lane_observation_y)
    else:
        seq_lane_observations_x = parse_edge_observation(path_lane_observation_x)
        seq_lane_observations_y = parse_edge_observation(path_lane_observation_y)
    # end if

    # grouping by with the lane-id
    dict_lane_observation_x = {}
    dict_lane_observation_y = {}
    if vis_by_lane:
        for _lane_id, _g_obj in itertools.groupby(sorted(seq_lane_observations_x, key=lambda x: x.lane_id), key=lambda x: x.lane_id):
            dict_lane_observation_x[_lane_id] = list(_g_obj)
        # end for
        for _lane_id, _g_obj in itertools.groupby(sorted(seq_lane_observations_y, key=lambda x: x.lane_id), key=lambda x: x.lane_id):
            dict_lane_observation_y[_lane_id] = list(_g_obj)
        # end for
    else:
        for _edge_id, _g_obj in itertools.groupby(sorted(seq_lane_observations_x, key=lambda x: x.edge_id), key=lambda x: x.edge_id):
            dict_lane_observation_x[_edge_id] = list(_g_obj)
        # end for
        for _edge_id, _g_obj in itertools.groupby(sorted(seq_lane_observations_y, key=lambda x: x.edge_id), key=lambda x: x.edge_id):
            dict_lane_observation_y[_edge_id] = list(_g_obj)
        # end for
    # end if

    logger.info(f'Number of lanes X: {len(dict_lane_observation_x)}, Number of lanes Y: {len(dict_lane_observation_y)}')
 
    # plotting observation per lane
    logger.info(f'Plotting observations...')
    for _lane_id in dict_lane_observation_x.keys():
        _seq_obs_x = dict_lane_observation_x.get(_lane_id, [])
        _seq_obs_y = dict_lane_observation_y.get(_lane_id, [])
        
        plot_lane_observation(path_dir_output, _lane_id, _seq_obs_x, _seq_obs_y)
    # end for


def test():
    # _path_lane_observation_x = Path('/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/sumo_cfg/0_debug/x/out/most.lane-observation.xml')
    # _path_lane_observation_y = Path('/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/sumo_cfg/0_debug/y/out/most.lane-observation.xml')
    _path_lane_observation_x = Path('/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/sumo_cfg/0_debug/x/out/most.edge-observation.xml')
    _path_lane_observation_y = Path('/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/sumo_cfg/0_debug/y/out/most.edge-observation.xml')


    _path_dir_output = Path('/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/simple-png/0/edge-observations/count')
    _path_dir_output.mkdir(parents=True, exist_ok=True)
    main(_path_lane_observation_x, _path_lane_observation_y, _path_dir_output, vis_by_lane=False)



if __name__ == '__main__':
    set_matplotlib_style(is_use_latex=False)
    test()