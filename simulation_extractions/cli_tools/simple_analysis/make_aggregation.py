"""A script to make aggregation of simulation output array.

Input: You write all configurations in a toml file.
Output: JSONL file that contains a list of aggregated records.
"""

import typing as ty
from pathlib import Path
import toml
import math

import dacite
import json
import jsonlines

import numpy as np

import dataclasses

import matplotlib.pyplot as plot

import logzero
logger = logzero.logger

# ------------------------------------------------

@dataclasses.dataclass
class OutputConfig:
    path_output_resource: str
    path_output_png: str
    dir_name_x: str = 'x'
    dir_name_y: str = 'y'
    

@dataclasses.dataclass
class InputConfig:
    path_simulation_output: str
    # key name in a npz file. It refers to a vector of ids, such as lane-id or edge-id.
    key_name_lane_or_edge_id_vector: str 


@dataclasses.dataclass
class ResoruceConfig:
    input_x: InputConfig
    input_y: ty.Optional[InputConfig]
    output: OutputConfig

    
@dataclasses.dataclass
class ConfigAggregation:
    n_time_bucket: int
    threshold_aggregation_value: float
    

@dataclasses.dataclass
class Config:
    Resoruce: ResoruceConfig
    Aggregation: ConfigAggregation
    
    
# ------------------------------------------------


class AggregatedRecord(ty.NamedTuple):
    lane_id: str
    weight: float
    time_bucket: int
    label: str


def aggregation_matrix(torch_tensor: np.ndarray, aggregation_by: int) -> np.ndarray:
    n_columns = torch_tensor.shape[-1]
    result_tensor = []
    current_agg_point = aggregation_by
    # agg. between [0: current_agg_point]
    sub_tensor = torch_tensor[:, 0:aggregation_by]
    mean_vector = np.mean(sub_tensor, axis=1)
    _ = mean_vector[:, None]
    result_tensor.append(_)
    # agg. from [current_agg_point:]
    while (current_agg_point + aggregation_by) < n_columns:
        sub_tensor = torch_tensor[:, current_agg_point: (aggregation_by + current_agg_point)]
        mean_vector = np.mean(sub_tensor, axis=1)
        _ = mean_vector[:, None]  # 2d tensor into 3d tensor
        assert len(_.shape) == 2
        result_tensor.append(_)
        current_agg_point += aggregation_by
    # end while
    sub_tensor = torch_tensor[:, current_agg_point: (aggregation_by + current_agg_point)]
    mean_vector = np.mean(sub_tensor, axis=1)
    _ = mean_vector[:, None]  # 2d tensor into 3d tensor
    assert len(_.shape) == 2
    result_tensor.append(_)

    __ = np.concatenate(result_tensor, axis=1)
    if __.shape[-1] > 2:
        # comment: __.shape[-1] haappens often in DEBUG mode.
        assert __.shape[-1] == math.ceil(n_columns / aggregation_by)

    return __


def __write_out_jsonl(path_output_jsonl: Path, seq_agg_record: ty.List[AggregatedRecord]):
    # writing aggregated records into a file to a directory x-side.
    path_output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f'Writing aggregated records into {path_output_jsonl}')
    with jsonlines.open(path_output_jsonl, mode='w') as writer:
        for __agg_record in seq_agg_record:
            writer.write(__agg_record._asdict())
        # end for
    # end with
    logger.debug('Done')
    
    
def __aggregate_time_bucket(config_obj: Config, 
                            n_time_bucket: int, 
                            array_sim_agg: np.ndarray, 
                            vector_lane_id: np.ndarray) -> ty.List[AggregatedRecord]:
    seq_agg_record = []
    for __i_time_bucket in range(n_time_bucket):
        __array_time_bucket = array_sim_agg[:, __i_time_bucket]
        __arr_index  = np.where(__array_time_bucket > config_obj.Aggregation.threshold_aggregation_value)[0]
        __arr_value = __array_time_bucket[__arr_index]
        # converting index into list of lane-id
        for __arr_ind, __arr_w in zip(__arr_index, __arr_value):
            __lane_id = vector_lane_id[__arr_ind]
            
            __time_bucket_from = __i_time_bucket * config_obj.Aggregation.n_time_bucket
            __time_bucket_to = __time_bucket_from + config_obj.Aggregation.n_time_bucket
            __description_lebel = f'Average during {__time_bucket_from} until {__time_bucket_to}'
            __agg_record = AggregatedRecord(lane_id=__lane_id, weight=__arr_w, time_bucket=__i_time_bucket, label=__description_lebel)
            seq_agg_record.append(__agg_record)
        # end for
    # end for
    
    return seq_agg_record


def __plot_time_series_agg(config_obj: Config, 
                           array_sim_x: np.ndarray, 
                           vector_timestep: np.ndarray,
                           array_sim_y: ty.Optional[np.ndarray],):
    vector_label_timestamp = vector_timestep[:, 1]
    # plot a time series graph.
    _file_name: str = Path(config_obj.Resoruce.input_x.path_simulation_output).stem
    Path(config_obj.Resoruce.output.path_output_png).mkdir(parents=True, exist_ok=True)
    _f_file_png = Path(config_obj.Resoruce.output.path_output_png) / f'{_file_name}.png'
    
    
    _f, _ax = plot.subplots(nrows=1, ncols=1, figsize=(10, 6))
    # end if
    
    assert len(array_sim_x.shape) == 2, f'array_sim_x.shape={array_sim_x.shape}. Must be (n-sensors, n-timesteps)'
    
    # Calculate average and standard deviation
    average_value_x = np.mean(array_sim_x, axis=0)
    std_value_x = np.std(array_sim_x, axis=0)
    # Create a simple plot
    _ax.plot(average_value_x, label='Time Series', color='red', linestyle='--')
    # _ax.axhline(y=std_value_x, color='red', linestyle='--', label='Average')
    _min_std = average_value_x - std_value_x
    min_std = np.where(_min_std < 0, 0, _min_std)
    _ax.fill_between(range(len(average_value_x)), 
                     min_std, 
                     average_value_x + std_value_x, 
                     color='red', 
                     alpha=0.1, label='Std Deviation')

    # y-side
    if array_sim_y is not None:
        assert len(array_sim_y.shape) == 2, f'array_sim_y.shape={array_sim_y.shape}. Must be (n-sensors, n-timesteps)'
        average_value_y = np.mean(array_sim_y, axis=0)
        std_value_y = np.std(array_sim_y, axis=0)
        _ax.plot(average_value_y, label='Time Series', color='blue', linestyle='--')
        # _ax.axhline(y=std_value_y, color='blue', linestyle='--', label='Average')
        _min_std = average_value_y - std_value_y
        min_std = np.where(_min_std < 0, 0, _min_std)        
        _ax.fill_between(range(len(average_value_y)), 
                         min_std, 
                         average_value_y + std_value_y, 
                         color='blue', 
                         alpha=0.1, label='Std Deviation')
    # end if
    
    input_file_name = Path(config_obj.Resoruce.input_x.path_simulation_output).stem
    
    _ax.set_xlabel('Time')
    _ax.set_ylabel('Value')
        
    # Set the locations of the xticks
    _ax.set_xticks(range(0, len(vector_label_timestamp), 100))
    # Set the labels of the xticks
    _ax.set_xticklabels(vector_label_timestamp[::100], rotation=45)
    
    _f.suptitle(f'Metric={input_file_name}. X: blue, Y: red.')
    _f.savefig(_f_file_png, bbox_inches='tight')
    logger.debug(f'Writing a time series graph into {_f_file_png}')


def main(path_config: Path):
    assert path_config.exists(), f'Config file not found: {path_config}'
    
    _config_obj = toml.load(path_config)
    config_obj = dacite.from_dict(data_class=Config, data=_config_obj)
    
    # ------------------------------------------------
    # I load array-x.
    logger.debug('Loading simulation output array (x)')
    assert Path(config_obj.Resoruce.input_x.path_simulation_output).exists(), f'File not found: {config_obj.Resoruce.input_x.path_simulation_output}'
    d_sim_out = np.load(config_obj.Resoruce.input_x.path_simulation_output)
    
    assert 'array' in d_sim_out, f'Key "array" not found in {config_obj.Resoruce.input_x.path_simulation_output}'
    assert config_obj.Resoruce.input_x.key_name_lane_or_edge_id_vector in d_sim_out, \
        f'Key "{config_obj.Resoruce.input_x.key_name_lane_or_edge_id_vector}" not found in {config_obj.Resoruce.input_x.key_name_lane_or_edge_id_vector}'
    
    array_sim_out_x = d_sim_out['array']
    vector_lane_id = d_sim_out[config_obj.Resoruce.input_x.key_name_lane_or_edge_id_vector]
    
    logger.debug('Aggregating simulation output array')
    array_sim_agg = aggregation_matrix(array_sim_out_x, config_obj.Aggregation.n_time_bucket)
    logger.debug('Aggregation done')
    # finding a set of index that meets the threshold criteria.
    n_time_bucket = array_sim_agg.shape[-1]
    # aggregation for x-side
    agg_record_x = __aggregate_time_bucket(config_obj=config_obj, 
                            n_time_bucket=n_time_bucket, 
                            array_sim_agg=array_sim_agg, 
                            vector_lane_id=vector_lane_id)
    _file_name: str = Path(config_obj.Resoruce.input_x.path_simulation_output).stem
    _f_file_jsonl = Path(config_obj.Resoruce.output.path_output_resource) / config_obj.Resoruce.output.dir_name_x / f'{_file_name}.jsonl'
    __write_out_jsonl(
        path_output_jsonl=_f_file_jsonl,
        seq_agg_record=agg_record_x)
    ## writing simple stats into a text file.
    _f_stats_file = Path(config_obj.Resoruce.output.path_output_resource) / config_obj.Resoruce.output.dir_name_x / f'{_file_name}.txt'
    _stats_obj = dict(array_shape=array_sim_agg.shape)
    with open(_f_stats_file, 'w') as f:
        f.write(json.dumps(_stats_obj))
    # end with
    # ------------------------------------------------
    if config_obj.Resoruce.input_y is not None:
        assert config_obj.Resoruce.input_y is not None, 'config_obj.Resoruce.input_y is None'
        logger.debug('Loading simulation output array (y)')
        assert Path(config_obj.Resoruce.input_y.path_simulation_output).exists(), f'File not found: {config_obj.Resoruce.input_y.path_simulation_output}'
        d_sim_out = np.load(config_obj.Resoruce.input_y.path_simulation_output)
        
        assert 'array' in d_sim_out, f'Key "array" not found in {config_obj.Resoruce.input_y.path_simulation_output}'
        assert config_obj.Resoruce.input_y.key_name_lane_or_edge_id_vector in d_sim_out, \
            f'Key "{config_obj.Resoruce.input_y.key_name_lane_or_edge_id_vector}" not found in {config_obj.Resoruce.input_y.key_name_lane_or_edge_id_vector}'
        
        array_sim_out_y = d_sim_out['array']
        vector_lane_id = d_sim_out[config_obj.Resoruce.input_y.key_name_lane_or_edge_id_vector]
        
        logger.debug('Aggregating simulation output array')
        array_sim_agg = aggregation_matrix(array_sim_out_y, config_obj.Aggregation.n_time_bucket)
        logger.debug('Aggregation done')
        # finding a set of index that meets the threshold criteria.
        n_time_bucket = array_sim_agg.shape[-1]
        # aggregation for y-side
        agg_record_y = __aggregate_time_bucket(config_obj=config_obj, 
                                n_time_bucket=n_time_bucket, 
                                array_sim_agg=array_sim_agg, 
                                vector_lane_id=vector_lane_id)
        _file_name: str = Path(config_obj.Resoruce.input_y.path_simulation_output).stem
        _f_file_jsonl = Path(config_obj.Resoruce.output.path_output_resource) / config_obj.Resoruce.output.dir_name_y / f'{_file_name}.jsonl'
        __write_out_jsonl(
            path_output_jsonl=_f_file_jsonl,
            seq_agg_record=agg_record_y)
        ## writing simple stats into a text file.
        _f_stats_file = Path(config_obj.Resoruce.output.path_output_resource) / config_obj.Resoruce.output.dir_name_y / f'{_file_name}.txt'
        _stats_obj = dict(array_shape=array_sim_agg.shape)
        with open(_f_stats_file, 'w') as f:
            f.write(json.dumps(_stats_obj))
        # end with
        
        assert np.array_equal(array_sim_out_x, array_sim_out_y) is False, 'array_sim_out_x and array_sim_out_y are the same'
    else:
        array_sim_out_y = None
    # end if

    vector_timestamp_labels: np.ndarray = d_sim_out['timestamps']
    
    __plot_time_series_agg(config_obj=config_obj, 
                           array_sim_x=array_sim_out_x, 
                           vector_timestep=vector_timestamp_labels,
                           array_sim_y=array_sim_out_y)

    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    __args = ArgumentParser()
    __args.add_argument('--path_config', type=str, help='Path to toml visualization config file', required=True)
    __args = __args.parse_args()
    
    # _path_config = Path('/home/kensuke_mit/sumo-sim-monaco-scenario/simulation_extractions/cli_tools/simple_analysis/config_make_aggregation.toml')
    main(Path(__args.path_config))
