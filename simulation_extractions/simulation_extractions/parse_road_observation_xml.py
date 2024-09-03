import typing as ty
import xml.etree.ElementTree as ET
import logzero
import toml
import json
import numpy as np
import csv
from dataclasses import dataclass

import sumolib

from pathlib import Path

logger = logzero.logger


"""A module for parsing SUMO edge observation file and for saving the parsed data as numpy arrays.
This module can work as an independent script; See `__if__name__ == '__main__'` part.
"""


class EdgeInformation(ty.NamedTuple):
    edge_id: str
    time_begin: float
    time_end: float
    sample_count: int
    travel_time: float
    density: float
    waiting_time: float
    time_loss: float
    speed: float
# end class


def get_edge_ids(net: sumolib.net.Net) -> ty.List[str]:
    """Get the all edge id
    """
    seq_edge_obj = net.getEdges(withInternal=True)
    seq_edge_ids = []
    for _edge_obj in seq_edge_obj:
        _edge_id = _edge_obj.getID()
        seq_edge_ids.append(_edge_id)
    # end for
    return seq_edge_ids


def parse_edge_observation_and_writing_array(path_sumo_net_xml: Path,
                           path_edge_observation: Path,
                           path_work_dir: Path,
                           file_name_travel_time: str = 'edge_travel_time.npz',
                           file_name_density: str = 'edge_density.npz',
                           file_name_waiting_time: str = 'edge_waiting_time.npz',
                           file_name_time_loss: str = 'edge_time_loss.npz',
                           file_name_count: str = 'edge_count.npz',
                           file_name_speed: str = 'edge_speed.npz',
                           file_name_var_id_and_edge_id: str = 'edge_id.csv',):
    """
    A function of parsing SUMO edge file output, and of saving the arrays of the parsed data.
    This function extracts the following information from the edge observation file; See: https://sumo.dlr.de/docs/Simulation/Output/Lane-_or_Edge-based_Traffic_Measures.html#generated_output
    """
    assert path_sumo_net_xml.exists()
    assert Path(path_edge_observation).exists()
    
    net = sumolib.net.readNet(path_sumo_net_xml.as_posix())
    # note: I gurantee that the order of `seq_edge_ids` is always consistent.
    seq_edge_ids = get_edge_ids(net)

    iter_elems = ET.iterparse(path_edge_observation.as_posix())
    
    stack_edge_info = []
    count_time_interval = 0  # n of the aggregated-time step.
    stack_time_interval = []
    for event, _elem in iter_elems:
        if _elem.tag == 'interval':
            # extracting time interval
            _time_begin = _elem.attrib['begin']
            _time_end = _elem.attrib['end']
            logger.debug(f'Interval -> {_time_begin}:{_time_end}')
            count_time_interval += 1
            stack_time_interval.append((_time_begin, _time_end))
            
            # extracting edge information
            _iter_edge_info = _elem.findall('edge')
            for _edge_elem in _iter_edge_info:
                _e_info = EdgeInformation(
                    time_begin=_time_begin,
                    time_end=_time_end,
                    edge_id=_edge_elem.attrib['id'],
                    travel_time=_edge_elem.attrib.get('traveltime', 0.0),
                    density=_edge_elem.attrib.get('density', 0.0),
                    waiting_time=_edge_elem.attrib.get('waitingTime', 0.0),
                    time_loss=_edge_elem.attrib.get('timeLoss', 0.0),
                    sample_count=_edge_elem.attrib.get('sampledSeconds', 0),
                    speed=_edge_elem.attrib.get('speed', 0.0),
                )
                stack_edge_info.append(_e_info)
            # end for
        # end if
    # end for
    
    # making arrays
    vector_edge_id = np.array(seq_edge_ids)
    array_travel_time = np.zeros([len(seq_edge_ids), count_time_interval])
    array_density = np.zeros([len(seq_edge_ids), count_time_interval])
    array_waiting_time = np.zeros([len(seq_edge_ids), count_time_interval])
    array_time_loss = np.zeros([len(seq_edge_ids), count_time_interval])
    array_sample_count = np.zeros([len(seq_edge_ids), count_time_interval])
    array_speed = np.zeros([len(seq_edge_ids), count_time_interval])

    # putting values to arrays
    _vector_time_interval = np.array(['-'.join(_t) for _t in stack_time_interval])
    for _e_info in stack_edge_info:
        _j = np.where(vector_edge_id == _e_info.edge_id)[0].item()  # a row number of an output array.
        _i = np.where(_vector_time_interval == f'{_e_info.time_begin}-{_e_info.time_end}')[0].item()  # a column number of an output array.
        array_travel_time[_j, _i] = _e_info.travel_time
        array_density[_j, _i] = _e_info.density
        array_waiting_time[_j, _i] = _e_info.waiting_time
        array_time_loss[_j, _i] = _e_info.time_loss
        array_sample_count[_j, _i] = _e_info.sample_count
        array_speed[_j, _i] = _e_info.speed
    # end for

    # saving files
    path_npz_travel_time = path_work_dir / file_name_travel_time
    path_npz_density = path_work_dir / file_name_density
    path_npz_waiting_time = path_work_dir / file_name_waiting_time
    path_npz_time_loss = path_work_dir / file_name_time_loss
    path_npz_sample_count = path_work_dir / file_name_count
    path_npz_speed = path_work_dir / file_name_speed
    path_var_id_and_edge_id = path_work_dir / file_name_var_id_and_edge_id
    
    # vechile count
    save_object_array = {'array': array_sample_count, 'edge_id': seq_edge_ids, 'timestamps': stack_time_interval}
    np.savez_compressed(path_npz_sample_count, **save_object_array)
    
    # travel time
    save_object_array = {'array': array_travel_time, 'edge_id': seq_edge_ids, 'timestamps': stack_time_interval}
    np.savez_compressed(path_npz_travel_time, **save_object_array)
    
    # density
    save_object_array = {'array': array_density, 'edge_id': seq_edge_ids, 'timestamps': stack_time_interval}
    np.savez_compressed(path_npz_density, **save_object_array)
    
    # waiting time
    save_object_array = {'array': array_waiting_time, 'edge_id': seq_edge_ids, 'timestamps': stack_time_interval}
    np.savez_compressed(path_npz_waiting_time, **save_object_array)
    
    # time loss
    save_object_array = {'array': array_time_loss, 'edge_id': seq_edge_ids, 'timestamps': stack_time_interval}
    np.savez_compressed(path_npz_time_loss, **save_object_array)
    
    # speed
    save_object_array = {'array': array_speed, 'edge_id': seq_edge_ids, 'timestamps': stack_time_interval}
    np.savez_compressed(path_npz_speed, **save_object_array)
    
    # saving a csv file. The csv has two columns; `variable_id` and `edge_id`.
    # Open the CSV file in write mode
    with open(path_var_id_and_edge_id, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(['variable_id', 'edge_id'])

        # Write the list elements with row numbers
        __edge_id_list = vector_edge_id.tolist()
        for row_number, element in enumerate(__edge_id_list):
            writer.writerow([row_number, element])
        # end for
    # end with
# end def
 
    
def __main(path_toml: Path):
    assert path_toml.exists(), f'{path_toml}'
    
    import tempfile
    import shutil
    import toml
    
    config_obj = toml.load(path_toml)
    assert 'resources' in config_obj, f'{config_obj}'
    assert 'path_root' in config_obj['resources'], f'{config_obj}'
    assert 'dir_name_sumo_config' in config_obj['resources'], f'{config_obj}'
    assert 'dir_name_postprocess' in config_obj['resources'], f'{config_obj}'
    
    path_project_root = Path(config_obj['resources']['path_root'])
    path_sumo_config_dir = path_project_root / config_obj['resources']['dir_name_sumo_config']
    path_postprocess_dir = path_project_root / config_obj['resources']['dir_name_postprocess']
    
    assert path_sumo_config_dir.exists(), f'{path_sumo_config_dir}'
    assert path_postprocess_dir.exists(), f'{path_postprocess_dir}'
    
    seq_sub_dir_sumo_seed_name = [x for x in path_sumo_config_dir.iterdir() if x.is_dir()]
    logger.info(f'Deleted {seq_sub_dir_sumo_seed_name} seed subdirectories.')
    
    for __path_dir_sumo_out_seed in seq_sub_dir_sumo_seed_name:
        __seed_id = __path_dir_sumo_out_seed.name
        assert (__path_dir_sumo_out_seed / 'x').exists(), f'{__path_dir_sumo_out_seed}'
        assert (__path_dir_sumo_out_seed / 'y').exists(), f'{__path_dir_sumo_out_seed}'
        # ----------------------------------------------------
        # extraction from `x`.    
        _path_sumo_net_xml = __path_dir_sumo_out_seed / 'x' / 'in/most.net.xml'
        _path_edge_observation = __path_dir_sumo_out_seed / 'x' / 'out/most.edge-observation.xml'
        _path_out_dir = path_postprocess_dir / __seed_id / 'x'
        
        assert _path_sumo_net_xml.exists(), f'{_path_sumo_net_xml}'
        assert _path_edge_observation.exists(), f'{_path_edge_observation}'
        
        _path_out_dir.mkdir(parents=True, exist_ok=True)

        parse_edge_observation_and_writing_array(
            path_sumo_net_xml=_path_sumo_net_xml,
            path_edge_observation=_path_edge_observation,
            path_work_dir=_path_out_dir)
        logger.info(f'Extracted files are at: {_path_out_dir}')
    
        # ----------------------------------------------------
        # extraction from `y`.
        _path_sumo_net_xml = __path_dir_sumo_out_seed / 'y' / 'in/most.net.xml'
        _path_edge_observation = __path_dir_sumo_out_seed / 'y' / 'out/most.edge-observation.xml'
        _path_out_dir = path_postprocess_dir / __seed_id / 'y'

        assert _path_sumo_net_xml.exists(), f'{_path_sumo_net_xml}'
        assert _path_edge_observation.exists(), f'{_path_edge_observation}'        
        
        _path_out_dir.mkdir(parents=True, exist_ok=True)

        parse_edge_observation_and_writing_array(
            path_sumo_net_xml=_path_sumo_net_xml,
            path_edge_observation=_path_edge_observation,
            path_work_dir=_path_out_dir)
        logger.info(f'Extracted files are at: {_path_out_dir}')
    # end for


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    __args = ArgumentParser()
    __args.add_argument('--path_toml', required=True)
    __opt = __args.parse_args()
    
    __main(Path(__opt.path_toml))
