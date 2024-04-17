import typing as ty
import xml.etree.ElementTree as ET
import logzero
import toml
import json
import numpy as np
from dataclasses import dataclass

import sumolib

from pathlib import Path

logger = logzero.logger


class EdgeInformation(ty.NamedTuple):
    edge_id: str
    time_begin: float
    time_end: float
    sample_count: int
    travel_time: float
    density: float
    waiting_time: float
    time_loss: float
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


def parse_edge_observation(path_sumo_net_xml: Path,
                           path_edge_observation: Path,
                           path_work_dir: Path,
                           file_name_travel_time: str = 'edge_travel_time.npz',
                           file_name_density: str = 'edge_density.npz',
                           file_name_waiting_time: str = 'edge_waiting_time.npz',
                           file_name_time_loss: str = 'edge_time_loss.npz',
                           file_name_count: str = 'edge_count.npz'):
    """
    A parser script of SUMO edge file output.
    See: https://sumo.dlr.de/docs/Simulation/Output/Lane-_or_Edge-based_Traffic_Measures.html#generated_output
    """
    assert path_sumo_net_xml.exists()
    assert Path(path_edge_observation).exists()
    
    net = sumolib.net.readNet(path_sumo_net_xml.as_posix())
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

    # putting values to arrays
    _vector_time_interval = np.array(['-'.join(_t) for _t in stack_time_interval])
    for _e_info in stack_edge_info:
        _j = np.where(vector_edge_id == _e_info.edge_id)[0].item()
        _i = np.where(_vector_time_interval == f'{_e_info.time_begin}-{_e_info.time_end}')[0].item()
        array_travel_time[_j, _i] = _e_info.travel_time
        array_density[_j, _i] = _e_info.density
        array_waiting_time[_j, _i] = _e_info.waiting_time
        array_time_loss[_j, _i] = _e_info.time_loss
        array_sample_count[_j, _i] = _e_info.sample_count
    # end for

    # saving files
    path_npz_travel_time = path_work_dir / file_name_travel_time
    path_npz_density = path_work_dir / file_name_density
    path_npz_waiting_time = path_work_dir / file_name_waiting_time
    path_npz_time_loss = path_work_dir / file_name_time_loss
    path_npz_sample_count = path_work_dir / file_name_count
    
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
# end def    
    
    
def __test():
    import tempfile
    import shutil
    
    path_sumo_net_xml = Path('/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/original_config/in/most.net.xml')
    path_edge_observation = Path('/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/original_config/out/most.edge-observation.xml')
    path_work_dir = Path(tempfile.mkdtemp())

    parse_edge_observation(
        path_sumo_net_xml=path_sumo_net_xml,
        path_edge_observation=path_edge_observation,
        path_work_dir=path_work_dir,
    )
    
    shutil.rmtree(path_work_dir)


if __name__ == '__main__':
    __test()