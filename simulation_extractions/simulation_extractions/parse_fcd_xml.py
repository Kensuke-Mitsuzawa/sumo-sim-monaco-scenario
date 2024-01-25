import typing as ty
import xml.etree.ElementTree as ET
import logzero
import toml
import json
import numpy as np
from dataclasses import dataclass

from dacite import from_dict

from pathlib import Path


from tqdm import tqdm

logger = logzero.logger


class AgentPosition(ty.NamedTuple):
    agent_id: str
    lane_id: str
    timestamp: float
    position_x: float
    position_y: float
    position_z: float
# end class


@dataclass
class RootConfig:
    path_sumo_cfg: str
    path_dir_save: str
    path_fcd_output: str
    
    file_name_intermediate_jsonl: str    
    file_name_array_lane_observation: str
    file_name_array_agent_position: str
# end if



def parse_fcd_xml(path_fcd_output: Path,
                  path_work_dir: Path,
                  file_name_intermediate_jsonl: str = 'extraction.jsonl',
                  interval_writing_out: int = 1000,
                  file_name_array_lane_observation: str = 'load_observation.npz',
                  file_name_array_agent_position: str = 'agent_position.npz'):
    """
    A parser script of SUMO fcd file output.
    Parsing FCD xml, it is much faster than executing vis traci. 
    """


    # PATH_TOML_CONFIG = '/dev-home/simulation_extractions/configurations/test_run.toml'
    # INTERVAL_WRITING_OUT = 1000  # per 1000 attributes, writing out.

    # __config_obj = toml.load(PATH_TOML_CONFIG)
    # config_obj = from_dict(RootConfig, __config_obj)


    assert Path(path_fcd_output).exists()

    set_lane_id = set()
    set_agent_id = set()
    seq_timestamps = list()

    seq_vehicle_container: list[AgentPosition] = []

    __timestamp: str = ''

    path_root_output = Path(path_work_dir)
    path_root_output.mkdir(exist_ok=True, parents=True)

    path_intermediate_jsonl = path_root_output / file_name_intermediate_jsonl 

    f_obj = path_intermediate_jsonl.open('w', buffering=1)

    for event, elem in ET.iterparse(path_fcd_output):
        # do something with each element here
        element_name: str = elem.tag
        if element_name == 'vehicle':
            __vehicle_attrib = elem.attrib
            
            __id_vehicle = __vehicle_attrib['id']
            __id_lane = __vehicle_attrib['lane']
            
            __x_vehicle = __vehicle_attrib['x']
            __y_vehicle = __vehicle_attrib['y']
            __z_vehicle  = __vehicle_attrib['z']
            
            set_lane_id.add(__id_lane)
            set_agent_id.add(__id_vehicle)
            
            seq_vehicle_container.append(AgentPosition(
                agent_id=__id_vehicle,
                lane_id=__id_lane,
                timestamp=__timestamp,
                position_x=__x_vehicle,
                position_y=__y_vehicle,
                position_z=__z_vehicle))
        elif element_name == 'timestep':
            __timestamp = elem.attrib['time']
            seq_timestamps.append(__timestamp)
            logger.debug(f'Timestep -> {__timestamp}')
        
        elem.clear()  # discard the element and free up memory
        
        if len(seq_vehicle_container) == interval_writing_out:
            for __record in seq_vehicle_container:
                __line = json.dumps(__record)
                f_obj.write(__line + '\n')
            # end for
            seq_vehicle_container = []
        # end if
        # TODO: probably I need to partially save.
    # end if

    f_obj.close()

    # sort lane-id, agent-id
    array_lane_id = np.array(sorted(list(set_lane_id)))
    array_agent_id = np.array(sorted(list(set_agent_id)))
    array_timestamp = np.array(seq_timestamps)

    # set the array.
    array_lane_observation = np.zeros([len(set_lane_id), len(seq_timestamps)])
    array_trajectory = np.zeros(tuple([len(set_agent_id), len(seq_timestamps), 3]))

    # re-opening jsonl
    # constructing lane-id count array / trajectory array
    f_obj_read = path_intermediate_jsonl.open('r', buffering=1)
    for __line in tqdm(f_obj_read):
        __attr_info = json.loads(__line)
        __agent_info = AgentPosition(*__attr_info)
        
        __lane_id_index = np.where(array_lane_id == __agent_info.lane_id)[0]
        __timestamp_index = np.where(array_timestamp == __agent_info.timestamp)[0]
        __agent_id_index = np.where(array_agent_id == __agent_info.agent_id)[0]
        
        # array value updating
        array_lane_observation[__lane_id_index, __timestamp_index] += 1
        # trajectory array
        array_trajectory[__agent_id_index, __timestamp_index, :] = [__agent_info.position_x, 
                                                                    __agent_info.position_y, 
                                                                    __agent_info.position_z]
        
    # end for

    # saving lane observation
    path_npz_lane_observation = path_root_output / file_name_array_lane_observation 
    save_object_lane_obs = {
        'array': array_lane_observation,
        'lane_id': array_lane_id,
        'timestamps': array_timestamp
    }
    np.savez_compressed(path_npz_lane_observation, **save_object_lane_obs)

    # saving trajectory array
    path_npz_agent_positions = path_root_output / file_name_array_agent_position 
    save_object_agent_positions = {
        'array': array_trajectory,
        'agent_id': array_agent_id,
        'timestamps': array_timestamp
    }
    np.savez_compressed(path_npz_agent_positions, **save_object_agent_positions)


    path_intermediate_jsonl.unlink()