from pathlib import Path
from tqdm import tqdm
import typing as ty

import toml

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from shapely.geometry import Polygon, LineString

import imageio.v3 as iio

from simulation_extractions import get_road_coordinate_simulation

from distributed import Client, LocalCluster

import logzero
logger = logzero.logger


"""This script is for generating a set of snapshots from a set of data and save it into a GIF video.
"""


class ArraySet(ty.NamedTuple):
    observation_array: xr.DataArray
    label_road: np.ndarray
    label_time: np.ndarray


def __plot_netowrk(seq_road_lane_obj: ty.List[get_road_coordinate_simulation.RoadLaneObject],
                   edge_id2obs: ty.Dict[str, float],
                   path_output_png: Path):
    """Plotting the network.
    
    Parameters
    ----------
    seq_road_lane_obj: ty.List[RoadLaneObject]
        The list of lane objects.
    path_output_png: Path
        The output path of the png file.
                
    Returns
    -------
    None
    """
    f, ax = plt.subplots(figsize=(10, 10))
     
    seq_lane_ids_with_coordinate = []
    
    processed_edge_ids = []
    
    # making a color code normalizer
    if len(edge_id2obs) == 0:
        mappable = None
    else:        
        seq_values = list(edge_id2obs.values())
        norm = mcolors.Normalize(vmin=np.min(seq_values), vmax=np.max(seq_values))
        cmap = plt.get_cmap('viridis')
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    # end if
    
    for __lane_obj in seq_road_lane_obj:
        # Create a Shapely polygon object
        if len(__lane_obj.polygon_coords) == 2:
            x = [__lane_obj.polygon_coords[0][0], __lane_obj.polygon_coords[1][0]]
            y = [__lane_obj.polygon_coords[0][1], __lane_obj.polygon_coords[1][1]]
        else:
            polygon = Polygon(__lane_obj.polygon_coords)
            # Extract the x and y coordinates of the polygon
            x, y = polygon.exterior.xy
        # end if
                
        # lane-id -> edge-id
        _edge_id = __lane_obj.lane_id.split('_')[0]
        if ":" in _edge_id:
            # if the intersection lane obj, skip.
            continue
        elif _edge_id in processed_edge_ids:
            continue
        else:
            processed_edge_ids.append(_edge_id)
        # end if
    
        # process for the selected variables.
        # TODO: changing color by the weight.
        
        if _edge_id in edge_id2obs:
            assert mappable is not None, f"mappable is None, but edge_id2obs is not empty."
            __color = mappable.to_rgba(edge_id2obs.get(_edge_id, 0.0))
            # __color = 'red'
            __linewidth = 0.5
            # putting lane-id, weights, and coordinates of the road.
            seq_lane_ids_with_coordinate.append(dict(
                lane_id=_edge_id,
                weights=edge_id2obs.get(_edge_id, 0.0),
                coordinate_tuples=list(zip(x, y))
            ))
        else:
            __color = 'black'
            __linewidth = 0.5
        # end if
        
        __index_sort = np.argsort(x)
        # __index_sort = np.lexsort((x, y))
        _x_sorted = np.array(x)[__index_sort]
        _y_sorted = np.array(y)[__index_sort]
        
        # Plot the polygon
        if __lane_obj.is_autoroute:
            # l'autoroute A8 avec
            ax.plot(_x_sorted, _y_sorted, color=__color, linestyle='solid', linewidth=__linewidth)
        if set(__lane_obj.allow) == set(['pedestrian', 'bicycle']) or set(__lane_obj.allow) == set(['pedestrian']):
            # pietons et velos -> dotted line style
            ax.plot(_x_sorted, _y_sorted, color=__color, linestyle='solid', linewidth=__linewidth)
        elif len(__lane_obj.allow) == 0:
            # les roues generiques -> solid line style
            ax.plot(_x_sorted, _y_sorted, color=__color, linestyle='solid', linewidth=__linewidth)
        elif 'rail' in __lane_obj.allow:
            # la rue de rail -> dash line style
            # ax.plot(_x_sorted, _y_sorted, color=__color, linestyle='dashdot', marker='>', linewidth=__linewidth)
            logger.debug('I skip the train lane.')
            pass
        else:
            print(__lane_obj.allow)
    # end for
        
    f.savefig(path_output_png.as_posix())


def generate_snapshot(input_array_dataset: ArraySet, 
                      seq_road_geo_container: ty.List[get_road_coordinate_simulation.RoadLaneObject],
                      path_dir_snapshot: Path,
                      timestep_per_snapshot: int = 100):
    assert timestep_per_snapshot > 0, f"timestep_per_snapshot: {timestep_per_snapshot} must be greater than 0"
    
    n_timesteps = input_array_dataset.observation_array.sizes['time']
    for _t in tqdm(range(0, n_timesteps, timestep_per_snapshot)):
        # 1D array at the time=_t
        array_t = input_array_dataset.observation_array[:, _t]
        logger.debug(f"array_t: {_t} - {_t + timestep_per_snapshot}")
        
        # lane_id2observation_value
        edge_id2obs = {}
        for __edge_id, __obs in zip(input_array_dataset.label_road, array_t):
            if __obs > 0:
                edge_id2obs[__edge_id] = __obs.item()
            # end if
        # end for

        path_snapshot = path_dir_snapshot / f"snapshot_{_t:04d}.png"        
        __plot_netowrk(seq_road_geo_container, edge_id2obs, path_snapshot)
        logger.debug(f"Saved: {path_snapshot}")
        
    
def _load_simulation_output(path_npz: Path, 
                            key_name_array: str,
                            key_name_label: str) -> ArraySet:
    """loading the simulation output, packing all into a one"""
    npz_obj = np.load(path_npz)
    assert key_name_array in npz_obj, f"key_name_array: {key_name_array} is not found in the npz file: {path_npz}"
    array = npz_obj[key_name_array]
    # logger.debug(f"array: {array}")
    # logger.debug(f"array.shape: {array.shape}")
    
    vec_label = npz_obj[key_name_label]
    vec_time = npz_obj['timestamps']
    
    n_road, n_time = array.shape
    assert n_road == len(vec_label), f"n_road: {n_road} != len(vec_label): {len(vec_label)}"
    assert n_time == len(vec_time), f"n_time: {n_time} != len(vec_time): {len(vec_time)}"
    
    x_array = xr.DataArray(array, dims=('sensor', 'time'))
    dataset = ArraySet(
        observation_array=x_array,
        label_road=vec_label,
        label_time=vec_time)
    
    return dataset


# TODO, annotation from the variable selection.
def main(path_tomo_config: Path, 
         path_sumo_xml: Path,
         timestep_per_snapshot: int = 10):
    """
    Args:
        path_tomo_config: a path to the configuration file.
        path_sumo_xml: a path to the SUMO net xml file.
        timestep_per_snapshot: the number of timesteps per snapshot.
    """
    assert Path(path_tomo_config).exists(), f"path_tomo_config: {path_tomo_config} does not exist."
    assert Path(path_sumo_xml).exists(), f"path_sumo_xml: {path_sumo_xml} does not exist."
    
    config_obj = toml.load(path_tomo_config)
    logger.debug(f"config_obj: {config_obj}")
    
    path_dir_output = Path(config_obj['Resoruce']['output']['path_output_resource'])
    
    # Load simulation output
    # Note; there is a mis-spelling 'Resoruce', which is a correct "variable name".
    assert 'Resoruce' in config_obj.keys(), f"Resoruce key is not found in the configuration file: {path_tomo_config}"
    assert 'output' in config_obj['Resoruce'], f"output key is not found in the configuration file: {path_tomo_config}"
    assert 'input_x' in config_obj['Resoruce'], f"input_x key is not found in the configuration file: {path_tomo_config}"
    assert 'input_y' in config_obj['Resoruce'], f"input_y key is not found in the configuration file: {path_tomo_config}"
    
    path_input_x = Path(config_obj['Resoruce']['input_x']['path_simulation_output'])
    path_input_y = Path(config_obj['Resoruce']['input_y']['path_simulation_output'])
    
    key_name_array = config_obj['Resoruce']['input_x']['key_name_lane_or_edge_id_vector']
    
    # loading geo coordinate of the roads
    logger.debug(f"loading net info from path_sumo_xml: {path_sumo_xml}")
    seq_road_container = get_road_coordinate_simulation.main(path_sumo_xml)
    logger.debug(f"Done")
    
    array_x = _load_simulation_output(path_input_x, 'array', key_name_array)
    array_y = _load_simulation_output(path_input_y, 'array', key_name_array)
    
    # L1 difference, and generate snapshots
    diff_array = np.abs(array_x.observation_array - array_y.observation_array)
    diff_dataset = ArraySet(
        observation_array=xr.DataArray(diff_array, dims=('sensor', 'time')),
        label_road=array_x.label_road,
        label_time=array_x.label_time)
    
    path_dir_l1_snapshot = path_dir_output / 'l1_snapshot'
    path_dir_l1_snapshot.mkdir(parents=True, exist_ok=True)
    generate_snapshot(diff_dataset,
                      seq_road_container, 
                      path_dir_l1_snapshot,
                      timestep_per_snapshot=timestep_per_snapshot)
    
    # array x
    
    # array y
    

if __name__ == "__main__":
    __path_toml_config = Path('/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/cli_tools/simple_analysis/configurations/config_edge_observation.toml')
    __path_sumo_xml = Path('/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/original_config/in/most.net.xml')
    main(__path_toml_config, __path_sumo_xml, timestep_per_snapshot=100)

