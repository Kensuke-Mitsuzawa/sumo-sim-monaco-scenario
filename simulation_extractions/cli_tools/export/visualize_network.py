import typing as ty
import xml.etree.ElementTree as ET
import logzero
from tqdm import tqdm
import jsonlines

import numpy as np

from shapely.geometry import Polygon, LineString
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict

import toml
import dacite
import json

from pathlib import Path

logger = logzero.logger


"""
Input: 
- `most.net.xml`
- json file of the detected network and weights.

Output:
Png. file of the network.
"""

@dataclass
class RootConfig:
    path_sumo_net_xml: str
    # path_output_png: str
    # path_weights_json: ty.Optional[str] = None
    
    size_fig_width: int = 30
    size_fig_height: int = 30


# ------------------------------------------------

@dataclass
class RoadLaneObject:
    __slots__ = ['lane_id', 'polygon_coords', 'lane_index', 'speed_limit', 'allow', 'lane_type', 'is_autoroute']
    
    lane_id: str
    polygon_coords: ty.List[ty.Tuple[float, float]]
    lane_index: int
    speed_limit: float
    allow: ty.List[str]
    lane_type: ty.Optional[str]
    is_autoroute: bool


@dataclass
class RoadWeights:
    __slots__ = ['lane_id', 'weight']
    lane_id: str
    weight: float
    
    
def __load_road_weights(path_weights_json: Path) -> ty.List[RoadWeights]:
    """The json file should be a list of list (tuple).
    `[lane-id, weight]`
    The lane-id must be str, weights must be float.
    """
    assert path_weights_json.exists(), f"Path not found: {path_weights_json}"
    
    with jsonlines.open(path_weights_json, mode='r') as f:
        seq_weights = [__r for __r in f]
    # end with
    
    seq_road_weights = []
    for __t_lane_id_weight in seq_weights:
        assert isinstance(__t_lane_id_weight, dict), f"Invalid format: {__t_lane_id_weight}"
        assert 'lane_id' in __t_lane_id_weight and isinstance(__t_lane_id_weight['lane_id'], str), f"Invalid format: {__t_lane_id_weight}"
        assert 'weight' in __t_lane_id_weight and isinstance(__t_lane_id_weight['weight'], float), f"Invalid format: {__t_lane_id_weight}"
        seq_road_weights.append(RoadWeights(
            lane_id=__t_lane_id_weight['lane_id'], 
            weight=__t_lane_id_weight['weight']))
    # end for 
    
    return seq_road_weights
     

def __parse_polygon_coords(shape_xml_attribute: str) -> ty.List[ty.Tuple[float, float]]:
    """Extracting only x, y coords"""
    __seq_coords = shape_xml_attribute.split()
    seq_coords = []
    
    for __char_coords in __seq_coords:
        __seq_x_y = __char_coords.split(',')
        __x = float(__seq_x_y[0])
        __y = float(__seq_x_y[1])
        seq_coords.append((__x, __y))
    # end for
    return seq_coords
    

def __parse_road_network(path_sumo_net_xml: Path, 
                         prefix_auto_route: str = 'highway.motorway') -> ty.List[RoadLaneObject]:
    """Extracting the lane information from the given SUMO net xml file.
    """
    seq_road_id_obj = []
    for event, elem in tqdm(ET.iterparse(path_sumo_net_xml)):
        elem_tag = elem.tag
        if elem_tag == 'edge':
            __edge_attributes = elem.attrib
            __edge_type = __edge_attributes['type'] if 'type' in __edge_attributes else None
            if __edge_type is not None:
                __is_auto_route = True if prefix_auto_route in __edge_type else False 
            else:
                __is_auto_route = False
            
            __seq_lane_elem = elem.findall('lane')
            for __lane_elem_obj in __seq_lane_elem:
                __lane_attrib_obj = __lane_elem_obj.attrib
                # getting attributes
                __lane_id = __lane_attrib_obj['id']
                __lane_index = int(__lane_attrib_obj['index'])
                __shape = __lane_attrib_obj['shape']
                __speed_limit = float(__lane_attrib_obj['speed'])
                if 'allow' in __lane_attrib_obj:
                    __agent_allow = __lane_attrib_obj['allow'].split()
                else:
                    __agent_allow = []
                # end if
                
                __seq_shape = __parse_polygon_coords(__shape)
                
                __road_lane_obj = RoadLaneObject(
                    lane_id=__lane_id,
                    polygon_coords=__seq_shape,
                    lane_index=__lane_index,
                    speed_limit=__speed_limit,
                    allow=__agent_allow,
                    lane_type=__edge_type,
                    is_autoroute=__is_auto_route
                )
                seq_road_id_obj.append(__road_lane_obj)
            # end for
        # end if
    # end for
    
    logger.info(f"Number of lanes: {len(seq_road_id_obj)}")
    
    return seq_road_id_obj



def __plot_netowrk(seq_road_lane_obj: ty.List[RoadLaneObject],
                   path_output_png: Path, 
                   config_obj: RootConfig,
                   road_weights: ty.Optional[ty.List[RoadWeights]] = None,
                   path_out_json: ty.Optional[Path] = None):
    """Plotting the network.
    
    Parameters
    ----------
    seq_road_lane_obj: ty.List[RoadLaneObject]
        The list of lane objects.
    path_output_png: Path
        The output path of the png file.
    config_obj: RootConfig
        The configuration object.
    road_weights: ty.Optional[ty.List[RoadWeights]]
        The list of weights for the lanes.
    path_out_json: ty.Optional[Path]
        The output path of the json file.
        
    Returns
    -------
    None
    """
    f, ax = plt.subplots(figsize=(config_obj.size_fig_width, config_obj.size_fig_height))
    
    if road_weights is not None:
        lane_id2weights = {__obj.lane_id: __obj.weight for __obj in road_weights}
    else:
        lane_id2weights = {}
    # end if
     
    seq_lane_ids_with_coordinate = []
    for __lane_obj in tqdm(seq_road_lane_obj):
        # Create a Shapely polygon object
        if len(__lane_obj.polygon_coords) == 2:
            x = [__lane_obj.polygon_coords[0][0], __lane_obj.polygon_coords[1][0]]
            y = [__lane_obj.polygon_coords[0][1], __lane_obj.polygon_coords[1][1]]
        else:
            polygon = Polygon(__lane_obj.polygon_coords)
            # Extract the x and y coordinates of the polygon
            x, y = polygon.exterior.xy
        # end if
                
        if __lane_obj.lane_id in lane_id2weights:
            __color = 'red'
            __linewidth = lane_id2weights[__lane_obj.lane_id] + 5.0
            logger.info(f"Use weights value -> lane_id: {__lane_obj.lane_id}, weight: {lane_id2weights[__lane_obj.lane_id]}")
            # putting lane-id, weights, and coordinates of the road.
            seq_lane_ids_with_coordinate.append(dict(
                lane_id=__lane_obj.lane_id,
                weights=lane_id2weights.get(__lane_obj.lane_id, 0.0),
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
            ax.plot(_x_sorted, _y_sorted, color=__color, linestyle='solid', marker='*', linewidth=__linewidth)
        if set(__lane_obj.allow) == set(['pedestrian', 'bicycle']) or set(__lane_obj.allow) == set(['pedestrian']):
            # pietons et velos -> dotted line style
            ax.plot(_x_sorted, _y_sorted, color=__color, linestyle=':', linewidth=__linewidth)
        elif len(__lane_obj.allow) == 0:
            # les roues generiques -> solid line style
            ax.plot(_x_sorted, _y_sorted, color=__color, linestyle='solid', linewidth=__linewidth)
        elif 'rail' in __lane_obj.allow:
            # la rue de rail -> dash line style
            ax.plot(_x_sorted, _y_sorted, color=__color, linestyle='dashdot', marker='>', linewidth=__linewidth)
        else:
            print(__lane_obj.allow)
    # end for
    
    if path_out_json is not None:
        f_json_weights = open(path_out_json, 'w')
        f_json_weights.write(json.dumps(seq_lane_ids_with_coordinate, indent=4))
    # end if
    
    f.savefig(path_output_png.as_posix())


def main(path_config: Path,
         path_output_png: Path,
         path_weights_json: ty.Optional[Path] = None):
    assert path_config.exists(), f"Path not found: {path_config}"
    
    __config_obj = toml.load(path_config)
    config_obj = dacite.from_dict(data_class=RootConfig, data=__config_obj)
    
    path_output_png.parent.mkdir(parents=True, exist_ok=True)
        
    path_sumo_net_xml = Path(config_obj.path_sumo_net_xml)
    assert path_sumo_net_xml.exists(), f"Path not found: {path_sumo_net_xml}"
    
    if path_weights_json is not None:
        logger.info(f"Loading weights from {path_weights_json}")
        seq_road_weights = __load_road_weights(Path(path_weights_json))
    else:
        seq_road_weights = None
    # end if
    
    seq_road_id_obj = __parse_road_network(path_sumo_net_xml)
    __plot_netowrk(
        seq_road_lane_obj=seq_road_id_obj,
        path_output_png=path_output_png,
        config_obj=config_obj, 
        road_weights=seq_road_weights)



if __name__ == "__main__":
    from argparse import ArgumentParser
    
    opt = ArgumentParser(description='Visualizing the study map.')
    opt.add_argument('--path_config', 
                     type=str, 
                     required=True, 
                     help='A toml configuration file.')
    opt.add_argument('--path_out_png', 
                     type=str, 
                     required=True,
                     help='The output path to save the png file.')
    opt.add_argument('--path_weight_jsonl', 
                     type=str, 
                     required=False, 
                     default=None,
                     help='Optional. The path to the jsonl file of the weights.')
    
    __args = opt.parse_args()
    
    main(
        path_config=Path(__args.path_config),
        path_output_png=Path(__args.path_out_png),
        path_weights_json=__args.path_weight_jsonl)
