import typing as ty
from dataclasses import dataclass

from pathlib import Path

import xml.etree.ElementTree as ET

from tqdm import tqdm


"""A module to extract road geo-coordinate.
"""


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
    
    # logger.info(f"Number of lanes: {len(seq_road_id_obj)}")
    
    return seq_road_id_obj


def main(path_sumo_net_xml: Path,
         prefix_auto_route: str = 'highway.motorway') -> ty.List[RoadLaneObject]:
    """
    Args:
        path_sumo_net_xml: a SUMO net xml file path.
        prefix_auto_route: str, default is 'highway.motorway'. Filtering condition of extracting roads.
    """
    return __parse_road_network(
        path_sumo_net_xml=path_sumo_net_xml, prefix_auto_route=prefix_auto_route)
