from pathlib import Path
import typing as ty
import json
import geojson
import sumolib
import simplekml
import dataclasses

from sumolib.net import Net
from sumolib.net.edge import Edge
from sumolib.net.node import Node
from sumolib.net.lane import Lane
import sumolib

import logzero
logger = logzero.logger


"""A CLI script writing out the detected weights to a GeoJSON file."""


def get_geo_information(path_sumo_net_xml: Path) -> ty.Dict[str, ty.Any]:
    # function to extract geo-coding system from SUMO net file.
    # Must be in the library
    path_sumo_net_xml = Path(path_sumo_net_xml)

    import xml.etree.ElementTree as ET

    geolocation_obj: ty.Dict = {}

    iter_elem = ET.iterparse(path_sumo_net_xml.as_posix())
    for _event, _elem in iter_elem:
        if _elem.tag == 'location':
            geolocation_obj = _elem.attrib
            geolocation_obj['netOffset'] = [float(v) for v in _elem.attrib['netOffset'].split(',')]
            geolocation_obj['convBoundary'] = [float(v) for v in _elem.attrib['convBoundary'].split(',')]
            geolocation_obj['origBoundary'] = [float(v) for v in _elem.attrib['origBoundary'].split(',')]
            break
    # end for
    # Note: the geo-coding is already epsg:4326. Hence no need to convert.
    return geolocation_obj


# an exmaple to get the edges and lanes

def get_lane_id_dict(net: sumolib.net.Net) -> ty.Dict[str, Lane]:
    """Get the lane id to lane object dictionary from the net object.
    """
    seq_edge_obj = net.getEdges(withInternal=True)
    seq_lane_id2lane = {}
    for _edge_obj in seq_edge_obj:
        _seq_lane_obj = _edge_obj.getLanes()
        for _lane_obj in _seq_lane_obj:
            seq_lane_id2lane[_lane_obj.getID()] = _lane_obj
        # end for
    # end for
    logger.debug(f'Number of lanes: {len(seq_lane_id2lane)}')
    return seq_lane_id2lane


def get_edge_polygon_shape_wgs84(dict_lane_id2lane: ty.Dict[str, Lane], 
                                 lane_id: str,
                                 net: sumolib.net.Net
                                 ) -> ty.Optional[ty.List[ty.Tuple[float, float]]]:
    """Get the polygon shape of the edge in WGS84 coordinate system.
    
    Returns
    -------
    ty.Optional[ty.List[ty.Tuple[float, float]]]
        The polygon shape of the edge in WGS84 coordinate system, (latitute, longitude)
    """
    if lane_id not in dict_lane_id2lane:
        logger.error(f'lane_id {lane_id} not found in the net.')
        return None
    # end if
    lane_obj = dict_lane_id2lane[lane_id]
    __polygon_shapes = lane_obj.getShape()
    if len(__polygon_shapes) == 0:
        logger.error(f'lane_id {lane_id} has no shape.')
        return None
    # end if
    __polygon_wsg84 = [net.convertXY2LonLat(__t_xy[0], __t_xy[1]) for __t_xy  in __polygon_shapes]
    logger.debug(f'Number of polygon shapes: {len(__polygon_wsg84)}')
    __seq_lat_lon = [(v[1], v[0]) for v in __polygon_wsg84]
    return __seq_lat_lon


def get_junction_object(net: sumolib.net.Net) -> ty.Dict[str, Node]:
    """Get the junction object from the net object.
    """
    seq_node_obj = net.getNodes()
    dict_nodeID2node= {__node_obj.getID(): __node_obj for __node_obj in seq_node_obj}
    return dict_nodeID2node


def get_junction_position_wgs84(dict_nodeID2node: ty.Dict[str, Node], 
                                junction_id: str,
                                net: sumolib.net.Net
                                ) -> ty.Optional[ty.Tuple[float, float]]:
    """Get the position of the junction in WGS84 coordinate system.
    
    Returns
    -------
    ty.Optional[ty.Tuple[float, float]]
        The position of the junction in WGS84 coordinate system.
    """
    if junction_id not in dict_nodeID2node:
        logger.error(f'junction_id {junction_id} not found in the net.')
        return None
    # end if
    __x, __y = dict_nodeID2node[junction_id].getCoord()
    lon, lat = net.convertXY2LonLat(__x, __y)
    logger.debug(f'{lat},{lon}')
    return lat, lon


def __process_one_time_bucket_weight(path_weight_json: Path,
                                     path_out_kml: Path,
                                     dict_nodeID2node: ty.Dict[str, Node],
                                     dict_laneID2lane: ty.Dict[str, Lane],
                                     net: sumolib.net.Net):
    logger.debug(f'loading weight file: {path_weight_json}')
    weight_file_content = json.load(path_weight_json.open())
    logger.info(f'N(weights) = {len(weight_file_content)}')
    
    seq_ids_node = [t_score for t_score in weight_file_content if ':' in t_score[0]]
    seq_ids_lanes = [t_score for t_score in weight_file_content if ':' not in t_score[0]]
    
    kml = simplekml.Kml()
    
    seq_positions = []
    # -------------------------------------
    # adding junction pins to Google Maps
    for __t_node_id in seq_ids_node:
        __junction_id = __t_node_id[0].split('_')[0].strip(':')
        _t_position = get_junction_position_wgs84(dict_nodeID2node, __junction_id, net=net)
        if _t_position is None:
            continue
        # end if
        
        # kml input must be (longitude, latitude)
        kml.newpoint(name=__t_node_id[0], 
                     coords=[(_t_position[1], _t_position[0])],
                     description=f'weight={__junction_id[1]}')
    # end for

    # -------------------------------------
    # Adding lane polygons to Google Maps
    for __t_lane_id in seq_ids_lanes:
        __lane_id = __t_lane_id[0]
        _t_position = get_edge_polygon_shape_wgs84(dict_laneID2lane, __lane_id, net=net)
        if _t_position is None:
            continue
        # end if
        
        # changing order of latitude and longitude
        _t_position_la_lon = [(v[1], v[0]) for v in _t_position]
        
        kml.newpolygon(
            name=__lane_id, 
            description=f'weight={__t_lane_id[1]}',
            outerboundaryis=_t_position_la_lon)
    # end for
    
    logger.debug(f'GeoJSON file written to {path_out_kml}')
    kml.save(path_out_kml.as_posix())



def main(path_sumo_net_xml: Path, 
         path_dir_load_weight: Path, 
         path_dir_kml: Path):
    assert path_dir_load_weight.exists(), f"Path {path_dir_load_weight} does not exist."
    path_dir_kml.mkdir(parents=True, exist_ok=True)

    seq_list_road_list_json = list(sorted(path_dir_load_weight.rglob('*.json')))
    assert len(seq_list_road_list_json) > 0, f'No json files found in {path_dir_load_weight}'
    
    logger.debug(f'loading sumo net file: {path_sumo_net_xml}')
    net = sumolib.net.readNet(path_sumo_net_xml.as_posix())
    dict_geo_coding_info = get_geo_information(path_sumo_net_xml)
    logger.debug(f'geo-coding info: {dict_geo_coding_info}')

    dict_nodeID2node = get_junction_object(net)
    dict_laneID2lane = get_lane_id_dict(net)
    
    for __path_weight_json in seq_list_road_list_json:
        logger.debug(f'loading weight file: {__path_weight_json}')
        _weight_file_content = json.load(__path_weight_json.open())
        logger.info(f'N(weights) = {len(_weight_file_content)}')
        
        _path_geo_csv_out = path_dir_kml / f'{__path_weight_json.stem}.kml'
        __process_one_time_bucket_weight(
            path_out_kml=_path_geo_csv_out,
            path_weight_json=__path_weight_json,
            dict_nodeID2node=dict_nodeID2node,
            dict_laneID2lane=dict_laneID2lane,
            net=net)


if __name__ == "__main__":
    from argparse import ArgumentParser
    opt = ArgumentParser()
    opt.add_argument("--path_sumo_net_xml", type=Path, required=True)
    opt.add_argument("--path_dir_load_weight", type=Path, required=True)
    opt.add_argument("--path_dir_kml", type=Path, required=True)
    __args = opt.parse_args()
    
    main(
        path_sumo_net_xml=__args.path_sumo_net_xml,
        path_dir_kml=__args.path_dir_kml,
        path_dir_load_weight=__args.path_dir_load_weight)