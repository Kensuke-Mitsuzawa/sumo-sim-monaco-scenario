"""A CLI script writing out the detected weights to a GeoJSON file."""

from pathlib import Path
import typing as ty
import json
import geojson
import simplekml
import dataclasses
import jsonlines
import math

import xml.etree.ElementTree as ET

import sumolib
from sumolib.net import Net
from sumolib.net.edge import Edge
from sumolib.net.node import Node
from sumolib.net.lane import Lane

import datetime

import logzero
logger = logzero.logger



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


def __get_simulation_world_time(current_timestamp: int) -> str:
    """Get the simulation world time from the current timestamp.
    """
    hours = math.floor(current_timestamp / 60 / 60)
    minutes = math.floor((current_timestamp - (hours * 60 * 60)) / 60)

    return f'{hours:02d}:{minutes:02d}'

def __process_one_time_bucket_weight(path_weight_jsonl: Path,
                                     path_output_geo_file: Path,
                                     dict_nodeID2node: ty.Dict[str, Node],
                                     dict_laneID2lane: ty.Dict[str, Lane],
                                     net: sumolib.net.Net,
                                     export_geofile_format: str,
                                     simulation_start_time: int,
                                     n_time_bucket: int,
                                     date_timestamp: ty.Optional[datetime.date] = None):
    assert export_geofile_format in ['kepler.gl', 'google-earth'], f'Unknown export_geofile_format: {export_geofile_format}'
    
    logger.debug(f'loading weight file: {path_weight_jsonl}')
    with jsonlines.open(path_weight_jsonl.as_posix()) as reader:
        weight_file_content = [_r for _r in reader]
    # end with
    logger.info(f'N(weights) = {len(weight_file_content)}')
    
    seq_ids_node = [t_score for t_score in weight_file_content if ':' in t_score['lane_id']]
    seq_ids_lanes = [t_score for t_score in weight_file_content if ':' not in t_score['lane_id']]
    
    kml = simplekml.Kml()
    seq_geojson_obj = []
    
    if date_timestamp is None:
        date_ = datetime.date.today().isoformat()
    else:
        date_ = date_timestamp.isoformat()
    # end if
    
    _type_object: str
    for weight_obj in weight_file_content:
        _lane_id_orig = weight_obj['lane_id']
        if ':' in _lane_id_orig:
            _type_object = 'junction'
        else:
            _type_object = 'lane'
        # end if
        assert _t_position is not None, f'No position found for {_lane_id_orig}'
        
        # define a timestamp. Use time-bucket
        _i_time_bucket: int = weight_obj['time_bucket']
        _timestep_bucket_start = simulation_start_time + (_i_time_bucket * n_time_bucket)
        _current_hour_min = __get_simulation_world_time(_timestep_bucket_start)
        _date_timestamp_bucket_start = f'{date_} {_current_hour_min}'
        
        # get the description
        _description = weight_obj['label']
        
        if _type_object == 'junction':
            __junction_id = _lane_id_orig.split('_')[0].strip(':')
            _t_position = get_junction_position_wgs84(dict_nodeID2node, __junction_id, net=net)
            
            assert _t_position is not None and isinstance(_t_position, tuple), f'No position found for {_lane_id_orig}'
            _lat = _t_position[0]
            _lon = _t_position[1]
            # define a URL link to Google Map.
            _gmap_url = f'https://www.google.com/maps/place/{_lat},{_lon}'
            
            if export_geofile_format == 'kepler.gl':
                __point = geojson.Point((float(_lon), float(_lat)))
                __prop = dict(
                    google_map_link=_gmap_url,
                    description=_description,
                    time_bucket=_i_time_bucket,
                    timestamp=_date_timestamp_bucket_start)
                _point_info = geojson.Feature(geometry=__point, properties=__prop)
                seq_geojson_obj.append(_point_info)
            elif export_geofile_format == 'google-earth':
                raise NotImplementedError('Google Earth export is not implemented yet.')            
                # kml.newpoint(name=_lane_id_orig, 
                #             coords=[(_lon, _lat)],
                #             description=f'{_description}\n{_date_timestamp_bucket_start}\n{_gmap_url}')
            else:
                raise ValueError(f'Unknown export_geofile_format: {export_geofile_format}')
            # end if
        elif _type_object == 'lane':
            __lane_id = _lane_id_orig
            _t_position = get_edge_polygon_shape_wgs84(dict_laneID2lane, __lane_id, net=net)
            
            assert _t_position is not None and isinstance(_t_position, list), f'No position found for {_lane_id_orig}'
            _lat = _t_position[0][0]
            _lon = _t_position[0][1]
            # define a URL link to Google Map.
            _gmap_url = f'https://www.google.com/maps/place/{_lat},{_lon}'
            
            
            if export_geofile_format == 'kepler.gl':
                __shape = geojson.MultiLineString(_t_position)
                __prop = dict(
                    google_map_link=_gmap_url,
                    description=_description,
                    time_bucket=_i_time_bucket,
                    timestamp=_date_timestamp_bucket_start)
                __shape_info = geojson.Feature(geometry=__shape, properties=__prop)
                seq_geojson_obj.append(__shape_info)
            elif export_geofile_format == 'google-earth':
                raise NotImplementedError('Google Earth export is not implemented yet.')
                # kml.newpolygon(
                #     name=_lane_id_orig, 
                #     description=f'{_description}\n{_date_timestamp_bucket_start}\n{_gmap_url}',
                #     outerboundaryis=_t_position)        
            else:
                raise ValueError(f'Unknown export_geofile_format: {export_geofile_format}')
            # end if
        else:
            raise ValueError(f'Unknown type object: {_type_object}')
        # end if
    # end for
        
    if export_geofile_format == 'kepler.gl':
        with open(path_output_geo_file, 'w') as f:
            geojson.dump(geojson.FeatureCollection(seq_geojson_obj), f)
        logger.debug(f'GeoJSON file written to {path_output_geo_file}')
    elif export_geofile_format == 'google-earth':
        kml.save(path_output_geo_file.as_posix())
        logger.debug(f'KML file written to {path_output_geo_file}')    
    else:
        raise ValueError(f'Unknown export_geofile_format: {export_geofile_format}')


def extract_simulation_time(path_sumo_sim_xml) -> ty.Tuple[int, int, float]:
    """
    Extract the simulation time from the SUMO simulation config file (XML).
    
    Return
    ------
    ty.Tuple[int, int, float]
        The start time, end time, and the time step of the simulation.
    """
    assert path_sumo_sim_xml.exists(), f'File not found: {path_sumo_sim_xml}'
    
    _time_start = ''
    _time_end = ''
    _time_step = ''
    for event, elem in ET.iterparse(path_sumo_sim_xml):
        # do something with each element here
        element_name: str = elem.tag
        if element_name == 'time':
            _time_start = elem.find('begin').get('value')
            _time_end = elem.find('end').get('value')
            _time_step = elem.find('step').get('value')
        # end if
    # end for
    
    assert _time_start != '', f'No start time found in {path_sumo_sim_xml}'
    assert _time_end != '', f'No end time found in {path_sumo_sim_xml}'
    if _time_step == '':
        logger.warning(f'No time step found in {path_sumo_sim_xml}')
        time_step = 1.0
    else:
        time_step = float(_time_step)
    # end if
    
    return int(_time_start), int(_time_end), time_step
            
    



def main(path_sumo_sim_xml: Path,
         path_sumo_net_xml: Path, 
         path_weight_jsonl: Path, 
         path_output_geo_file: Path,
         export_geofile_format: str,
         n_time_bucket: int,):
    assert path_weight_jsonl.exists(), f"Path {path_weight_jsonl} does not exist."
    assert path_sumo_sim_xml.exists(), f"Path {path_sumo_sim_xml} does not exist."
    assert path_sumo_net_xml.exists(), f"Path {path_sumo_net_xml} does not exist."
    
    # get simulation time info    
    sim_time_start, __, time_time_step = extract_simulation_time(path_sumo_sim_xml=path_sumo_sim_xml)
    
    logger.debug(f'loading sumo net file: {path_sumo_net_xml}')
    net = sumolib.net.readNet(path_sumo_net_xml.as_posix())
    dict_geo_coding_info = get_geo_information(path_sumo_net_xml)
    logger.debug(f'geo-coding info: {dict_geo_coding_info}')

    dict_nodeID2node = get_junction_object(net)
    dict_laneID2lane = get_lane_id_dict(net)
        
    __process_one_time_bucket_weight(
        path_weight_jsonl=path_weight_jsonl,
        path_output_geo_file=path_output_geo_file,
        dict_nodeID2node=dict_nodeID2node,
        dict_laneID2lane=dict_laneID2lane,
        net=net,
        export_geofile_format=export_geofile_format,
        n_time_bucket=n_time_bucket,
        date_timestamp=datetime.date.fromtimestamp(sim_time_start)
    )


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    path_sumo_sim_xml = Path()
    path_sumo_net_xml = Path()
    path_weight_jsonl = Path() 
    path_output_geo_file = Path()
    export_geofile_format = 'kepler.gl'
    n_time_bucket = 600
    
    main(
        path_weight_jsonl=path_weight_jsonl,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_sumo_net_xml=path_sumo_net_xml,
        path_output_geo_file=path_output_geo_file,
        export_geofile_format=export_geofile_format,
        n_time_bucket=n_time_bucket)