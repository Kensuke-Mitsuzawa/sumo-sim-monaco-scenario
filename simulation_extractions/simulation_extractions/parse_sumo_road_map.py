import typing as ty
from pathlib import Path

import sumolib
from sumolib.net import Net
from sumolib.net.edge import Edge
from sumolib.net.node import Node
from sumolib.net.lane import Lane

import xml.etree.ElementTree as ET

import math

import numpy as np

import datetime
import geojson



"""
A class of generating output attributes for the kepler.gl.
"""

class VariableWeightObject(ty.NamedTuple):
    route_id: str
    weight: float
    time_bucket: int
    label: str
    

class GeoAttributeInformation(ty.NamedTuple):
    route_id: str   # lane-id or edge-id
    geo_json: str  # geojson object
    value: float  # value to be visualised
    google_map_link: str  # google map link to the corresponding route.
    description: str  # a short message of describing the information.
    timestamp: str  # timestamp    
    time_bucket: ty.Optional[int] = None  # time bucket
    


class KeplerAttributeGenerator(object):
    def __init__(self,
                 path_sumo_net_xml: Path,
                 path_sumo_sim_xml: Path) -> None:
        assert path_sumo_net_xml.exists(), f'path_sumo_net_xml does not exist: {path_sumo_net_xml}'
        self.path_sumo_net_xml = path_sumo_net_xml
        self.path_sumo_sim_xml = path_sumo_sim_xml
        
        self.net = sumolib.net.readNet(path_sumo_net_xml.as_posix())
    
        self.dict_nodeID2node = self._get_junction_object()
        self.dict_edgeID2edge = self._get_edge_id_dict()
        self.dict_laneID2lane = self._get_lane_id_dict()
        
    def get_geo_information(self) -> ty.Dict[str, ty.Any]:
        # function to extract geo-coding system from SUMO net file.
        # Must be in the library
        path_sumo_net_xml = Path(self.path_sumo_net_xml)

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

    def _get_lane_id_dict(self) -> ty.Dict[str, Lane]:
        """Get the lane id to lane object dictionary from the net object.
        """
        seq_edge_obj = self.net.getEdges(withInternal=True)
        seq_lane_id2lane = {}
        for _edge_obj in seq_edge_obj:
            _seq_lane_obj = _edge_obj.getLanes()
            for _lane_obj in _seq_lane_obj:
                seq_lane_id2lane[_lane_obj.getID()] = _lane_obj
            # end for
        # end for
        # logger.debug(f'Number of lanes: {len(seq_lane_id2lane)}')
        return seq_lane_id2lane

    def _get_edge_id_dict(self) -> ty.Dict[str, Edge]:
        """Get the egde id to lane object dictionary from the net object.
        """
        seq_edge_obj = self.net.getEdges(withInternal=True)
        seq_edge_id2lane = {}
        for _edge_obj in seq_edge_obj:
            _egde_id = _edge_obj.getID()
            seq_edge_id2lane[_egde_id] = _edge_obj
        # end for
        # logger.debug(f'Number of lanes: {len(seq_edge_id2lane)}')
        return seq_edge_id2lane

    def _get_edge_polygon_shape_wgs84(self,
                                      dict_road_id2lane: ty.Union[ty.Dict[str, Edge], ty.Dict[str, Lane], ty.Dict[str, Node]], 
                                      lane_id: str) -> ty.Optional[ty.List[ty.Tuple[float, float]]]:
        """Get the polygon shape of the edge in WGS84 coordinate system.
        
        Returns
        -------
        ty.Optional[ty.List[ty.Tuple[float, float]]]
            The polygon shape of the edge in WGS84 coordinate system, (latitute, longitude)
        """
        if lane_id not in dict_road_id2lane:
            # logger.error(f'lane_id {lane_id} not found in the net.')
            return None
        # end if
        lane_obj = dict_road_id2lane[lane_id]
        __polygon_shapes = lane_obj.getShape()
        if len(__polygon_shapes) == 0:
            # logger.error(f'lane_id {lane_id} has no shape.')
            return None
        # end if
        __polygon_wsg84 = [self.net.convertXY2LonLat(__t_xy[0], __t_xy[1]) for __t_xy  in __polygon_shapes]
        # logger.debug(f'Number of polygon shapes: {len(__polygon_wsg84)}')
        __seq_lat_lon = [(v[1], v[0]) for v in __polygon_wsg84]
        return __seq_lat_lon

    def _get_junction_object(self) -> ty.Dict[str, Node]:
        """Get the junction object from the net object.
        """
        seq_node_obj = self.net.getNodes()
        dict_nodeID2node= {__node_obj.getID(): __node_obj for __node_obj in seq_node_obj}
        return dict_nodeID2node

    def _get_junction_position_wgs84(self,
                                     dict_nodeID2node: ty.Dict[str, Node], 
                                     junction_id: str
                                     ) -> ty.Optional[ty.Tuple[float, float]]:
        """Get the position of the junction in WGS84 coordinate system.
        
        Returns
        -------
        ty.Optional[ty.Tuple[float, float]]
            The position of the junction in WGS84 coordinate system.
        """
        if junction_id not in dict_nodeID2node:
            # logger.error(f'junction_id {junction_id} not found in the net.')
            return None
        # end if
        __x, __y = dict_nodeID2node[junction_id].getCoord()
        lon, lat = self.net.convertXY2LonLat(__x, __y)
        # logger.debug(f'{lat},{lon}')
        return lat, lon

    def _get_simulation_world_time(self, current_timestamp: float) -> str:
        """Get the simulation world time from the current timestamp.
        """
        hours = math.floor(current_timestamp / 60 / 60)
        minutes = math.floor((current_timestamp - (hours * 60 * 60)) / 60)

        return f'{hours:02d}:{minutes:02d}'

    def extract_simulation_time(self) -> ty.Tuple[int, int, float]:
        """
        Extract the simulation time from the SUMO simulation config file (XML).
        
        Return
        ------
        ty.Tuple[int, int, float]
            The start time, end time, and the time step of the simulation.
        """
        assert self.path_sumo_sim_xml.exists(), f'File not found: {self.path_sumo_sim_xml}'
        
        _time_start = ''
        _time_end = ''
        _time_step = ''
        for event, elem in ET.iterparse(self.path_sumo_sim_xml):
            # do something with each element here
            element_name: str = elem.tag
            if element_name == 'time':
                _time_start = elem.find('begin').get('value')
                _time_end = elem.find('end').get('value')
                _time_step = elem.find('step-length').get('value')
            # end if
        # end for
        
        assert _time_start != '', f'No start time found in {self.path_sumo_sim_xml}'
        assert _time_end != '', f'No end time found in {self.path_sumo_sim_xml}'
        if _time_step == '':
            # logger.warning(f'No time step found in {path_sumo_sim_xml}')
            time_step = 1.0
        else:
            time_step = float(_time_step)
        # end if
        
        return int(_time_start), int(_time_end), time_step
    
    
    def _get_route_object_type(self, route_id: str) -> str:
        """Get the type of the route object."""
        _lane_id_orig = route_id
        if ':' in _lane_id_orig:
            _type_object = 'junction'
        else:
            _type_object = 'lane'
        # end if
        return _type_object        
        
    def _get_geo_json_object(self, 
                             _lane_id_orig: str,
                             lane_or_egde: str) -> str:
        """Private method. Get the geo-json object of the route object."""
        # comment: the key name is 'lane_id', yet possibly "edge-id".
        if ':' in _lane_id_orig:
            _type_object = 'junction'
        else:
            _type_object = 'lane'
        # end if
                
        if _type_object == 'junction':
            __junction_id = _lane_id_orig.split('_')[0].strip(':')
            _t_position = self._get_junction_position_wgs84(self.dict_nodeID2node, __junction_id)
            
            assert _t_position is not None and isinstance(_t_position, tuple), f'No position found for {_lane_id_orig}'
            _lat = _t_position[0]
            _lon = _t_position[1]
                        
            __shape = geojson.Point((float(_lon), float(_lat)))
        elif _type_object == 'lane':
            __lane_id = _lane_id_orig
            if lane_or_egde == 'edge':
                _t_position = self._get_edge_polygon_shape_wgs84(self.dict_edgeID2edge, __lane_id)
            elif lane_or_egde == 'lane':
                _t_position = self._get_edge_polygon_shape_wgs84(self.dict_laneID2lane, __lane_id)
            else:
                raise ValueError(f'Unknown lane_or_egde: {lane_or_egde}')
            # end if

            assert _t_position is not None and isinstance(_t_position, list), f'No position found for {_lane_id_orig}'
            _lat = _t_position[0][0]
            _lon = _t_position[0][1]
            
            _t_position_lon_lat = [[(v[1], v[0]) for v in _t_position]]
            __shape = geojson.MultiLineString(_t_position_lon_lat)
        else:
            raise ValueError(f'Unknown type object: {_type_object}')
        # end if
        
        return geojson.dumps(__shape)
    
    def _generate_google_map_link(self,
                                  route_id: str,                                  
                                  lane_or_egde: str) -> str:
        """Generating a link URL to the Google map"""
        _type_object = self._get_route_object_type(route_id)
        
        if _type_object == 'junction':
            __junction_id = route_id.split('_')[0].strip(':')
            _t_position = self._get_junction_position_wgs84(self.dict_nodeID2node, __junction_id)
        else:        
            if lane_or_egde == 'edge':
                _t_position = self._get_edge_polygon_shape_wgs84(self.dict_edgeID2edge, route_id)
            elif lane_or_egde == 'lane':
                _t_position = self._get_edge_polygon_shape_wgs84(self.dict_laneID2lane, route_id)
            else:
                raise ValueError(f'Unknown lane_or_egde: {lane_or_egde}')
            # end if
        # end if        
        
        assert _t_position is not None and isinstance(_t_position, tuple), f'No position found for {route_id}'
        _lat = _t_position[0]
        _lon = _t_position[1]
        # define a URL link to Google Map.
        _gmap_url = f'https://www.google.com/maps/place/{_lat},{_lon}'
        
        return _gmap_url

    # ------------------------------------------------------------------------------------------
    # APIs
    
    def generate_attributes_traffic_observation(self,
                                                array_traffic_observation: np.ndarray,
                                                seq_lane_id: ty.List[str],
                                                observation_every_step_per: int,
                                                size_time_bucket: int,
                                                lane_or_egde: str = 'lane',
                                                threshold_value: int = 5,
                                                description_base_text: ty.Optional[str] = None,
                                                date_timestamp: ty.Optional[datetime.date] = None
                                                ) -> ty.List[GeoAttributeInformation]:
        """Generating the attribute of geo-info for Kepler.gl. This method is used for the traffic observation.
        The traffic observation can be, for example, traffic amounts (count), waiting time, etc."""
        seq_geojson_obj = []
        
        if date_timestamp is None:
            date_ = datetime.date.today().isoformat()
        else:
            date_ = date_timestamp.isoformat()
        # end if
        
        sim_time_start, __, time_time_step = self.extract_simulation_time()
                
        n_timesteps = array_traffic_observation.shape[1]
        # for _t in range(0, n_timesteps, observation_every_step_per):
        for _time_step in range(0, n_timesteps, observation_every_step_per):
            # computing in which time-buckets the current timestep belongs to.
            _i_time_bucket: int = _time_step // size_time_bucket
            
            # computing the timestamp.
            _current_hour_min = self._get_simulation_world_time(sim_time_start + _time_step)
            _date_timestamp = f'{date_} {_current_hour_min}'
            
            # for _lane_id in seq_lane_id:
            for _index_route, _route_id in enumerate(seq_lane_id):
                # getting the value
                _value = array_traffic_observation[_index_route, _time_step]
                if _value < threshold_value:
                    continue
                # end if
                
                # getting the geo-json object.
                _geo_json = self._get_geo_json_object(_route_id, lane_or_egde)
                                
                # getting the url link to the Google Map
                _gmap_url = self._generate_google_map_link(_route_id, lane_or_egde)                
                
                # generating the description message.
                _description = f'{description_base_text} at {_route_id} at {_current_hour_min} (time-bucket: {_i_time_bucket})'
                                
                # packing all information to the object.
                __prop = GeoAttributeInformation(
                    route_id=_route_id,
                    geo_json=_geo_json,
                    value=_value,
                    google_map_link=_gmap_url,
                    description=_description,
                    time_bucket=_i_time_bucket,
                    timestamp=_date_timestamp)
                seq_geojson_obj.append(__prop)
            # } end for
        # } end for
        return seq_geojson_obj

        
    def generate_attributes_variable_weight(
        self,
        seq_variable_weight_model: ty.List[VariableWeightObject],
        observation_every_step_per: int,
        n_time_bucket: int,
        lane_or_egde: str = 'lane',
        date_timestamp: ty.Optional[datetime.date] = None) -> ty.List[GeoAttributeInformation]:
        """Generating the attribute of geo-info for Kepler.gl. This method is used for the variable weight model. 
        
        For kepler.gl, refer to the following link:
        https://docs.kepler.gl/docs/user-guides/b-kepler-gl-workflow/a-add-data-to-the-map
        """
        
        seq_geojson_obj = []
        
        if date_timestamp is None:
            date_ = datetime.date.today().isoformat()
        else:
            date_ = date_timestamp.isoformat()
        # end if
        
        sim_time_start, __, time_time_step = self.extract_simulation_time()
        
        for weight_obj in seq_variable_weight_model:
            # comment: the key name is 'lane_id', yet possibly "edge-id".

            # define a timestamp. Use time-bucket
            _i_time_bucket: int = weight_obj.time_bucket
            _timestep_bucket_start = sim_time_start + (_i_time_bucket * n_time_bucket * observation_every_step_per)
            _current_hour_min = self._get_simulation_world_time(_timestep_bucket_start)
            _date_timestamp_bucket_start = f'{date_} {_current_hour_min}'
            
            # get the description
            _description = weight_obj.label
            _value = weight_obj.weight
            
            _lane_id_orig = weight_obj.route_id
            
            _geo_json = self._get_geo_json_object(
                _lane_id_orig, lane_or_egde)

            _gmap_url = self._generate_google_map_link(weight_obj.route_id, lane_or_egde)

            __prop = GeoAttributeInformation(
                route_id=_lane_id_orig,
                geo_json=_geo_json,
                value=_value,
                google_map_link=_gmap_url,
                description=_description,
                time_bucket=_i_time_bucket,
                timestamp=_date_timestamp_bucket_start)
            seq_geojson_obj.append(__prop)
        # end for
        return seq_geojson_obj
