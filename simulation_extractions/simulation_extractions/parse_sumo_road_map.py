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

from dataclasses import dataclass, asdict

from tqdm import tqdm

import datetime
import geojson
import itertools



"""
A class of generating output attributes for the kepler.gl.
"""

class VariableWeightObject(ty.NamedTuple):
    route_id: str
    weight: float
    time_bucket: int
    label: str
    
    
@dataclass
class GeoAttributeInformation:
    route_id: str   # lane-id or edge-id
    geo_json: str  # geojson object
    value: float  # value to be visualised
    google_map_link: str  # google map link to the corresponding route.
    description: str  # a short message of describing the information.
    timestamp: str  # timestamp    
    time_bucket: ty.Optional[int] = None  # time bucket
    
    def to_dict(self):
        return asdict(self)


class KeplerAttributeGenerator(object):
    def __init__(self,
                 path_sumo_net_xml: Path,
                 path_sumo_sim_xml: Path) -> None:
        """A class to generate the attributes for the kepler.gl."""
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
        
        assert _t_position is not None and isinstance(_t_position, (list, tuple)), f'No position found for {route_id}'
        if isinstance(_t_position, list):
            _lat = _t_position[0][0]
            _lon = _t_position[0][1]
        elif isinstance(_t_position, tuple):
            _lat = _t_position[0]
            _lon = _t_position[1]
        # end if
        
        # define a URL link to Google Map.
        _gmap_url = f'https://www.google.com/maps/place/{_lat},{_lon}'
        
        return _gmap_url

    # ------------------------------------------------------------------------------------------
    # APIs
    
    def generate_attributes_traffic_observation(self,
                                                array_traffic_observation: np.ndarray,
                                                seq_lane_id: ty.List[str],
                                                time_step_interval_export: int,
                                                observation_every_step_per: int,
                                                size_time_bucket: int,
                                                lane_or_egde: str = 'lane',
                                                threshold_value: float = 5.0,
                                                mode_value: str = 'agg_sum', 
                                                description_base_text: ty.Optional[str] = None,
                                                date_timestamp: ty.Optional[datetime.date] = None,
                                                aux_array_is_x_more_y: ty.Optional[np.ndarray] = None
                                                ) -> ty.List[GeoAttributeInformation]:
        """Generating the attribute of geo-info for Kepler.gl. This method is used for the traffic observation.
        The traffic observation can be, for example, traffic amounts (count), waiting time, etc.
        
        Args:
            time_step_interval_export: User's parameter. Time step interval to export the data.
            observation_every_step_per: Observation every step in the simulation setting.
        """
        seq_geojson_obj = []
        
        assert mode_value in ('agg_sum', 'observation_at_t'), f'Unknown mode_value: {mode_value}'
        
        
        if date_timestamp is None:
            date_ = datetime.date.today().isoformat()
        else:
            date_ = date_timestamp.isoformat()
        # end if
        
        sim_time_start, __, time_time_step = self.extract_simulation_time()
                
        n_timesteps = array_traffic_observation.shape[1]
        # for _t in range(0, n_timesteps, observation_every_step_per):
        for _time_step in tqdm(range(0, n_timesteps, time_step_interval_export)):
            # computing in which time-buckets the current timestep belongs to.
            _i_time_bucket: int = (_time_step // size_time_bucket) + 1
            
            _is_bucket_first_step = _time_step % size_time_bucket == 0
            _is_bucket_final_step = (_time_step + time_step_interval_export) % size_time_bucket == 0
            
            # computing the timestamp.
            _current_hour_min = self._get_simulation_world_time(sim_time_start + (_time_step * observation_every_step_per))
            _date_timestamp = f'{date_} {_current_hour_min}'
            
            # for _lane_id in seq_lane_id:
            for _index_route, _route_id in enumerate(seq_lane_id):
                # getting the value
                if mode_value == 'agg_sum':
                    __t_previous = _time_step - time_step_interval_export
                    if __t_previous < 0:
                        _value = array_traffic_observation[_index_route, 0:_time_step].sum()
                    else:
                        _value = array_traffic_observation[_index_route, __t_previous:_time_step].sum()
                elif mode_value == 'observation_at_t':
                    _value = array_traffic_observation[_index_route, _time_step]
                else:
                    raise ValueError(f'Unknown mode_value: {mode_value}')
                # end if

                if _value < threshold_value and (not _is_bucket_first_step and not _is_bucket_final_step):
                    # note: I need to save the first record, even though the value is less than the threshold.
                    continue
                # end if
                
                # getting the geo-json object.
                _geo_json = self._get_geo_json_object(_route_id, lane_or_egde)
                                
                # getting the url link to the Google Map
                _gmap_url = self._generate_google_map_link(_route_id, lane_or_egde)                
                
                # refer to the aux_array_is_x_more_y, get which variable is bigger.
                if aux_array_is_x_more_y is not None:
                    if mode_value == 'agg_sum':
                        __t_previous = _time_step - time_step_interval_export
                        if __t_previous < 0:
                            _array_boolean_flag = aux_array_is_x_more_y[_index_route, 0:_time_step]
                        else:
                            _array_boolean_flag = aux_array_is_x_more_y[_index_route, __t_previous:_time_step]
                        # end if

                        from collections import Counter
                        # counting the number of True/False values.
                        _dict_count = dict(Counter(_array_boolean_flag.tolist()))
                        if sum(_dict_count.values()) > 0:
                            __message = 'Frequency. N(X>Y): {}, N(X<Y): {}'.format(_dict_count.get(True, 0), _dict_count.get(False, 0))
                        else:
                            __message = ''
                        # end if
                    elif mode_value == 'observation_at_t':
                        _array_boolean_flag = aux_array_is_x_more_y[_index_route, _time_step]
                        __message = 'X>Y' if _array_boolean_flag else 'X<Y'
                    else:
                        raise ValueError(f'Unknown mode_value: {mode_value}')
                    # end if
                else:
                    __message = ''
                # end if

                # generating the description message.
                _description = f'{description_base_text} at {_route_id} at {_current_hour_min} (time-bucket: {_i_time_bucket}). The value mode is {mode_value}.'
                _description += f' {__message}'

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
        size_time_bucket: int,
        lane_or_egde: str = 'lane',
        date_timestamp: ty.Optional[datetime.date] = None,
        n_bucket_size: int = -1  # the number of buckets to be visualised. If -1, all buckets are visualised.
        ) -> ty.List[GeoAttributeInformation]:
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

        # check the time bucket starting value. Starting at 1 or 0?
        _bucket_index_start: int = min([d.time_bucket for d in seq_variable_weight_model])
        if _bucket_index_start == 0:
            is_add_one = True
        else:
            is_add_one = False
        # end if

        if n_bucket_size == -1:
            seq_bucket_index = list(set([d.time_bucket for d in seq_variable_weight_model]))
        else:
            seq_bucket_index = list(range(_bucket_index_start, n_bucket_size))
        # end if

        dict_bucket_index2weight = {
            key: list(group)
            for key, group in itertools.groupby(sorted(seq_variable_weight_model, key=lambda x: x.time_bucket), key=lambda x: x.time_bucket)
        }

        for __bucket_index in seq_bucket_index:
            # comment: the key name is 'lane_id', yet possibly "edge-id".
            _seq_weight_obj = dict_bucket_index2weight.get(__bucket_index)
            # define a timestamp. Use time-bucket

            if _seq_weight_obj is None:
                if is_add_one:
                    _i_time_bucket: int = __bucket_index + 1
                else:
                    _i_time_bucket: int = __bucket_index
                # end if

                __prop = GeoAttributeInformation(
                    route_id="",
                    geo_json="{}",
                    value=-1,
                    google_map_link="",
                    description="No Variables",
                    time_bucket=_i_time_bucket,
                    timestamp=_date_timestamp_bucket_start)
                seq_geojson_obj.append(__prop)
                continue
            # end if
            for __weight_obj in _seq_weight_obj:
                if is_add_one:
                    _i_time_bucket: int = __weight_obj.time_bucket + 1
                else:
                    _i_time_bucket: int = __weight_obj.time_bucket
                # end if

                _timestep_bucket_start = sim_time_start + (__weight_obj.time_bucket * size_time_bucket * observation_every_step_per)
                _current_hour_min = self._get_simulation_world_time(_timestep_bucket_start)
                _date_timestamp_bucket_start = f'{date_} {_current_hour_min}'
                
                # get the description
                _description = __weight_obj.label
                _value = __weight_obj.weight
                
                _lane_id_orig = __weight_obj.route_id
                
                _geo_json = self._get_geo_json_object(
                    _lane_id_orig, lane_or_egde)

                _gmap_url = self._generate_google_map_link(__weight_obj.route_id, lane_or_egde)

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
        # end for
        return seq_geojson_obj
