import typing as ty
import xml.etree.ElementTree as ET
import logzero

import numpy as np

import collections
from itertools import chain

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

logzero.loglevel(logzero.INFO)
# logzero.loglevel(logzero.DEBUG)
logger = logzero.logger


class LaneObservationContainer(ty.NamedTuple):
    lane_id: str
    edge_id: str
    time_begin: float
    time_end: float
    count_vehicles: float
    density: float


class TripInfoContainer(ty.NamedTuple):
    vehicle_id: str
    vehicle_type: str
    time_depart: float
    time_arrival: float
    depart_delay: float
    duration: float
    waiting_time: float
    stop_time: float
    time_loss: float
    routeLength: float
    departLane: str
    arrivalLane: str


def parse_lane_observation(path_lane_observation: Path,
                           target_egde_id: ty.Optional[str] = None
                           ) -> ty.List[LaneObservationContainer]:
    """Parse the lane-observation file.
    """
    seq_lane_observations = []
    
    iter_elems = ET.iterparse(path_lane_observation.as_posix())
    for event, _elem in iter_elems:
        # _iter_interval_node = _elem.findall('interval')
        # for _node_interval in _iter_interval_node:
        if _elem.tag == 'interval':
            _time_step_begin = float(_elem.attrib['begin'])
            _time_step_end = float(_elem.attrib['end'])
            # extracting lane-nodes
            _iter_lane_node = _elem.findall('*/lane')
            for _node_lane in _iter_lane_node:
                _lane_id = _node_lane.attrib['id']
                if '_' in _lane_id:
                    _edge_id = _lane_id.split('_')[0]
                else:
                    _edge_id = ''
                # end if
                
                if target_egde_id is not None and target_egde_id not in _lane_id:
                    logger.debug(f'Found target edge id -> {_edge_id}')
                    continue
                # end if
                
                # collecting metrics
                _count = float(_node_lane.attrib['sampledSeconds'])
                _density = float(_node_lane.attrib['laneDensity'])
                
                _obs_container = LaneObservationContainer(
                    lane_id=_lane_id,
                    edge_id=_edge_id,
                    time_begin=_time_step_begin,
                    time_end=_time_step_end,
                    count_vehicles=_count,
                    density=_density)
                seq_lane_observations.append(_obs_container)
            # end for
        # end for
    # end for
    return seq_lane_observations


def parse_tripinfo(path_tripinfo: Path, 
                   target_vehicle_id: ty.Optional[ty.List[str]] = None
                   ) -> ty.List[TripInfoContainer]:
    seq_trip_info = []
    
    iter_elems = ET.iterparse(path_tripinfo.as_posix())
    for event, _elem in iter_elems:
        if _elem.tag == 'tripinfo':
            _id = _elem.attrib['id']
            if target_vehicle_id is not None and _id not in target_vehicle_id:
                continue
            # end if
            
            depart_lane = _elem.attrib['departLane']
            arrival_lane = _elem.attrib['arrivalLane']
            
            _trip_info = TripInfoContainer(
                vehicle_id=_id,
                vehicle_type=_elem.attrib['vType'],
                time_depart=float(_elem.attrib['depart']),
                time_arrival=float(_elem.attrib['arrival']),
                depart_delay=float(_elem.attrib['departDelay']),
                duration=float(_elem.attrib['duration']),
                waiting_time=float(_elem.attrib['waitingTime']),
                stop_time=float(_elem.attrib['stopTime']),
                time_loss=float(_elem.attrib['timeLoss']),
                routeLength=float(_elem.attrib['routeLength']),
                departLane=depart_lane,
                arrivalLane=arrival_lane)
            seq_trip_info.append(_trip_info)
        # end if
    # end for
    return seq_trip_info


def investigation_highway_a8_diff(path_vehroute_x: Path, 
                                  path_vehroute_y: Path,
                                  target_highway_edge_italy_france: str = '152276',
                                  target_highway_edge_france_italy: str = '152277'):
    """
    Investigate the difference between the traffic path of the highway A8.
    In other words, vehicles took which route by the blocking of the A8?
    """
    class TargetVehicleContainer(ty.NamedTuple):
        scenario: str
        vehicle_id: str
        route: str
        vehicle_type: str
        time_depart: str  # I exepct time-step. but, can be triggered
        time_arrival: str
        routes: ty.Optional[ty.List[str]] = None
    # end if
            
    def _collect_vehicles_scenario_x() -> ty.List[TargetVehicleContainer]:
        # On the scenario X, I list up vehicle ids that passed the A8.
        seq_v_container_via_a8 = []
        
        iter_elems_x = ET.iterparse(path_vehroute_x.as_posix())
        for event, _elem in iter_elems_x:
            # 'vehicle' is the only target. I do not care person in this investigation.
            if _elem.tag == 'vehicle':
                logger.debug(f'Vehicle ID -> {_elem.attrib["id"]}')
                _vehicle_id = _elem.attrib["id"]
                # get the xml element of routeing.
                _elem_veh_route = _elem.find('route')
                if _elem_veh_route is None:
                    continue
                # end if
                _seq_edge_ids = _elem_veh_route.attrib['edges'].split()
                if target_highway_edge_italy_france in _seq_edge_ids:
                    logger.debug(f'Vehicle ID -> {_elem.attrib["id"]}')
                    logger.debug(f'Route -> {_seq_edge_ids}')
                    _v_container = TargetVehicleContainer(
                        scenario='X',
                        vehicle_id=_vehicle_id,
                        route=target_highway_edge_italy_france,
                        vehicle_type=_elem.attrib['type'],
                        time_depart=_elem.attrib['depart'],
                        time_arrival=_elem.attrib['arrival'])
                    seq_v_container_via_a8.append(_v_container)
                elif target_highway_edge_france_italy in _seq_edge_ids:
                    logger.debug(f'Vehicle ID -> {_elem.attrib["id"]}')
                    logger.debug(f'Route -> {_seq_edge_ids}')
                    _v_container = TargetVehicleContainer(
                        scenario='X',
                        vehicle_id=_vehicle_id,
                        route=target_highway_edge_italy_france,
                        vehicle_type=_elem.attrib['type'],
                        time_depart=_elem.attrib['depart'],
                        time_arrival=_elem.attrib['arrival'])                
                    seq_v_container_via_a8.append(_v_container)
                else:
                    pass
                # end if
            # end if
        # end for
        return seq_v_container_via_a8
    # end def
    
    def _get_vehicle_route_scenario_y(
        seq_vehicles_x: ty.List[TargetVehicleContainer]) -> ty.List[TargetVehicleContainer]:
        """Extract routes in scenario Y that A8-routes-scenario X vehicles took.
        """
        set_vehicle_ids_x = set([_v.vehicle_id for _v in seq_vehicles_x])
        seq_v_container_via_a8 = []
        
        iter_elems_x = ET.iterparse(path_vehroute_y.as_posix())
        for event, _elem in iter_elems_x:
            # 'vehicle' is the only target. I do not care person in this investigation.
            if _elem.tag == 'vehicle':
                logger.debug(f'Vehicle ID -> {_elem.attrib["id"]}')
                _vehicle_id = _elem.attrib["id"]
                if _vehicle_id in set_vehicle_ids_x:
                    # get the xml element of routeing.
                    _elem_veh_route = _elem.find('route')
                    if _elem_veh_route is None:
                        continue
                    # end if
                    _seq_edge_ids = _elem_veh_route.attrib['edges'].split()
                    logger.debug(f'Vehicle ID -> {_elem.attrib["id"]}')
                    logger.debug(f'Route -> {_seq_edge_ids}')
                    _v_container = TargetVehicleContainer(
                        scenario='Y',
                        vehicle_id=_vehicle_id,
                        route=target_highway_edge_italy_france,
                        vehicle_type=_elem.attrib['type'],
                        time_depart=_elem.attrib['depart'],
                        time_arrival=_elem.attrib['arrival'],
                        routes=_seq_edge_ids) 
                    seq_v_container_via_a8.append(_v_container)
                # end if
            # end if
        # end for
        return seq_v_container_via_a8
    # end def
    
    seq_vehicles_x = _collect_vehicles_scenario_x()
    logger.info(f'Extracted {len(seq_vehicles_x)} vehicles from scenario X.')
    seq_vehicle_route_y = _get_vehicle_route_scenario_y(seq_vehicles_x)
    
    # get statistics of routing.
    seq_edge_ids = list(chain.from_iterable([v_container.routes for v_container in seq_vehicle_route_y if v_container.routes is not None]))
    counter_edge = collections.Counter(seq_edge_ids)
    
    # grouping the routes
    seq_routes = [tuple(v_container.routes) for v_container in seq_vehicle_route_y if v_container.routes is not None]
    logger.info(f'Alternative routes types = {len(set(seq_routes))}')
    counter_routes = collections.Counter(seq_routes)

    # by thie investigation, no vechiles took the alternative route into Monaco.
    


def investigation_edge_lycee_albert_premiere(
    path_sumo_output_x: Path,
    path_sumo_output_y: Path,
    path_dir_fig_output: Path,
    file_name_tripinfo: str = 'most.tripinfo.xml',
    file_name_vehroute: str = 'most.vehroute.xml',
    file_name_lane_observation: str = 'most.lane-observation.xml',
    target_edge_id: str = '152763',
    size_time_bucket: int = 5000,
    t_simulation_start: int = 14400,
    t_simulation_end: int = 50400,):
    
    path_vehroute_x = path_sumo_output_x / file_name_vehroute
    path_vehroute_y = path_sumo_output_y / file_name_vehroute
    
    class TargetVehicleContainer(ty.NamedTuple):
        scenario: str
        vehicle_id: str
        destination_edge_id: str
        vehicle_type: str
        time_depart: str  # I exepct time-step. but, can be triggered
        time_arrival: str
        routes: ty.Optional[ty.List[str]] = None        
    # end def
    
    # first question: are there differences in vehicle ids that comes to the edge?
    def _collect_vehicles(path_vehroute: Path) -> ty.List[TargetVehicleContainer]:
        # On the scenario X, I list up vehicle ids that passed the A8.
        seq_v_container = []
        
        iter_elems_x = ET.iterparse(path_vehroute.as_posix())
        for event, _elem in iter_elems_x:
            # 'vehicle' is the only target. I do not care person in this investigation.
            if _elem.tag == 'vehicle':
                logger.debug(f'Vehicle ID -> {_elem.attrib["id"]}')
                _vehicle_id = _elem.attrib["id"]
                # get the xml element of routeing.
                _elem_veh_route = _elem.find('route')
                if _elem_veh_route is None:
                    continue
                # end if
                
                _seq_edge_ids = _elem_veh_route.attrib['edges'].split()
                # filtering by the given edge id
                if target_edge_id not in _seq_edge_ids:
                    continue
                # end if
                
                _edge_destination = _seq_edge_ids[-1]
                _v_container = TargetVehicleContainer(
                    scenario='X',
                    vehicle_id=_vehicle_id,
                    destination_edge_id=_edge_destination,
                    vehicle_type=_elem.attrib['type'],
                    time_depart=_elem.attrib['depart'],
                    time_arrival=_elem.attrib['arrival'])
                seq_v_container.append(_v_container)
            # end if
        # end for
        return seq_v_container
    # end def
        
    # collect the route information
    seq_route_container_x = _collect_vehicles(path_vehroute_x)
    seq_route_container_y = _collect_vehicles(path_vehroute_y)
    # check if differences in vehicle ids
    set_v_id_x = set([v_container.vehicle_id for v_container in seq_route_container_x])
    set_v_id_y = set([v_container.vehicle_id for v_container in seq_route_container_y])
    set_v_id_diff = set_v_id_x.symmetric_difference(set_v_id_y)
    logger.info(f'N(V-id-X)={len(set_v_id_x)}, N(V-id-Y)={len(set_v_id_y)}, N(V-id-diff)={len(set_v_id_diff)}')
    
    # count: how many agents set the edge as the destination?
    seq_set_edge_destination_x = [v_container.vehicle_id for v_container in seq_route_container_x if v_container.destination_edge_id == target_edge_id]
    seq_set_edge_destination_y = [v_container.vehicle_id for v_container in seq_route_container_y if v_container.destination_edge_id == target_edge_id]
    logger.info(f'-'*30)
    logger.info(f'How many agents set the edge={target_edge_id} as the destination?')
    logger.info(f'N(X)={len(seq_set_edge_destination_x)}, N(Y)={len(seq_set_edge_destination_y)}')
    # -------------------------------------------------------------------------------
    
    # check delay in travelling time.
    # I obtain the vehicle ids. Vehicle-ids that traveled via the target edge.
    set_vehicle_ids_common = set(set_v_id_x).intersection(set(set_v_id_y))
    seq_trip_info_x = parse_tripinfo(path_sumo_output_x / file_name_tripinfo, target_vehicle_id=list(set_vehicle_ids_common))
    seq_trip_info_y = parse_tripinfo(path_sumo_output_y / file_name_tripinfo, target_vehicle_id=list(set_vehicle_ids_common))
    df_tripinfo_x = pd.DataFrame([o._asdict() for o in seq_trip_info_x])
    df_tripinfo_x['variable'] = 'x'
    df_tripinfo_y = pd.DataFrame([o._asdict() for o in seq_trip_info_y])
    df_tripinfo_y['variable'] = 'y'
    df_tripinfo = pd.concat([df_tripinfo_x, df_tripinfo_y])
    
    logger.info(f'-'*30)
    sum_route_length_x = df_tripinfo_x['routeLength'].sum()
    sum_route_length_y = df_tripinfo_y['routeLength'].sum()
    logger.info(f'route_length,{sum_route_length_x},{sum_route_length_y}')
    sum_time_loss_x = df_tripinfo_x['time_loss'].sum()
    sum_time_loss_y = df_tripinfo_y['time_loss'].sum()
    logger.info(f'time_loss,{sum_time_loss_x},{sum_time_loss_y}')
    sum_duration_x = df_tripinfo_x['duration'].sum()
    sum_duration_y = df_tripinfo_x['duration'].sum()
    logger.info(f'duration,{sum_duration_x},{sum_duration_y}')
    logger.info(f'waiting_time,{df_tripinfo_x["waiting_time"].sum()},{df_tripinfo_y["waiting_time"].sum()}')
    

    def _plot_trip_ino(seq_tripinfo_x: ty.List[TripInfoContainer],
                       seq_tripinfo_y: ty.List[TripInfoContainer],
                       metric_name: str):
        """Plot the trip information."""
        df_tripinfo_x = pd.DataFrame([o._asdict() for o in seq_tripinfo_x])
        df_tripinfo_x['variable'] = 'x'
        df_tripinfo_y = pd.DataFrame([o._asdict() for o in seq_tripinfo_y])
        df_tripinfo_y['variable'] = 'y'
        df_tripinfo = pd.concat([df_tripinfo_x, df_tripinfo_y])
        fig, ax = plt.subplots()
        sns.barplot(data=df_tripinfo, x='vehicle_type', y=metric_name, hue='variable', ax=ax, alpha=0.5)
        ax.set_title(f'Metric={metric_name}. Vehicles via edge={target_edge_id}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_yscale("log")        
        path_fig = path_dir_fig_output / f'tripinfo_agg_{metric_name}.png'
        fig.savefig(path_fig.as_posix(), bbox_inches='tight')
        logger.info(f'Saved the figure at {path_fig}')
    # end def
    _plot_trip_ino(seq_trip_info_x, seq_trip_info_y, 'time_loss')
    _plot_trip_ino(seq_trip_info_x, seq_trip_info_y, 'duration')
    _plot_trip_ino(seq_trip_info_x, seq_trip_info_y, 'waiting_time')
    _plot_trip_ino(seq_trip_info_x, seq_trip_info_y, 'stop_time')
    _plot_trip_ino(seq_trip_info_x, seq_trip_info_y, 'depart_delay')
    _plot_trip_ino(seq_trip_info_x, seq_trip_info_y, 'routeLength')
    
    # -------------------------------------------------------------------------------
    # count: how many agents had the delay in arriving?
    # X-axis: time-step, Y-axis: number of vehicles
    logger.debug(f'-'*30)
    logger.info(f'Parsing lane-observation files...')
    seq_lane_obs_x = parse_lane_observation(
        path_sumo_output_x / file_name_lane_observation,
        target_egde_id=target_edge_id)
    seq_lane_obs_y = parse_lane_observation(
        path_sumo_output_y / file_name_lane_observation,
        target_egde_id=target_edge_id)

    # time_step_min = min([obs.time_begin for obs in seq_lane_obs_x] + [obs.time_begin for obs in seq_lane_obs_y])
    # time_step_max = max([obs.time_begin for obs in seq_lane_obs_x] + [obs.time_begin for obs in seq_lane_obs_y])
    # re-structing the observation data.
    # there are two lanes, with minus and without minus.
    records_x = pd.DataFrame([
        {'lane_id': o.lane_id, 't': o.time_begin, 'count': o.count_vehicles, 'density': o.density, 'variable': 'x'} 
        for o in seq_lane_obs_x])
    records_y = pd.DataFrame([
            {'lane_id': o.lane_id, 't': o.time_begin, 'count': o.count_vehicles, 'density': o.density, 'variable': 'y'} 
            for o in seq_lane_obs_y])
    # visualizing count of vehicles (lane +)
    df_vis_plus_x = records_x[~records_x['lane_id'].str.contains('-')]
    df_vis_plus_y = records_y[~records_y['lane_id'].str.contains('-')]
    df_vis_lane_plus = pd.concat([df_vis_plus_x, df_vis_plus_y])
    fig, ax = plt.subplots()
    sns.lineplot(data=df_vis_lane_plus, x='t', y='count', hue='variable', ax=ax, alpha=0.5)
    # drawing vertical lines per 5000 steps
    for _t_line in np.arange(t_simulation_start, t_simulation_end, size_time_bucket):
        ax.axvline(_t_line, color='green', linestyle='--')
    # end for
    __path_fig_output = path_dir_fig_output / 'count_vehicles_lane_plus.png'
    fig.savefig(__path_fig_output.as_posix(), bbox_inches='tight')
    logger.info(f'Saved the figure at {__path_fig_output}')
    
    # visualizing count of vehicles (lane minus)
    df_vis_minus_x = records_x[records_x['lane_id'].str.contains('-')]
    df_vis_minus_y = records_y[records_y['lane_id'].str.contains('-')]
    df_vis_lane_minus = pd.concat([df_vis_minus_x, df_vis_minus_y])
    fig, ax = plt.subplots()
    sns.lineplot(data=df_vis_lane_minus, x='t', y='count', hue='variable', ax=ax, alpha=0.5)
    # drawing vertical lines per 5000 steps
    for _t_line in np.arange(t_simulation_start, t_simulation_end, size_time_bucket):
        ax.axvline(_t_line, color='green', linestyle='--')
    # end for
    __path_fig_output = path_dir_fig_output / 'count_vehicles_lane_minus.png'
    fig.savefig(__path_fig_output.as_posix(), bbox_inches='tight')
    logger.info(f'Saved the figure at {__path_fig_output}')    


def main(
    path_sumo_output_x: Path,
    path_sumo_output_y: Path,
    file_name_tripinfo: str = 'most.tripinfo.xml',
    file_name_vehroute: str = 'most.vehroute.xml',
    file_name_lane_observation: str = 'most.lane-observation.xml'):
    
    # investigation_highway_a8_diff(
    #     path_sumo_output_x / file_name_vehroute,
    #     path_sumo_output_y / file_name_vehroute)
    import tempfile
    path_dir_output = Path(tempfile.mkdtemp())
    
    investigation_edge_lycee_albert_premiere(
        path_sumo_output_x,
        path_sumo_output_y,
        path_dir_fig_output=path_dir_output)


if __name__ == '__main__':
    _path_sumo_output_x = Path('/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/sumo_cfg/0_debug/x/out')
    _path_sumo_output_y = Path('/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/sumo_cfg/0_debug/y/out')
    main(
        _path_sumo_output_x,
        _path_sumo_output_y)