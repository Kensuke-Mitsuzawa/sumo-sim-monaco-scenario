# %%
from pathlib import Path
import sumolib
import typing as ty

import shutil

import xml.etree.ElementTree as ET

import logzero
logger = logzero.logger


# %%
PATH_SUMO_NET = "/home/kensuke_mit/sumo-sim-monaco-scenario/simulation_extractions/sumo_configs/f1_scenario/in/most.net.xml"
assert Path(PATH_SUMO_NET).exists(), f"not found: {PATH_SUMO_NET}"


# %%
SOURCE_PATH_ROUTE_DIR = "/home/kensuke_mit/sumo-sim-monaco-scenario/simulation_extractions/sumo_configs/base/until_afternoon/original_config/in/route"
SOURCE_PATH_ADD_DIR = "/home/kensuke_mit/sumo-sim-monaco-scenario/simulation_extractions/sumo_configs/base/until_afternoon/original_config/in/add"

TARGET_PATH_ROUTE_DIR = "/home/kensuke_mit/sumo-sim-monaco-scenario/simulation_extractions/sumo_configs/f1_scenario/in/route"
TARGET_PATH_ADD_DIR = "/home/kensuke_mit/sumo-sim-monaco-scenario/simulation_extractions/sumo_configs/f1_scenario/in/add"


# %%
# copy files from the source to the target
shutil.rmtree(TARGET_PATH_ROUTE_DIR, ignore_errors=True)
shutil.rmtree(TARGET_PATH_ADD_DIR, ignore_errors=True)

shutil.copytree(SOURCE_PATH_ROUTE_DIR, TARGET_PATH_ROUTE_DIR)
shutil.copytree(SOURCE_PATH_ADD_DIR, TARGET_PATH_ADD_DIR)


# %%
# ----------------------------------------------------
# getting prohibited edges
# Load the network
net = sumolib.net.readNet(PATH_SUMO_NET)  # Replace with your network file

# Initialize an empty list to store the prohibited edges
prohibited_edges = []

# Iterate over all edges in the network
for edge in net.getEdges():
    # Check if the edge allows any vehicle class
    if not any(edge.allows(vClass) for vClass in ['passenger', 'bus', 'truck', 'motorcycle', 'bicycle', 'pedestrian']):
        # If the edge does not allow any vehicle class, add it to the list
        prohibited_edges.append(edge)
        logger.debug(f"edge: {edge.getID()}")
    # end if
# end for
# Now, `prohibited_edges` contains all edges that all vehicles are prohibited to pass through


# %%
logger.debug(f'N(prohibited lanes) = {len(prohibited_edges)}')

# %%
seq_prohibited_edges = [__e.getID() for __e in prohibited_edges]

# ----------------------------------------------------

# %% [markdown]
# # Vehicle Route Modifications
# 
# I obatined `prohibited_edges`. I search agents routes. If agents' routes contain the prohibited edges, I delete it from the pre-defined route.


# %%
source_path_route_dir = Path(SOURCE_PATH_ROUTE_DIR)
assert source_path_route_dir.exists(), f"not found: {source_path_route_dir}"
target_path_route_dir = Path(TARGET_PATH_ROUTE_DIR)
assert target_path_route_dir.exists(), f"not found: {target_path_route_dir}"

source_path_add_dir = Path(SOURCE_PATH_ADD_DIR)
assert source_path_add_dir.exists(), f"not found: {source_path_add_dir}"
target_path_add_dir = Path(TARGET_PATH_ADD_DIR)
assert target_path_add_dir.exists(), f"not found: {target_path_add_dir}"


# %% [markdown]
# ----------------------------------------------------
# Bus routes

# %%

# updating bus stops.

s_path_bus_stop_definition = source_path_add_dir / "most.busstops.add.xml"
t_path_bus_stop_definition = target_path_add_dir / "most.busstops.add.xml"

seq_bus_stops_deleted = []

tree_xml = ET.parse(s_path_bus_stop_definition.as_posix())
root = tree_xml.getroot()

for elem in root:
    # do something with each element here
    element_name: str = elem.tag
    if element_name == 'busStop':
        __bus_stop_attrib = elem.attrib
        __bus_stop_id = __bus_stop_attrib['id']
        __bus_stop_lane = __bus_stop_attrib['lane']
        __edge_id, __ = __bus_stop_lane.split('_')
        if __edge_id in seq_prohibited_edges:
            seq_bus_stops_deleted.append(__bus_stop_id)
            logger.debug(f"bus-stop-id: {__bus_stop_id}, Set invisible.")
            root.remove(elem)
        # end if
    # end if
# end for

logger.debug(f"deleted bus-stops: {seq_bus_stops_deleted}")

tree_xml.write(t_path_bus_stop_definition.as_posix())
logger.debug(f"done: {t_path_bus_stop_definition}")

# updating the bus routing.
s_path_bus_route_definition = source_path_route_dir / "most.buses.flows.xml"
assert s_path_bus_route_definition.exists(), f"not found: {s_path_bus_route_definition}"
t_path_bus_route_definition = target_path_route_dir / "most.buses.flows.xml"

tree_xml = ET.parse(s_path_bus_route_definition.as_posix())
root = tree_xml.getroot()

for elem in root:
    # do something with each element here
    element_name: str = elem.tag
    if element_name == 'route':
        __route_attrib = elem.attrib
        __seq_edges: ty.List[str] = __route_attrib['edges'].split()
        # redundant approach to remove the prohibited edges, but I do this style to keep the route ordering.
        __seq_edges_updated = [__edge for __edge in __seq_edges if __edge not in seq_prohibited_edges]
        # logger.debug(f"Bus-No: {__route_attrib['id']}, before-mod-N(edges): {len(__seq_edges)}, updated-N(edges): {len(__seq_edges_updated)}")
        # updating edge information.
        __route_attrib['edges'] = " ".join(__seq_edges_updated)
        
        # deleting the bus stop.
        __seq_stop_elem = elem.findall('stop')
        for _elem_stop in __seq_stop_elem:
            _stop_id = _elem_stop.attrib['busStop']
            if _stop_id in seq_bus_stops_deleted:
                elem.remove(_elem_stop)
                logger.debug(f"removed: {_elem_stop}")
            # end if
        # end for
    # end if
# end for

tree_xml.write(t_path_bus_route_definition.as_posix())
logger.debug(f"done: {t_path_bus_route_definition}")


# ----------------------------------------------------

# %% [markdown]
# Updating parking information, setting the max-capacity=0.
# Getting the parking-id where the parking locates on the prohibited routing.

# %%
s_path_parking_add = Path(source_path_add_dir) / 'most.parking.allvisible.add.xml'
t_path_parking_add = Path(target_path_add_dir) / 'most.parking.allvisible.add.xml'

tree_xml = ET.parse(s_path_parking_add.as_posix())
root = tree_xml.getroot()

seq_parking_id_unavailable = []

for elem in root:
    # do something with each element here
    element_name: str = elem.tag
    if element_name == 'parkingArea':
        _p_id = elem.attrib['id']
        _p_lane_id = elem.attrib['lane']
        _p_edge_id = _p_lane_id.split('_')[0]
        # logger.debug(_p_edge_id)
        # when the parking area is located on the prohibited edge, 
        if _p_edge_id in seq_prohibited_edges:
            seq_parking_id_unavailable.append(_p_id)
            # set the parking capacity to 0.
            elem.attrib['roadsideCapacity'] = "0"
            logger.debug(f"parking-id: {_p_id}, Set 0 capacity.")
        # end if
    # end if
    # updating rerouter
    if element_name == 'rerouter':
        iter_rerouter_def = elem.findall('interval')
        for __interval_def in iter_rerouter_def:
            iter_rerouter_def = __interval_def.findall('parkingAreaReroute')
            for _elem_park_reroute in iter_rerouter_def:
                _p_id = _elem_park_reroute.attrib['id']
                if _p_id in seq_parking_id_unavailable:
                    _elem_park_reroute.attrib['visible'] = "false"
                    # logger.debug(f"parking-id: {_p_id}, Set invisible.")
                # end if
            # end for
        # end for
    # end if
# end for

tree_xml.write(t_path_parking_add.as_posix())
logger.debug(f"done: {t_path_parking_add}")


# %% [markdown]
# updating commercial vehicle routing.

# %%
# updating commercial vehicle routing.
s_path_commercial_route = source_path_route_dir / "most.commercial.rou.xml"
assert s_path_commercial_route.exists(), f"not found: {s_path_commercial_route}"

t_path_commercial_route = target_path_route_dir / "most.commercial.rou.xml"

# %%
# updating the bus routing.
tree_xml = ET.parse(s_path_commercial_route.as_posix())
root = tree_xml.getroot()

for elem in root:
    # do something with each element here
    element_name: str = elem.tag
    if element_name == 'vehicle':
        _v_id = elem.attrib['id']
        
        elem_route = elem.find('route')
        __route_attrib = elem_route.attrib
        
        __seq_edges: ty.List[str] = __route_attrib['edges'].split()
        # redundant approach to remove the prohibited edges, but I do this style to keep the route ordering.
        __seq_edges_updated = [__edge for __edge in __seq_edges if __edge not in seq_prohibited_edges]
        # logger.debug(f"No: {_v_id}, before-mod-N(edges): {len(__seq_edges)}, updated-N(edges): {len(__seq_edges_updated)}")
        # updating edge information.
        __route_attrib['edges'] = " ".join(__seq_edges_updated)
        
        # updating "stop" element if the parking lot is located on the prohibited edge.
        # <stop parkingArea="1151" until="45004" />
        elem_stop = elem.find('stop')
        if elem_stop is not None:
            _p_id = elem_stop.attrib['parkingArea']
            if _p_id in seq_parking_id_unavailable:
                elem.remove(elem_stop)
                logger.debug(f"removed: {elem_stop}")
    # end if
# end for

tree_xml.write(t_path_commercial_route.as_posix())        
logger.debug(f"done: {t_path_commercial_route}")

# %% [markdown]
# Pedestrian routes.

# %%
s_path_pedestrial_route = source_path_route_dir / "most.pedestrian.rou.xml"
assert s_path_pedestrial_route.exists(), f"not found: {s_path_pedestrial_route}"

t_path_pedestrial_route = target_path_route_dir / "most.pedestrian.rou.xml"

# %%
"""
    <person id="pedestrian_1-1-pt_7251" type="pedestrian" depart="18001">
        <walk edges="-152836#4 -152836#5 153171#1 153151 152870#2 153160 -152969#1 -152969#0" busStop="131101"/>
        <ride busStop="131086" lines="4:SaintRoman" intended="bus_4:SaintRoman.3" depart="18470.0"/>
        <walk edges="-152349 -152590#2 152590#1"/>
    </person>
"""

"""
    <vehicle id="pedestrian_1-1-veh_105_tr" type="ptw" depart="triggered" departLane="best" arrivalPos="79.85744050395286">
        <route edges="152870#0 152870#1 152870#2 152870#3 -152832#6 -152832#5 -152832#4 -152832#3 -152832#2 -152832#1 -152832#0 -152987#1 -152987#0 -152795#1 152808 152959#3 152959#4 152959#5 152779 152818#0 152818#1 152818#2 152818#3 152816#0 152816#1 152816#2 152816#3 152816#4 152816#5 -152810#2 -152810#1 -152804"/>
        <stop parkingArea="1147" until="43492"/>
    </vehicle>
"""


# updating the bus routing.
tree_xml = ET.parse(s_path_pedestrial_route.as_posix())
root = tree_xml.getroot()

for elem in root:
    # do something with each element here
    element_name: str = elem.tag
    # process for person tag
    if element_name == 'person':
        # updating walk route information
        elem_walk_def = elem.findall('walk')
        for _elem_walk in elem_walk_def:
            # deleting the invalid edges from the route.
            _seq_edges = _elem_walk.attrib['edges'].split()
            _seq_edges_updated = [_edge for _edge in _seq_edges if _edge not in seq_prohibited_edges]
            _elem_walk.attrib['edges'] = " ".join(_seq_edges_updated)
            if len(_seq_edges_updated) < len(_seq_edges):
                logger.debug(f"pedestrian: {elem.attrib['id']}, before-mod-N(edges): {len(_seq_edges)}, updated-N(edges): {len(_seq_edges_updated)}")
            # end if
            
            # if deleted bus-stops is in the route, I delete the bus-stop.
            if 'busStop' in _elem_walk.attrib and _elem_walk.attrib['busStop'] in seq_bus_stops_deleted:
                del _elem_walk.attrib['busStop']
            # end if
        # end for
        
        # updating ride information
        elem_ride_def = elem.findall('ride')
        for _elem_ride in elem_ride_def:
            _ride_edge_from = ''
            _ride_edge_to = ''
            if 'from' in _elem_ride.attrib:
                _ride_edge_from = _elem_ride.attrib['from']
            # end if
            if 'to' in _elem_ride.attrib:
                _ride_edge_to = _elem_ride.attrib['to']
            # end if
            if _ride_edge_from in seq_prohibited_edges or _ride_edge_to in seq_prohibited_edges:
                elem.remove(_elem_ride)
                logger.debug(f"removed: {_elem_ride}")
            # end if
            
            # bus stop.
            if 'busStop' in _elem_ride.attrib and _elem_ride.attrib['busStop'] in seq_bus_stops_deleted:
                elem.remove(_elem_ride)
            # end if
        # end for
    # end if
    
    if element_name == 'vehicle':
        elem_route = elem.find('route')
        assert elem_route is not None, f"not found: route"
        __route_attrib = elem_route.attrib
        
        __seq_edges: ty.List[str] = __route_attrib['edges'].split()
        # redundant approach to remove the prohibited edges, but I do this style to keep the route ordering.
        __seq_edges_updated = [__edge for __edge in __seq_edges if __edge not in seq_prohibited_edges]
        # logger.debug(f"No: {_v_id}, before-mod-N(edges): {len(__seq_edges)}, updated-N(edges): {len(__seq_edges_updated)}")
        # updating edge information.
        __route_attrib['edges'] = " ".join(__seq_edges_updated)
        elem_route.attrib = __route_attrib
        
        # updating stop info
        elem_stop = elem.find('stop')
        if elem_stop is not None:
            _p_id = elem_stop.attrib['parkingArea']
            if _p_id in seq_parking_id_unavailable:
                elem.remove(elem_stop)
                logger.debug(f"removed: {elem_stop}")
            # end if
        # end if
# end for

tree_xml.write(t_path_pedestrial_route.as_posix())        
logger.debug(f"done: {t_path_pedestrial_route}")

# %% [markdown]
# special agents

# %%
s_path_special_route = source_path_route_dir / "most.special.rou.xml"
assert s_path_special_route.exists(), f"not found: {s_path_special_route}"

t_path_special_route = target_path_route_dir / "most.special.rou.xml"

# %%
# updating the bus routing.
tree_xml = ET.parse(s_path_special_route.as_posix())
root = tree_xml.getroot()

for elem in root:
    # do something with each element here
    element_name: str = elem.tag    
    if element_name == 'vehicle':
        elem_route = elem.find('route')
        assert elem_route is not None, f"not found: route"
        __route_attrib = elem_route.attrib
        
        __seq_edges: ty.List[str] = __route_attrib['edges'].split()
        # redundant approach to remove the prohibited edges, but I do this style to keep the route ordering.
        __seq_edges_updated = [__edge for __edge in __seq_edges if __edge not in seq_prohibited_edges]
        # logger.debug(f"No: {_v_id}, before-mod-N(edges): {len(__seq_edges)}, updated-N(edges): {len(__seq_edges_updated)}")
        # updating edge information.
        __route_attrib['edges'] = " ".join(__seq_edges_updated)
        elem_route.attrib = __route_attrib
# end for

tree_xml.write(t_path_special_route.as_posix())        
logger.debug(f"done: {t_path_special_route}")


