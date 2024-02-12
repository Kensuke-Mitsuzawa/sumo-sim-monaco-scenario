import xml.etree.ElementTree as ET
from pathlib import Path
import typing as ty


"""Handling of config files"""


def update_output_path(path_sumo_config: Path, 
                       path_output_dir: Path,
                       path_saved_sumo_config: ty.Optional[Path] = None):
    """Over-writing a path to a new output directory.
    """
    if path_saved_sumo_config is None:
        path_saved_sumo_config = path_sumo_config
    # end if
    
    
    # Parse the XML file
    tree = ET.parse(path_sumo_config)
    root = tree.getroot()

    # Find the 'output-prefix' element and update its 'value' attribute
    for elem in root.iter('output-prefix'):
        elem.set('value', path_output_dir.as_posix())

    # Write the updated XML back to the file
    tree.write(path_saved_sumo_config.as_posix())



def update_seed_value(path_sumo_config: Path, 
                      seed_value: int,
                      path_saved_sumo_config: ty.Optional[Path] = None):
    if path_saved_sumo_config is None:
        path_saved_sumo_config = path_sumo_config
    # end if
    
    # Parse the XML file
    tree = ET.parse(path_sumo_config)
    root = tree.getroot()

    # Find the 'output-prefix' element and update its 'value' attribute
    for elem in root.iter('seed'):
        elem.set('value', str(seed_value))

    # Write the updated XML back to the file
    tree.write(path_saved_sumo_config.as_posix())



def update_end_time_value(path_sumo_config: Path, 
                          endtime_value: int = '14900',
                          path_saved_sumo_config: ty.Optional[Path] = None):
    if path_saved_sumo_config is None:
        path_saved_sumo_config = path_sumo_config
    # end if
    
    # Parse the XML file
    tree = ET.parse(path_sumo_config)
    root = tree.getroot()

    # Find the 'output-prefix' element and update its 'value' attribute
    for elem in root.iter('end'):
        elem.set('value', endtime_value)

    # Write the updated XML back to the file
    tree.write(path_saved_sumo_config.as_posix())



def assert_fcd_definition(path_sumo_config: Path):
    """Assert that the fcd definition is present in the sumo config file.
    """
    # Parse the XML file
    tree = ET.parse(path_sumo_config)
    root = tree.getroot()

    elem_output = root.find('output')
    assert elem_output is not None, f'No `output` element found in {path_sumo_config}'
    elem_fcd = elem_output.find('fcd-output')
    assert elem_fcd is not None, f'No `fcd-output` element found in {path_sumo_config}'
    
    return False


def assert_additional_xml_file(path_additional_xml: Path):
    """Assert that the additional.xml file is present.
    """
    assert path_additional_xml.exists(), f'No additional.xml file found at {path_additional_xml}'

    # Parse the XML file
    tree = ET.parse(path_additional_xml)
    root = tree.getroot()
    
    elem_edge = root.find('edgeData')
    elem_lane = root.find('laneData')
    
    assert elem_edge is not None, f'No `edgeData` element found in {path_additional_xml}'
    assert elem_lane is not None, f'No `laneData` element found in {path_additional_xml}'
    
    return False