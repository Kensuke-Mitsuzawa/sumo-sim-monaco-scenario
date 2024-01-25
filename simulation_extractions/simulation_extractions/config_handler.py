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
