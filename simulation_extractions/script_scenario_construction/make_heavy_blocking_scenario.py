"""A script file for creating a scenario for the Sardine scenario.
"""


import toml
from pathlib import Path
import shutil

import xml.etree.ElementTree as ET

import logzero

logger = logzero.logger


PATH_TOML = Path("config_heavy_blocking_scenario.toml")


assert PATH_TOML.exists(), f"File {PATH_TOML} does not exist"
config_obj = toml.load(PATH_TOML)

path_dir_scenario_source = Path(config_obj['PATH_DIR_SCENARIO_SOURCE'])
assert path_dir_scenario_source.exists(), f"Directory {path_dir_scenario_source} does not exist"

scenario_name: str = config_obj['SCENARIO_NAME']

path_dir_scenario = path_dir_scenario_source.parent

if (path_dir_scenario / scenario_name).exists():
    logger.info('deleting the existing scenario directory.')
    shutil.rmtree(path_dir_scenario / scenario_name)
# end if

path_dir_scenario.mkdir(exist_ok=True)
shutil.copytree(path_dir_scenario_source, path_dir_scenario / scenario_name)
logger.info(f"Scenario {scenario_name} created in {path_dir_scenario}")

shutil.copy('rerouter_heavy_blocking.xml', path_dir_scenario / scenario_name / 'rerouter.xml')
logger.info(f"Rerouter for scenario {scenario_name} created in {path_dir_scenario / scenario_name / 'rerouter.xml'}")

# adding rerouter to the sumo.cfg
name_sumo_cfg: str = config_obj['NAME_SUMO_CFG']
path_sumo_cfg = path_dir_scenario / scenario_name / name_sumo_cfg

xml_obj = ET.parse(path_sumo_cfg.as_posix())
root = xml_obj.getroot()

elem_input = root.find('input')
assert elem_input is not None, f"Element input not found in {path_sumo_cfg}"
elem_additional = elem_input.find('additional-files')
assert elem_additional is not None, f"Element additional-files not found in {path_sumo_cfg}"
str_list_files: str = elem_additional.attrib['value']
str_list_files += ',rerouter.xml'
elem_additional.attrib['value'] = str_list_files
xml_obj.write(path_sumo_cfg.as_posix())
logger.info(f"Rerouter added to {path_sumo_cfg}")



