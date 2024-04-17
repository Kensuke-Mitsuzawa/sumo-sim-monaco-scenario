from pathlib import Path
import typing as ty

import pandas as pd

import numpy as np

import shutil

import jsonlines

from simulation_extractions.parse_sumo_road_map import (
    KeplerAttributeGenerator,
    GeoAttributeInformation,
    VariableWeightObject
)

import logzero
logger = logzero.logger


def main(path_sumo_net_xml: Path,
         path_sumo_sim_xml: Path,
         mode_generation: str,
         path_output_csv: Path,
         size_time_bucket: int,
         path_variable_weight_jsonl: ty.Optional[Path],
         path_simulation_array: ty.Optional[Path],
         time_step_interval_export: ty.Optional[int],
         observation_every_step_per: int = 10,
         observation_threshold_value: int = 5,
         lane_or_egde: str = 'edge'):
    """
    Args:
        time_step_interval_export (int): User's parameter. Time step interval to export the data.
        observation_every_step_per (int): Observation every step in the simulation setting.
    """
    assert path_sumo_net_xml.exists(), f"Path does not exist: {path_sumo_net_xml}"
    assert path_sumo_sim_xml.exists(), f"Path does not exist: {path_sumo_sim_xml}"
    
    assert mode_generation in ('observation', 'variable'), f"Mode generation not found: {mode_generation}"
    
    attribute_generator = KeplerAttributeGenerator(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml
    )
    
    if mode_generation == 'observation':
        assert path_simulation_array is not None, "Path simulation array is required for observation mode."
        assert time_step_interval_export is not None, "Time step interval export is required for observation mode."
        assert path_simulation_array.exists(), f"Path does not exist: {path_simulation_array}"
        assert size_time_bucket is not None, "Size time bucket is required for observation mode."
    
        # loading array objects and checking
        sim_obj_x = np.load(path_simulation_array)
        assert 'array' in sim_obj_x, f"Key 'array' not found in {path_simulation_array}"
        assert 'edge_id' in sim_obj_x, f"Key 'edge_id' not found in {path_simulation_array}"
        
        # generate attributes data of simulation {X, Y} for kepler.gl.
        logger.debug("Generating attributes data for simulation...")
        observation_attr_x = attribute_generator.generate_attributes_traffic_observation(
            array_traffic_observation=sim_obj_x['array'],
            seq_lane_id=sim_obj_x['edge_id'],
            size_time_bucket=size_time_bucket,
            lane_or_egde=lane_or_egde,
            description_base_text='',
            observation_every_step_per=observation_every_step_per,
            time_step_interval_export=time_step_interval_export,
            threshold_value=observation_threshold_value)
        # to dict
        seq_dict_obs = [_o.to_dict() for _o in observation_attr_x]
        pd.DataFrame(seq_dict_obs).to_csv(path_output_csv, index=False)
        logger.debug(f"Saved to {path_output_csv}")
    elif mode_generation == 'variable':
        assert path_variable_weight_jsonl is not None, "Path variable weight jsonl is required for variable mode."
        assert path_variable_weight_jsonl.exists(), f"Path does not exist: {path_variable_weight_jsonl}"
        
        logger.debug("Generating attributes data for variable weight...")
        seq_variable_weight = []
        with jsonlines.open(path_variable_weight_jsonl) as reader:
            for _o in reader:
                if 'route_id' in _o:
                    __route_id = _o['route_id']
                elif 'lane_id' in _o:
                    __route_id = _o['lane_id']
                    _o['route_id'] = __route_id
                    del _o['lane_id']
                else:
                    raise ValueError(f"Key 'route_id' or 'lane_id' not found in {_o}")
                # end if
                _obj = VariableWeightObject(**_o)
                seq_variable_weight.append(_obj)
            # end for
        # end with
        logger.debug(f"Loaded {len(seq_variable_weight)} variable weights.")
        
        _n_time_bucket = max([w_obj.time_bucket for w_obj in seq_variable_weight])
        
        observation_attr = attribute_generator.generate_attributes_variable_weight(
            seq_variable_weight_model=seq_variable_weight,
            observation_every_step_per=observation_every_step_per,
            lane_or_egde=lane_or_egde,
            size_time_bucket=size_time_bucket)
        # to dict
        seq_dict_obs = [_o.to_dict() for _o in observation_attr]
        pd.DataFrame(seq_dict_obs).to_csv(path_output_csv, index=False)
        logger.debug(f"Saved to {path_output_csv}")
    else:
        raise ValueError(f"Mode generation not found: {mode_generation}")
    # end if
    
    
def _create_l1_distance_observation_ad_hoc(path_array_x: Path,
                                           path_array_y: Path) -> Path:
    """I want to create an array of L1 distance between two observation arrays.
    I create the array and save it to the temporary directory.
    """
    import tempfile
    
    assert path_array_x.exists(), f"Path does not exist: {path_array_x}"
    assert path_array_y.exists(), f"Path does not exist: {path_array_y}"
    
    # loading array objects and checking
    sim_obj_x = np.load(path_array_x)
    assert 'array' in sim_obj_x, f"Key 'array' not found in {path_array_x}"
    assert 'edge_id' in sim_obj_x, f"Key 'edge_id' not found in {path_array_x}"
    
    sim_obj_y = np.load(path_array_y)
    assert 'array' in sim_obj_y, f"Key 'array' not found in {path_array_y}"
    assert 'edge_id' in sim_obj_y, f"Key 'edge_id' not found in {path_array_y}"
    
    # check the shape.
    assert sim_obj_x['array'].shape == sim_obj_y['array'].shape, f"Shape mismatch: {sim_obj_x['array'].shape} != {sim_obj_y['array'].shape}"
    assert sim_obj_x['edge_id'].tolist() == sim_obj_y['edge_id'].tolist(), f"Edge ID mismatch: {sim_obj_x['edge_id']} != {sim_obj_y['edge_id']}"
    
    # L1 abs differecne.
    array_l1_diff = np.abs(sim_obj_x['array'] - sim_obj_y['array'])
    
    # saving the the L1 abs diff array. Use the temp. directory path.
    path_tmp_dir = Path(tempfile.mkdtemp()) / 'l1_array.npz'
    dict_array = dict(array=array_l1_diff, edge_id=sim_obj_x['edge_id'])
    np.savez(path_tmp_dir, **dict_array)
    
    return path_tmp_dir

def _test_process_array_observation():

    path_sumo_net_xml = Path("/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/heavy_blocking_scenario/in/most.net.xml")
    path_sumo_sim_xml = Path("/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/heavy_blocking_scenario/sumo_cfg.cfg")
    
    size_time_bucket = 500
    # -----------------------------------------------------
    # exporting variable weights to csv
    
    # MMD CV AGG
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/output_jsons/edge_observation/interpretable_mmd-cv_selection.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/kepler_csv/variable_detection_mmd_cv_agg.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)
    
    # MMD Alg one
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/output_jsons/edge_observation/interpretable_mmd-algorithm_one.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/kepler_csv/variable_detection_mmd_alg_one.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)
    
    # MMD Alg one
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/output_jsons/edge_observation/wasserstein_independence-.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/kepler_csv/variable_detection_wasserstein_baseline.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)

    # L1 difference
    path_var_detection_l1 = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/simple-aggregation/0/edge_observation_l1_diff.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/kepler_csv/variable_detection_l1_distance.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_l1,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None,
    )
    
    # -----------------------------------------------------
    # exporting observation data to csv
    observation_threshold_value = 5
    
    path_array_x = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/x/edge_observation.npz")
    
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/kepler_csv/observation_x.csv')
    _mode_generation = 'observation'
    
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_array_x,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )


    path_array_y = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/y/edge_observation.npz")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/kepler_csv/observation_y.csv')
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_array_y,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )
    
    
    # L1 distance in the observation mode.
    # I do not have the file yet, so, I create the file here in ad-hoc style.
    path_temp_l1_array = _create_l1_distance_observation_ad_hoc(path_array_x, path_array_y)
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_observation/kepler_csv/observation_l1_distance.csv')
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_temp_l1_array,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )
    path_temp_l1_array.unlink()


def _test_process_array_waiting_time():

    path_sumo_net_xml = Path("/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/heavy_blocking_scenario/in/most.net.xml")
    path_sumo_sim_xml = Path("/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/heavy_blocking_scenario/sumo_cfg.cfg")
    
    size_time_bucket = 500
    # -----------------------------------------------------
    # exporting variable weights to csv
    
    # MMD CV AGG
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/output_jsons/edge_waiting_time/interpretable_mmd-cv_selection.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/kepler_csv/variable_detection_mmd_cv_agg.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)
    
    # MMD Alg one
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/output_jsons/edge_waiting_time/interpretable_mmd-algorithm_one.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/kepler_csv/variable_detection_mmd_alg_one.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)
    
    # MMD Alg one
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/output_jsons/edge_waiting_time/wasserstein_independence-.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/kepler_csv/variable_detection_wasserstein_baseline.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)


    # -----------------------------------------------------
    # exporting observation data to csv
    observation_threshold_value = 5
    
    path_array_x = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/x/edge_waiting_time.npz")
    
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/kepler_csv/observation_x.csv')
    _mode_generation = 'observation'
    
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_array_x,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )


    path_array_y = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/y/edge_waiting_time.npz")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/kepler_csv/observation_y.csv')
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_array_y,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )
    
    
    # L1 distance in the observation mode.
    # I do not have the file yet, so, I create the file here in ad-hoc style.
    path_temp_l1_array = _create_l1_distance_observation_ad_hoc(path_array_x, path_array_y)
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_waiting_time/kepler_csv/observation_l1_distance.csv')
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_temp_l1_array,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )
    path_temp_l1_array.unlink()


def _test_process_array_density():

    path_sumo_net_xml = Path("/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/heavy_blocking_scenario/in/most.net.xml")
    path_sumo_sim_xml = Path("/home/mitsuzaw/codes/dev/sumo-sim-monaco/simulation_extractions/sumo_configs/base/until_afternoon/heavy_blocking_scenario/sumo_cfg.cfg")
    
    size_time_bucket = 500
    # -----------------------------------------------------
    # exporting variable weights to csv
    
    # MMD CV AGG
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/output_jsons/edge_density/interpretable_mmd-cv_selection.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/kepler_csv/variable_detection_mmd_cv_agg.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)
    
    # MMD Alg one
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/output_jsons/edge_density/interpretable_mmd-algorithm_one.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/kepler_csv/variable_detection_mmd_alg_one.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)
    
    # MMD Alg one
    path_var_detection_mmd_cv_agg = Path("/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/output_jsons/edge_density/wasserstein_independence-.jsonl")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/kepler_csv/variable_detection_wasserstein_baseline.csv')
    _mode_generation = 'variable'
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,        
        path_output_csv=_path_output_csv,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=path_var_detection_mmd_cv_agg,
        size_time_bucket=size_time_bucket,
        path_simulation_array=None,
        time_step_interval_export=None)


    # -----------------------------------------------------
    # exporting observation data to csv
    observation_threshold_value = 5
    
    path_array_x = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/x/edge_density.npz")
    
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/kepler_csv/observation_x.csv')
    _mode_generation = 'observation'
    
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_array_x,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )


    path_array_y = Path("/media/DATA/mitsuzaw/sumo-sim-monaco-scenario/until_afternoon/heavy-blocking-scenario/postprocess/0/y/edge_density.npz")
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/kepler_csv/observation_y.csv')
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_array_y,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )
    
    
    # L1 distance in the observation mode.
    # I do not have the file yet, so, I create the file here in ad-hoc style.
    path_temp_l1_array = _create_l1_distance_observation_ad_hoc(path_array_x, path_array_y)
    _path_output_csv = Path('/media/DATA/mitsuzaw/project_papers/project_data_centric/sumo_monaco/0/edge_density/kepler_csv/observation_l1_distance.csv')
    main(
        path_sumo_net_xml=path_sumo_net_xml,
        path_sumo_sim_xml=path_sumo_sim_xml,
        path_simulation_array=path_temp_l1_array,
        path_output_csv=_path_output_csv,
        size_time_bucket=size_time_bucket,
        time_step_interval_export=50,
        mode_generation=_mode_generation,
        path_variable_weight_jsonl=None,
        observation_threshold_value=observation_threshold_value
    )
    path_temp_l1_array.unlink()
    

if __name__ == '__main__':
    # _test_process_array_observation()
    _test_process_array_waiting_time()
    _test_process_array_density()
