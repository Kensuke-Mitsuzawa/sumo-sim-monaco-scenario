import typing as ty
from pathlib import Path
from dataclasses import dataclass
import toml
import shutil
import logzero
from distributed import LocalCluster, Client

from dacite import from_dict

from simulation_extractions import (
    run_simulation,
    parse_fcd_xml,
    config_handler
)

import random

logger = logzero.logger


"""An interface script.
You can run SUMO simulations.
You can run simulations iteratively with different seeds.
"""


@dataclass
class ResourceConfig:
    path_root: str

    path_dir_sumo_base_x: str
    path_dir_sumo_base_y: str

    dir_name_sumo_config: str = 'sumo_cfg'
    dir_name_postprocess: str = 'postprocess'
    # dir_name_sumo_output: str = 'sumo_output'
    
    name_sumo_cfg_x: str = 'sumo_cfg.cfg'
    name_sumo_cfg_y: str = 'sumo_cfg.cfg'

    name_fcd_file: str = 'fcd.output.xml'

    file_name_intermediate_jsonl: str = 'extraction.jsonl'
    file_name_array_lane_observation: str = 'load_observation.npz'
    file_name_array_agent_position: str = 'agent_position.npz'    


@dataclass
class SumoRun:
    path_sumo_home: str
    iteration: int
    is_update_seed: bool


@dataclass
class DaskConfig:
    is_use_dask: bool
    n_workers: int
    n_threads_per_worker: int = 1
    host_scheduler: str = ''
    
    
@dataclass
class PostProcessing:
    is_delete_fcd: bool
    is_delete_intermediate_json: bool


@dataclass
class RootConfig:
    resources: ResourceConfig
    sumo_run: SumoRun
    dask_config: DaskConfig
    postprocessing: PostProcessing
    


# --------------------------------


class DaskFuncArgs(ty.NamedTuple):
    path_fcd_output: Path
    path_work_dir: Path


def __parse_fcd(args: DaskFuncArgs):
    # a function called by Dask.
    parse_fcd_xml.parse_fcd_xml(
        path_fcd_output=args.path_fcd_output,
        path_work_dir=args.path_work_dir,
    )
    



def main(toml_config: Path, is_run_test: bool = False):
    __config = toml.load(toml_config)
    config_type = from_dict(RootConfig, __config)
    
    path_root = Path(config_type.resources.path_root)
    path_root.mkdir(exist_ok=True, parents=True)
    
    path_sumo_cfg = path_root / config_type.resources.dir_name_sumo_config
    path_sumo_cfg.mkdir(exist_ok=True, parents=True)
    
    n_iteration = config_type.sumo_run.iteration
    
    assert Path(config_type.resources.path_dir_sumo_base_x).exists()
    assert Path(config_type.resources.path_dir_sumo_base_y).exists()
    
    # path_sumo_output = path_root / config_type.resources.dir_name_sumo_output
    # path_sumo_output.mkdir(exist_ok=True, parents=True)
    
    assert Path(config_type.sumo_run.path_sumo_home).exists()
    
    path_postprocess_out = path_root / config_type.resources.dir_name_postprocess
    path_postprocess_out.mkdir(exist_ok=True, parents=True)
    
    
    seq_sumo_config_to_be_run = []
    seq_sumo_output_dir: ty.List[ty.Tuple[Path, int, str]] = []
    
    # copy the base-SUMO-cfg files to the resource dir
    for __i_iter in range(n_iteration):
        logger.info(f'Generating sumo-config for the {__i_iter} iteration...')
        __path_dir_iter = path_sumo_cfg / str(__i_iter)
        __path_dir_iter.mkdir(exist_ok=True, parents=True)
        
        __path_x = __path_dir_iter / 'x'
        __path_y = __path_dir_iter / 'y'
        
        # __path_x.mkdir(exist_ok=True, parents=True)
        # __path_y.mkdir(exist_ok=True, parents=True)
        
        shutil.copytree(config_type.resources.path_dir_sumo_base_x, __path_x, dirs_exist_ok=True)
        shutil.copytree(config_type.resources.path_dir_sumo_base_y, __path_y, dirs_exist_ok=True)
        
        __path_output_x = __path_x / 'output'
        __path_output_y = __path_y / 'output'
        __path_output_x.mkdir(exist_ok=True, parents=True)
        __path_output_y.mkdir(exist_ok=True, parents=True)
        
        
        logger.debug(f'Sumo output file will be at {__path_output_x}')
        logger.debug(f'Sumo output file will be at {__path_output_y}')
        
        if config_type.sumo_run.is_update_seed:
            __seed = random.randint(10, 50)
            logger.debug(f'Updating seed value with {__seed}')
            
            config_handler.update_seed_value(
                path_sumo_config=__path_x / config_type.resources.name_sumo_cfg_x,
                seed_value=__seed)
            config_handler.update_seed_value(
                path_sumo_config=__path_y / config_type.resources.name_sumo_cfg_y,
                seed_value=__seed)
        # end if
        
        if is_run_test:
            logger.debug('Running in test mode.')
            config_handler.update_end_time_value(
                path_sumo_config=__path_x / config_type.resources.name_sumo_cfg_x)
            config_handler.update_end_time_value(
                path_sumo_config=__path_y / config_type.resources.name_sumo_cfg_y)
        # end if
        
        seq_sumo_config_to_be_run.append(__path_x / config_type.resources.name_sumo_cfg_x)
        seq_sumo_config_to_be_run.append(__path_y / config_type.resources.name_sumo_cfg_y)
        
        seq_sumo_output_dir.append(tuple([__path_output_x, __i_iter, 'x']))
        seq_sumo_output_dir.append(tuple([__path_output_y, __i_iter, 'y']))
    # end for
    
    if config_type.dask_config.is_use_dask:
        if config_type.dask_config.host_scheduler == '':
            __host_worker = None
        else:
            __host_worker = config_type.dask_config.host_scheduler
        # end if
        dask_cluster = LocalCluster(n_workers=config_type.dask_config.n_workers, 
                                    threads_per_worker=config_type.dask_config.n_threads_per_worker, 
                                    host=__host_worker)
        dask_client = dask_cluster.get_client()
        
        logger.info('Dask-cluster is ready!')
    else:
        dask_client = None
    # end if
    
    # running sumo
    run_simulation.run_simulations(
        sumo_configurations=seq_sumo_config_to_be_run,
        dask_client=dask_client,
        path_sumo_home=Path(config_type.sumo_run.path_sumo_home)
    )
    
    # parsing fcd file
    parsing_task_args = []
    
    for __t_fcd_output in seq_sumo_output_dir:
        __path_fcd = __t_fcd_output[0] / config_type.resources.name_fcd_file
        __i_iter: int = __t_fcd_output[1]
        __xy_flag: str = __t_fcd_output[2]
        
        assert __path_fcd.exists(), f'FCD file does not exist at {__path_fcd}'
        
        __path_target = path_postprocess_out / str(__i_iter) / __xy_flag
        __path_target.mkdir(exist_ok=True, parents=True)
        
        parsing_task_args.append(DaskFuncArgs(
            path_fcd_output=__path_fcd,
            path_work_dir=__path_target.as_posix()
        ))
    # end for
    
    logger.debug('parsing fcd files...')
    if config_type.dask_config.is_use_dask:
        assert dask_client is not None
        task_queue = dask_client.map(__parse_fcd, parsing_task_args)
        __seq_results: ty.List[bool] = dask_client.gather(task_queue)
    else:
        [__parse_fcd(__arg) for __arg in parsing_task_args]
    # end if
    
    # deleting fcd file
    logger.debug('deleting fcd files...')
    for __t_fcd_output in seq_sumo_output_dir:
        __path_fcd = __t_fcd_output[0] / config_type.resources.name_fcd_file
        __path_fcd.unlink()


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    __args = ArgumentParser()
    __args.add_argument('--path_toml', required=True)
    __opt = __args.parse_args()
    
    main(__opt.path_toml, is_run_test=False)
