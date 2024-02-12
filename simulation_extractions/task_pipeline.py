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
    parse_road_observation_xml,
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
    
    name_net_xml: str = 'in/most.net.xml'

    name_fcd_file: str = 'fcd.output.xml'
    name_edge_observation: str = 'most.edge-observation.xml'
    name_lane_observation: str = 'most.lane-observation.xml'
    
    name_additional_xml_file: str = 'additional.xml'

    file_name_intermediate_jsonl: str = 'extraction.jsonl'
    file_name_array_lane_observation: str = 'load_observation.npz'
    file_name_array_agent_position: str = 'agent_position.npz'
    
    is_parse_fcd: bool = False


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
    config_obj: RootConfig
    path_edge_observation: Path
    path_lane_observation: Path
    path_work_dir: Path
    path_fcd_output: ty.Optional[Path]


def __parse_file(args: DaskFuncArgs):
    # a function called by Dask.
    
    # parsing edge file
    path_sumo_net = Path(args.config_obj.resources.path_dir_sumo_base_x) / args.config_obj.resources.name_net_xml
    parse_road_observation_xml.parse_edge_observation(
        path_sumo_net_xml=path_sumo_net,
        path_edge_observation=args.path_edge_observation,
        path_work_dir=args.path_work_dir,
    )
    
    # parsing lane file (planned. But not implemented yet.)
    
    if args.path_fcd_output is not None:
        parse_fcd_xml.parse_fcd_xml(
            path_fcd_output=args.path_fcd_output,
            path_work_dir=args.path_work_dir,
        )
    # end if



class SumoPathResourceTuple(ty.NamedTuple):
    path_output: Path
    i_iter: int
    xy_flag: str


def prepare_sumo_config(path_dir_sumo_cfg: Path, 
                        config_type: RootConfig, 
                        n_iteration: int,
                        is_run_test: bool = False
                        ) -> ty.Tuple[ty.List[Path], ty.List[SumoPathResourceTuple]]:
    """Prepare SUMO configuration files.	
    """
    seq_sumo_config_to_be_run = []
    seq_sumo_output_dir: ty.List[SumoPathResourceTuple] = []

    for __i_iter in range(n_iteration):
        logger.info(f'Generating sumo-config for the {__i_iter} iteration...')
        __path_dir_iter = path_dir_sumo_cfg / str(__i_iter)
        __path_dir_iter.mkdir(exist_ok=True, parents=True)
        
        __path_x = __path_dir_iter / 'x'
        __path_y = __path_dir_iter / 'y'
        
        # assert fcd definition if `is_parse_fcd` is True.
        if config_type.resources.is_parse_fcd:
            config_handler.assert_fcd_definition(__path_x / config_type.resources.name_sumo_cfg_x)
            config_handler.assert_fcd_definition(__path_y / config_type.resources.name_sumo_cfg_y)
        # end if
    
        # assertion of the additional.xml file.
        assert config_type.resources.name_additional_xml_file        
        path_additional_xml_x = Path(config_type.resources.path_dir_sumo_base_x) / config_type.resources.name_additional_xml_file
        assert path_additional_xml_x.exists(), f'No additional.xml file found at {path_additional_xml_x}'
        config_handler.assert_additional_xml_file(path_additional_xml_x)
        
        path_additional_xml_y = Path(config_type.resources.path_dir_sumo_base_y) / config_type.resources.name_additional_xml_file
        assert path_additional_xml_y.exists(), f'No additional.xml file found at {path_additional_xml_y}'
        config_handler.assert_additional_xml_file(path_additional_xml_y)
        
        # __path_x.mkdir(exist_ok=True, parents=True)
        # __path_y.mkdir(exist_ok=True, parents=True)
        
        shutil.copytree(config_type.resources.path_dir_sumo_base_x, __path_x, dirs_exist_ok=True)
        shutil.copytree(config_type.resources.path_dir_sumo_base_y, __path_y, dirs_exist_ok=True)
        
        __path_output_x = __path_x / 'out'
        __path_output_y = __path_y / 'out'
        __path_output_x.mkdir(exist_ok=True, parents=True)
        __path_output_y.mkdir(exist_ok=True, parents=True)
        
        
        logger.debug(f'Sumo output file will be at {__path_output_x}')
        logger.debug(f'Sumo output file will be at {__path_output_y}')
        
        # updating sumo seed value.
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
        
        # updating the end time value (only for testing).
        if is_run_test:
            logger.debug('Running in test mode.')
            config_handler.update_end_time_value(
                path_sumo_config=__path_x / config_type.resources.name_sumo_cfg_x)
            config_handler.update_end_time_value(
                path_sumo_config=__path_y / config_type.resources.name_sumo_cfg_y)
        # end if
    
        seq_sumo_config_to_be_run.append(__path_x / config_type.resources.name_sumo_cfg_x)
        seq_sumo_config_to_be_run.append(__path_y / config_type.resources.name_sumo_cfg_y)
        
        seq_sumo_output_dir.append(SumoPathResourceTuple(__path_output_x, __i_iter, 'x'))
        seq_sumo_output_dir.append(SumoPathResourceTuple(__path_output_y, __i_iter, 'y'))
    # end def
    
    return seq_sumo_config_to_be_run, seq_sumo_output_dir



def main(toml_config: Path, is_run_test: bool = False):
    __config = toml.load(toml_config)
    config_type = from_dict(RootConfig, __config)
    
    path_root = Path(config_type.resources.path_root)
    path_root.mkdir(exist_ok=True, parents=True)
    
    path_dir_sumo_cfg = path_root / config_type.resources.dir_name_sumo_config
    path_dir_sumo_cfg.mkdir(exist_ok=True, parents=True)
    
    n_iteration = config_type.sumo_run.iteration
    
    assert Path(config_type.resources.path_dir_sumo_base_x).exists()
    assert Path(config_type.resources.path_dir_sumo_base_y).exists()
    
    # path_sumo_output = path_root / config_type.resources.dir_name_sumo_output
    # path_sumo_output.mkdir(exist_ok=True, parents=True)
    
    assert Path(config_type.sumo_run.path_sumo_home).exists()
    
    path_postprocess_out = path_root / config_type.resources.dir_name_postprocess
    path_postprocess_out.mkdir(exist_ok=True, parents=True)
    
    # updating sumo config files
    seq_sumo_config_to_be_run, seq_sumo_output_dir = prepare_sumo_config(
        path_dir_sumo_cfg=path_dir_sumo_cfg,
        config_type=config_type,
        n_iteration=n_iteration,
        is_run_test=is_run_test)

    
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
        path_sumo_home=Path(config_type.sumo_run.path_sumo_home),
    )
    
    # parsing output file
    parsing_task_args = []
    
    # output files
    for __t_output in seq_sumo_output_dir:
        # egde and lane files are mandatory.
        __path_edge_xml = __t_output[0] / config_type.resources.name_edge_observation
        __path_lane_xml = __t_output[0] / config_type.resources.name_lane_observation
        
        assert __path_edge_xml.exists(), f'Edge observation file does not exist at {__path_edge_xml}'
        assert __path_lane_xml.exists(), f'Edge observation file does not exist at {__path_lane_xml}'
        
        # fcd file is optional.
        if config_type.resources.is_parse_fcd:
            __path_fcd = __t_output[0] / config_type.resources.name_fcd_file
            assert __path_fcd.exists(), f'FCD file does not exist at {__path_fcd}'
        else:
            __path_fcd = None
        # end if
        
        __i_iter: int = __t_output[1]
        __xy_flag: str = __t_output[2]
        
        __path_target = path_postprocess_out / str(__i_iter) / __xy_flag
        __path_target.mkdir(exist_ok=True, parents=True)
        
        # set files-argument to the function argument.
        parsing_task_args.append(DaskFuncArgs(
            config_obj=config_type,
            path_edge_observation=__path_edge_xml,
            path_lane_observation=__path_lane_xml,
            path_work_dir=__path_target,
            path_fcd_output=__path_fcd,
        ))
    # end for
    
    logger.debug('parsing fcd files...')
    if config_type.dask_config.is_use_dask:
        assert dask_client is not None
        task_queue = dask_client.map(__parse_file, parsing_task_args)
        __seq_results: ty.List[bool] = dask_client.gather(task_queue)
    else:
        [__parse_file(__arg) for __arg in parsing_task_args]
    # end if
    
    # deleting fcd file
    for __t_fcd_output in seq_sumo_output_dir:
        __path_fcd = __t_fcd_output[0] / config_type.resources.name_fcd_file
        if __path_fcd.exists():
            logger.debug('deleting fcd files...')
            __path_fcd.unlink()
        # end if
    # end for


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    __args = ArgumentParser()
    __args.add_argument('--path_toml', required=True)
    __opt = __args.parse_args()
    
    main(__opt.path_toml, is_run_test=True)
