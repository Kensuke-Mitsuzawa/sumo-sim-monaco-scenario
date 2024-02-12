import os
import typing as ty
import sys
from functools import partial

import subprocess

from pathlib import Path
from distributed import LocalCluster, Client

from tqdm import tqdm

"""
Running SUMO simulations in multi-processing
"""


def __run_simulation_shell(path_sumo_cfg: Path,
                           path_sumo_home: ty.Optional[Path]):
    """
    Running SUMO simulation in shell.
    """
    if 'SUMO_HOME' not in os.environ:
        assert path_sumo_home is not None, '`path_sumo_home` must be given when no `SUMO_HOME`.'
    # end if
    
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    # end if
    
    sumoCmd = [f'{path_sumo_home}/bin/sumo', '-c', path_sumo_cfg.as_posix()]
    bash_command  = ' '.join(sumoCmd)
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()


def __run_simulation_with_traci(path_sumo_cfg: Path,
                                path_sumo_home: ty.Optional[Path]):
    if 'SUMO_HOME' not in os.environ:
        assert path_sumo_home is not None, '`path_sumo_home` must be given when no `SUMO_HOME`.'
    # end if
    
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    # end if
    
    import traci
    
    pbar = tqdm()

    sumoCmd = [f'{path_sumo_home}/bin/sumo', '-c', path_sumo_cfg]
    traci.start(sumoCmd)

    time_start: float = traci.simulation.getTime()
    time_end: float = traci.simulation.getEndTime()
    time_delta_data_extraction = traci.simulation.getDeltaT()

    timestamp_index = 0
    # while traci.simulation.getMinExpectedNumber() > 0:
    while traci.simulation.getTime() <= time_end:
        # Advance the simulation one step
        traci.simulationStep()
        pbar.update(1)
    # end for

    # Close the TraCI connection
    traci.close()
    
    return True
        


def run_simulations(sumo_configurations: ty.List[Path],
                    path_sumo_home: ty.Optional[Path] = None,
                    dask_client: ty.Optional[Client] = None,
                    is_use_traci: bool = False):
    """Running SUMO simulations in multi-processing.
    
    Parameters
    ----------
    sumo_configurations: list
        a list of SUMO simulation configs.
    path_sumo_home: `Path`.
        where a sumo is installed. `SUMO_HONE`.
        
    Return
    ----------
    None.
    """
    
    if 'SUMO_HOME' not in os.environ:
        assert path_sumo_home is not None, '`path_sumo_home` must be given when no `SUMO_HOME`.'
    # end if
    
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    # end if

    is_traci_available = False
    
    try:
        import traci
        is_traci_available = True
    except ImportError:
        is_traci_available = False
    # end if
    
    if is_traci_available and is_use_traci:
        func_simulation = __run_simulation_with_traci
    else:
        func_simulation = __run_simulation_shell
    # end if
    
    if dask_client is None:
        [func_simulation(__p_config, path_sumo_home) for __p_config in sumo_configurations]
    else:
        func_run_simulation = partial(func_simulation, path_sumo_home=path_sumo_home)
        task_queue = dask_client.map(func_run_simulation, sumo_configurations)
        __seq_results: ty.List[bool] = dask_client.gather(task_queue)
