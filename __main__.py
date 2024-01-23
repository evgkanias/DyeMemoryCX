import dyememorycx.simulation as sim
import dyememorycx.dye_io as dio

import multiprocessing as mp
import numpy as np
import loguru as lg
import datetime
import sys
import glob
import os

lg.logger.remove()
LOGGER = lg.logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <5}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>    "
           "<level>{message}</level>",
    filter=lambda record: "experiment" not in record["extra"]
)


def run_simulation(simulation_name='pi', route='random', ts_outbound=1000, ts_inbound=1500,
                   step_size=0.01, seed=2023, noise=0.1, animation=None,
                   cx_class='StoneCX', cx_params=None, threads=None):

    if not isinstance(seed, list):
        seed = [seed]

    if not isinstance(noise, list):
        noise = [noise]

    if len(seed) == 2 and seed[0] < seed[1]:
        seed_end = seed[1]
        seed[1] = seed[0] + 1
        for i in range(seed[1] + 1, seed_end + 1):
            seed.append(i)

    if len(noise) == 2 and noise[0] < noise[1]:
        noise = np.arange(noise[0], noise[1] + 0.1, 0.1).tolist()

    if threads is None:
        threads = np.minimum(len(seed), 4)

    tasks = []
    for ns in noise:
        ns_str = int(ns * 100)
        for sd in seed:
            simulation_name_i = os.path.join(rf"{simulation_name}-{ns_str:03d}",
                                             f"{datetime.datetime.now().strftime('%Y%m%d-%H%M')}_{sd}")
            tasks.append(
                (simulation_name_i, route, ts_outbound, ts_inbound, step_size, sd, ns, animation, cx_class, cx_params))

    mp.set_start_method('forkserver')  # fixes problem with multiprocessing in mac
    results = mp.Pool(threads).imap_unordered(sim.run_simulation, tasks)

    for result in results:
        lg.logger.info(f"Finished: {result}", flush=True)


if __name__ == '__main__':
    import warnings
    import argparse
    import yaml

    with (warnings.catch_warnings()):
        warnings.simplefilter("ignore")

        parser = argparse.ArgumentParser(
            description="Run a path integration test."
        )

        parser.add_argument("module", choices=['run_simulation', 'show_results', 'fit_curves', 'single_curve'])
        parser.add_argument("-c", "--config", dest='config_file', type=str, required=False, default=None,
                            help="File with the configuration for the experiment.")
        parser.add_argument("-r", "--result", dest='results_file', type=str, required=False, default=None,
                            help='Load the results instead of running them.')
        parser.add_argument("-a", "--summarise-results", dest='results_directory', type=str, required=False, default=None,
                            help='Load the results in the directory to plot summary.')
        parser.add_argument("-s", "--save", dest='save_directory', type=str, required=False, default=None,
                            help='Save the plots in directory.')
        parser.add_argument("-t", "--plot-file-type", dest='plot_type', type=str, required=False, default='PNG',
                            help='The file-types of the plots.')
        parser.add_argument('-o', '--optimise', dest='optimise', type=bool, required=False, default=False,
                            help='Run optimisation of the dye parameters.')
        parser.add_argument('--show', dest='show_plots', type=bool, required=False, default=True,
                            help='Activates or deactivates showing the plots.')
        parser.add_argument('--threads', dest='nb_threads', type=int, required=False, default=None,
                            help='The number of threads to use for running multiple experiments in parallel.')

        p_args = parser.parse_args()

        kwargs = {}
        if p_args.module in ["run_simulation"]:
            if p_args.config_file is None:
                parser.print_help()
                lg.logger.error("Configuration file was not set.")
            else:
                lg.logger.info(f"Reading configuration file: {p_args.config_file}")
                with open(p_args.config_file, 'r') as f:
                    kwargs = yaml.safe_load(f)
                kwargs['threads'] = p_args.nb_threads
                run_simulation(**kwargs)

        if p_args.module in ["run_simulation", "show_results"]:
            if p_args.module in ["show_results"] and p_args.results_file is None and p_args.results_directory is None:
                parser.print_help()
                lg.logger.error("Results file or directory has to be set.")
            if p_args.results_file is not None:
                dat = {}
                if p_args.results_file in ["current", "last"]:
                    list_of_files = glob.glob(os.path.join('data', 'stats', '**', '*.npz'), recursive=True)
                    latest_file = os.path.abspath(max(list_of_files, key=os.path.getctime))
                    lg.logger.info(f"Reading simulation data file: {latest_file}")
                    dat[latest_file] = np.load(latest_file)
                elif os.path.exists(p_args.results_file) and os.path.isdir(p_args.results_file):
                    list_of_files = glob.glob(os.path.join(p_args.results_file, '**', '*.npz'), recursive=True)

                    for f in list_of_files:
                        dat[f] = np.load(f)
                else:
                    lg.logger.info(f"Reading simulation data file: {p_args.results_file}")
                    dat[p_args.results_file] = np.load(p_args.results_file)

                for name_, dat_ in dat.items():
                    dir_name, file_name = os.path.split(name_)
                    _, dir_name = os.path.split(dir_name)
                    file_name = file_name[:-4]
                    sim.plot_results(dat_, name=f"{dir_name}_{file_name}",
                                     save=p_args.save_directory, save_format=p_args.plot_type, show=p_args.show_plots)

            if p_args.results_directory is not None:
                if p_args.results_directory in ["current", "last"]:
                    list_of_dirs = glob.glob(os.path.join('data', 'stats', '*'))
                    latest_dir = os.path.abspath(max(list_of_dirs, key=os.path.getmtime))
                else:
                    latest_dir = os.path.abspath(p_args.results_directory)

                lg.logger.info(f"Reading simulation data from directory: {latest_dir}")
                list_of_files = glob.glob(os.path.join(latest_dir, '*.npz'))

                dat = []
                for file_i in list_of_files:
                    lg.logger.info(f"Reading data from: {file_i}")
                    dat.append(np.load(file_i))

                sim.plot_summarised_results(dat, name=os.path.split(latest_dir)[-1],
                                            save=p_args.save_directory, save_format=p_args.plot_type,
                                            show=p_args.show_plots)

        if p_args.module in ['fit_curves']:
            dio.plot_fitted_curves(optimise=p_args.optimise,
                                   save=p_args.save_directory, save_format=p_args.plot_type, show=p_args.show_plots)

        if p_args.module in ['single_curve']:
            dio.plot_dye_memory_dynamics(
                save=p_args.save_directory, save_format=p_args.plot_type, show=p_args.show_plots)
