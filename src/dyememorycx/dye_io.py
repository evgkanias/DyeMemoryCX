import dyememorycx.dye as dye

import scipy.optimize as so
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import os


N_A = dye.AVOGADRO_CONSTANT
h = dye.PLANK_CONSTANT
speed_of_light = dye.SPEED_OF_LIGHT


def load_parameters(config_dir):
    dye_parameters = {
        "before annealing": {},
        "after annealing": {}
    }
    data_map = {'S1': 'dcS1', 'S2': 'dcS2', 'SC': '1500rpm'}
    for condition in ['before annealing', 'after annealing']:
        for sample in ['S1', 'S2', 'SC']:
            filename = f"dye-{sample.lower()}{condition[0]}.yaml"
            with open(os.path.join(config_dir, "initial", filename), 'r') as f:
                dye_parameters[condition][sample] = yaml.safe_load(f)["cx_params"]
            with open(os.path.join(config_dir, filename), 'r') as f:
                dye_parameters[condition][sample]["opt"] = yaml.safe_load(f)["cx_params"]
            dye_parameters[condition][sample]['data'] = None
            data_file = os.path.join(config_dir, '..', 'data', 'dyeFilmData_Thomas_20231212',
                                     f"{condition[0]}A{data_map[sample]}_653nm.csv")
            if os.path.exists(data_file):
                dye_parameters[condition][sample]['data'] = pd.read_csv(data_file)

            if sample == 'S2':
                dye_parameters[condition][sample]['time_illuminated'] = 7 * 60
            else:
                dye_parameters[condition][sample]['time_illuminated'] = 15 * 60

    return dye_parameters


def plot_fitted_curves(optimise=False, config_dir=None, show=True, save=None, save_format=None):

    fig = plt.figure('fit-data', figsize=(7.5, 4))
    i = 0
    if config_dir is None:
        config_dir = 'configs'
    dye_parameters = load_parameters(config_dir)

    for condition in dye_parameters:
        for sample in dye_parameters[condition]:
            params = dye_parameters[condition][sample]
            t_func = transmittance_func(**params)
            p_opt = p_init = np.array([params["w_max"] / params["volume"], params["k"], params["c_tot"]])

            y_data = None
            if params['data'] is None:
                print(f'{sample}:{condition} > SKIP!')
                x_data = np.exp(np.linspace(0, np.log(4000), 50))
            else:
                x_data, y_data = generate_data(params['data'], params['time_illuminated'])

                if optimise:
                    p_opt = optimise_transmittance(x_data, y_data, t_func, p_init)
                else:
                    p_opt = np.array([params["opt"]["w_max"] / params["opt"]["volume"],
                                      params["opt"]["k"], params["opt"]["c_tot"]])
                w_v_init, k_init, c_init = p_init
                w_v_opt, k_opt, c_opt = p_opt
                print(f"{sample}:{condition} > "
                      f"k: init: {k_init:.2e}, opt: {k_opt:.2e} | "
                      f"c_tot: init: {c_init:.2e}, opt: {c_opt:.2e} | "
                      f"volume: init: {params['w_max'] / w_v_init:.2e}, opt: {params['w_max'] / w_v_opt:.2e}")

            # x_min, x_max = -15 * 60, np.max(x_data)
            for j, x_min, x_max in zip([0, 1], [-20 * 60, 73000], [8500, 90000]):
                plt.subplot(2, 6, 1 + 2 * i + j)

                if y_data is not None:
                    plt.scatter(x_data[1:], y_data[1:], s=20, marker='.', color='black')
                    plt.scatter(x_data[:1], y_data[:1], s=50, marker='.', color='white', edgecolor='black')

                x_pre = np.linspace(-20 * 60, 0, 20 * 60 + 1)
                x_post = np.linspace(0, x_max, x_max)
                x = np.r_[x_pre, x_post]
                plt.fill_between([-params["time_illuminated"], 0], [0, 0], [1, 1],
                                 color='red', alpha=0.1, edgecolor=None)
                plt.plot(x, t_func(x, * p_init), color=f"C{i % 3}", ls='--')
                plt.plot(x, t_func(x, * p_opt), color=f"C{i % 3}", ls='-')
                plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                plt.xticks([0, 5000, 80000])
                plt.xlim(x_min, x_max)
                plt.ylim(0, 1)
                if i >= 3:
                    plt.xlabel("t (sec)")
                if i % 3 > 0 or j > 0:
                    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [""] * 6)
                else:
                    plt.ylabel("S / blank")
            i += 1

    fig.tight_layout()
    plt.tight_layout()
    if save is not None:
        plt.savefig(os.path.join(save, f"fit_curves.{save_format.lower()}"), dpi=600, format=save_format)
    if show:
        plt.show()


def plot_dye_memory_dynamics(config_dir=None, show=True, save=None, save_format=None):
    fig = plt.figure('fit-data', figsize=(2.5, 2))

    if config_dir is None:
        config_dir = 'configs'
    dye_parameters = load_parameters(config_dir)

    t_func = transmittance_func(**dye_parameters["after annealing"]["S2"])

    p_opt = np.array([dye_parameters["after annealing"]["S2"]["opt"]["w_max"] /
                      dye_parameters["after annealing"]["S2"]["opt"]["volume"],
                      dye_parameters["after annealing"]["S2"]["opt"]["k"],
                      dye_parameters["after annealing"]["S2"]["opt"]["c_tot"]])

    x_min, x_max = -30, 1000  # hours

    x_pre = np.linspace(x_min * 360, 0, abs(x_min * 360) + 1)
    x_post = np.linspace(0, x_max * 360, x_max * 360 + 1)
    x = np.r_[x_pre, x_post]
    plt.fill_between([-dye_parameters["after annealing"]["S2"]["time_illuminated"] / 360, 0], [0, 0], [1, 1],
                     color='red', alpha=0.1, edgecolor=None)
    y_exp = t_func(x, * p_opt)
    y_lin = np.maximum((np.r_[x_pre, -x_post * 0.002]
                        + dye_parameters["after annealing"]["S2"]["time_illuminated"]) /
                       dye_parameters["after annealing"]["S2"]["time_illuminated"] * 0.25, 0)
    y_lin += y_exp.min()
    plt.plot(x / 360, y_lin, color="magenta", ls='-')
    plt.plot(x / 360, y_exp, color="orange", ls='-')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.xticks([0, 600])
    plt.xlim(x_min, x_max)
    plt.ylim(0, 1)
    plt.xlabel("t (hours)")
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [""] * 6)
    plt.ylabel("S / blank")

    print(f"Time illuminated: {dye_parameters['after annealing']['S2']['time_illuminated'] // 60} min")

    fig.tight_layout()
    plt.tight_layout()
    if save is not None:
        plt.savefig(os.path.join(save, f"single_curve.{save_format.lower()}"), dpi=600, format=save_format)
    if show:
        plt.show()


def generate_data(data, time_illuminated):
    x_data = data['t (s)'].to_numpy()
    d_data = data['dark'].to_numpy()
    b_data = data['blank'].to_numpy()
    dark = np.mean(d_data)
    blank = np.mean(b_data)
    s_data = data['S'].to_numpy()
    y_data = s_data / blank

    # add the dark time-step
    x_data = np.r_[-time_illuminated - 1, x_data]
    y_data = np.r_[dark / blank, y_data]

    return x_data, y_data


def optimise_transmittance(x, y, func, p_init=None):
    res = so.curve_fit(func, x, y, p0=p_init, bounds=(0, np.inf))

    return res[0]


def transmittance_func(time_illuminated=0, **params):
    tr_params = {p: params[p] for p in ['epsilon', 'length']}
    dc_params = {p: params[p] for p in ['phi']}

    E = h * speed_of_light / params["wavelength"]

    def f(t, w_max_vol, k, c_tot):

        k_phi = w_max_vol / (E * N_A)

        def transmittance(x):
            return dye.transmittance(x, **tr_params, c_tot=c_tot)

        c_off = np.zeros_like(t)

        t0 = -time_illuminated - 2
        for i, t1 in enumerate(t):
            c_pre = 0 if i == 0 else c_off[i - 1]
            for t_ in range(int(t0) + 1, int(t1) + 1):
                dc_dt = dye.dcdt(float(-time_illuminated <= t_ < 0), transmittance, k_phi=k_phi, k=k, **dc_params)
                c_pre = np.clip(c_pre + dc_dt(0, c_pre), 0, 1)
            c_off[i] = c_pre
            t0 = t1

        return transmittance(c_off)

    return f
