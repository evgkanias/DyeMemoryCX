import dyememorycx.dye as dye

import pandas as pd
import numpy as np
import yaml
import os


optimise = False

N_A = dye.AVOGADRO_CONSTANT
h = dye.PLANK_CONSTANT
speed_of_light = dye.SPEED_OF_LIGHT
dye_parameters = {
    "before annealing": {},
    "after annealing": {}
}
config_dir = os.path.join("configs")
with open(os.path.join(config_dir, "init", "dye-s1b.yaml"), 'r') as f:
    dye_parameters["before annealing"]["S1"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(config_dir, "init", "dye-s2b.yaml"), 'r') as f:
    dye_parameters["before annealing"]["S2"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(config_dir, "init", "dye-scb.yaml"), 'r') as f:
    dye_parameters["before annealing"]["SC"] = yaml.safe_load(f)["cx_params"]

with open(os.path.join(config_dir, "init", "dye-s1a.yaml"), 'r') as f:
    dye_parameters["after annealing"]["S1"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(config_dir, "init", "dye-s2a.yaml"), 'r') as f:
    dye_parameters["after annealing"]["S2"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(config_dir, "init", "dye-sca.yaml"), 'r') as f:
    dye_parameters["after annealing"]["SC"] = yaml.safe_load(f)["cx_params"]

dye_parameters["before annealing"]["S1"]["data"] = None
dye_parameters["before annealing"]["S2"]["data"] = pd.read_csv(
    os.path.join("data", "dyeFilmData_Thomas_20231212", "bAdcS2_653nm.csv"))
dye_parameters["before annealing"]["SC"]["data"] = pd.read_csv(
    os.path.join("data", "dyeFilmData_Thomas_20231212", "bA1500rpm_653nm.csv"))
dye_parameters["after annealing"]["S1"]["data"] = pd.read_csv(
    os.path.join("data", "dyeFilmData_Thomas_20231212", "aAdcS1_653nm.csv"))
dye_parameters["after annealing"]["S2"]["data"] = pd.read_csv(
    os.path.join("data", "dyeFilmData_Thomas_20231212", "aAdcS2_653nm.csv"))
dye_parameters["after annealing"]["SC"]["data"] = pd.read_csv(
    os.path.join("data", "dyeFilmData_Thomas_20231212", "aA1500rpm_653nm.csv"))

if not optimise:
    with open(os.path.join(config_dir, "dye-s1b.yaml"), 'r') as f:
        dye_parameters["before annealing"]["S1"]["opt"] = yaml.safe_load(f)["cx_params"]
    with open(os.path.join(config_dir, "dye-s2b.yaml"), 'r') as f:
        dye_parameters["before annealing"]["S2"]["opt"] = yaml.safe_load(f)["cx_params"]
    with open(os.path.join(config_dir, "dye-scb.yaml"), 'r') as f:
        dye_parameters["before annealing"]["SC"]["opt"] = yaml.safe_load(f)["cx_params"]

    with open(os.path.join(config_dir, "dye-s1a.yaml"), 'r') as f:
        dye_parameters["after annealing"]["S1"]["opt"] = yaml.safe_load(f)["cx_params"]
    with open(os.path.join(config_dir, "dye-s2a.yaml"), 'r') as f:
        dye_parameters["after annealing"]["S2"]["opt"] = yaml.safe_load(f)["cx_params"]
    with open(os.path.join(config_dir, "dye-sca.yaml"), 'r') as f:
        dye_parameters["after annealing"]["SC"]["opt"] = yaml.safe_load(f)["cx_params"]

dye_parameters["before annealing"]["S1"]["time_illuminated"] = 15 * 60
dye_parameters["before annealing"]["S2"]["time_illuminated"] = 7 * 60
dye_parameters["before annealing"]["SC"]["time_illuminated"] = 15 * 60
dye_parameters["after annealing"]["S1"]["time_illuminated"] = 15 * 60
dye_parameters["after annealing"]["S2"]["time_illuminated"] = 7 * 60
dye_parameters["after annealing"]["SC"]["time_illuminated"] = 15 * 60


def transmittance_func(time_illuminated=0, **params):
    tr_params = {p: params[p] for p in ['epsilon', 'length']}
    dc_params = {p: params[p] for p in ['phi']}

    E = h * speed_of_light / params["wavelength"]

    def f(t, w_max_vol, k, c_tot):

        k_phi = w_max_vol / (E * N_A)
        transmittance = lambda x: dye.transmittance(x, **tr_params, c_tot=c_tot)

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


if __name__ == '__main__':
    import scipy.optimize as so
    import matplotlib.pyplot as plt

    plt.figure('fit-data', figsize=(7.5, 4))
    i = 0
    for condition in dye_parameters:
        for sample in dye_parameters[condition]:
            params = dye_parameters[condition][sample]
            t_func = transmittance_func(**params)
            p_opt = p_init = np.array([params["W_max"] / params["volume"], params["k"], params["c_tot"]])

            y_data = None
            if params['data'] is None:
                print(f'{sample}:{condition} > SKIP!')
                x_data = np.exp(np.linspace(0, np.log(4000), 50))
            else:
                x_data = params['data']['t (s)'].to_numpy()
                d_data = params['data']['dark'].to_numpy()
                b_data = params['data']['blank'].to_numpy()
                dark = np.mean(d_data)
                blank = np.mean(b_data)
                s_data = params['data']['S'].to_numpy()
                y_data = s_data / blank
                # y_data = (s_data - d_data) / (b_data - d_data)

                # add the dark time-step
                x_data = np.r_[-params["time_illuminated"] - 1, x_data]
                y_data = np.r_[dark / blank, y_data]

                if optimise:
                    p_opt, pcov = so.curve_fit(t_func, x_data, y_data, p0=p_init, bounds=(0, np.inf))
                else:
                    p_opt = np.array([params["opt"]["W_max"] / params["opt"]["volume"],
                                      params["opt"]["k"], params["opt"]["c_tot"]])
                w_v_init, k_init, c_init = p_init
                w_v_opt, k_opt, c_opt = p_opt
                print(f"{sample}:{condition} > "
                      f"k: init: {k_init:.2e}, opt: {k_opt:.2e} | "
                      f"c_tot: init: {c_init:.2e}, opt: {c_opt:.2e} | "
                      f"volume: init: {params['W_max'] / w_v_init:.2e}, opt: {params['W_max'] / w_v_opt:.2e}")

            # x_min, x_max = -15 * 60, np.max(x_data)
            for j, x_min, x_max in zip([0, 1], [-1000 * 360, 73000], [10000 * 360, 90000]):
            # for j, x_min, x_max in zip([0, 1], [-20 * 60, 73000], [8500, 90000]):
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
                # plt.plot(x_post, t_func(x_post, * p_init), color=f"C{i-1}", ls='--')
                # plt.plot(x_post, t_func(x_post, * p_opt), color=f"C{i-1}", ls='-')
                plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                plt.xticks([0, 5000, 80000])
                plt.xlim(x_min, x_max)
                plt.ylim(0, 1)
                # if j == 0:
                #     plt.title(f"{sample} ({condition})")
                if i >= 3:
                    plt.xlabel("t (sec)")
                if i % 3 > 0 or j > 0:
                    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [""] * 6)
                else:
                    plt.ylabel("S / blank")
            i += 1

    plt.tight_layout()
    plt.show()

