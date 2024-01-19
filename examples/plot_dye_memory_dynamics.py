import dyememorycx.dye as dye

import pandas as pd
import numpy as np
import yaml
import os


N_A = dye.AVOGADRO_CONSTANT
h = dye.PLANK_CONSTANT
speed_of_light = dye.SPEED_OF_LIGHT
config_dir = os.path.join("data", "configs")

with open(os.path.join(config_dir, "dye-s2a.yaml"), 'r') as f:
    dye_params = yaml.safe_load(f)["cx_params"]

with open(os.path.join(config_dir, "optimised", "dye-s2a.yaml"), 'r') as f:
    dye_params["opt"] = yaml.safe_load(f)["cx_params"]

dye_params["time_illuminated"] = 7 * 60


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
    import matplotlib.pyplot as plt

    plt.figure('fit-data', figsize=(2.5, 2))
    i = 0
    t_func = transmittance_func(**dye_params)

    p_opt = np.array([dye_params["opt"]["W_max"] / dye_params["opt"]["volume"],
                      dye_params["opt"]["k"], dye_params["opt"]["c_tot"]])
    w_v_opt, k_opt, c_opt = p_opt

    x_min, x_max = -30, 1000  # hours

    x_pre = np.linspace(x_min * 360, 0, abs(x_min * 360) + 1)
    x_post = np.linspace(0, x_max * 360, x_max * 360 + 1)
    x = np.r_[x_pre, x_post]
    plt.fill_between([-dye_params["time_illuminated"] / 360, 0], [0, 0], [1, 1],
                     color='red', alpha=0.1, edgecolor=None)
    y_exp = t_func(x, * p_opt)
    y_lin = np.maximum((np.r_[x_pre, -x_post * 0.002]
                        + dye_params["time_illuminated"]) / dye_params["time_illuminated"] * 0.25, 0)
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

    print(f"Time illuminated: {dye_params['time_illuminated'] // 60} min")

    plt.tight_layout()
    plt.show()

