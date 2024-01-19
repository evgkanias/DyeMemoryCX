import dyememorycx.dye as dye
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

optimised = False

N_A = dye.AVOGADRO_CONSTANT
h = dye.PLANK_CONSTANT
speed_of_light = dye.SPEED_OF_LIGHT
dye_parameters = {
    "before annealing": {},
    "after annealing": {},
    # "default": {}
}
configs_path = os.path.join("data", "configs")
if not optimised:
    configs_path = os.path.join(configs_path, "init")

with open(os.path.join(configs_path, "dye-s1b.yaml"), 'r') as f:
    dye_parameters["before annealing"]["S1"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(configs_path, "dye-s2b.yaml"), 'r') as f:
    dye_parameters["before annealing"]["S2"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(configs_path, "dye-scb.yaml"), 'r') as f:
    dye_parameters["before annealing"]["SC"] = yaml.safe_load(f)["cx_params"]

with open(os.path.join(configs_path, "dye-s1a.yaml"), 'r') as f:
    dye_parameters["after annealing"]["S1"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(configs_path, "dye-s2a.yaml"), 'r') as f:
    dye_parameters["after annealing"]["S2"] = yaml.safe_load(f)["cx_params"]
with open(os.path.join(configs_path, "dye-sca.yaml"), 'r') as f:
    dye_parameters["after annealing"]["SC"] = yaml.safe_load(f)["cx_params"]

# with open(os.path.join("..", "data", "configs", "ap-048.yaml"), 'r') as f:
#     dye_parameters["default"]["default"] = yaml.safe_load(f)["cx_params"]

if __name__ == "__main__":

    u_start, u_end = 500, 1500
    u = np.zeros(25000)
    u[u_start:u_end] = 1

    plt.figure("Dye parameters", figsize=(7, 2.5))
    for i, (title, group) in enumerate(dye_parameters.items()):
        plt.subplot(1, len(dye_parameters), 1 + i)
        plt.fill_between([0, u_start], [0, 0], [1, 1], color='grey', alpha=0.1, edgecolor=None)
        plt.fill_between([u_end, u.size], [0, 0], [1, 1], color='grey', alpha=0.1, edgecolor=None)
        for j, (name, params) in enumerate(group.items()):
            if "wavelength" in params and "W_max" in params and "volume" in params:
                E = h * speed_of_light / params["wavelength"]
                k_phi = params["W_max"] / (E * params["volume"] * N_A)
            else:
                k_phi = 1.

            transmittance = lambda x: dye.transmittance(
                x, epsilon=params["epsilon"], length=params["length"], c_tot=params["c_tot"])

            c_off = np.zeros(2500)
            for t in range(1, 2500):
                dc_dt = dye.dcdt(u[t-1], transmittance, k=params["k"], phi=params["phi"], k_phi=k_phi)
                c_off[t] = np.clip(c_off[t-1] + dc_dt(0, c_off[t-1]), 0, 1)

            plt.plot(transmittance(c_off), label=name)

        plt.xlim(0, 2499)
        plt.ylim(0, 1)
        plt.title(title)
        plt.xlabel("time (sec)")
        if i == 0:
            plt.ylabel("transmittance")

    plt.legend()
    plt.tight_layout()
    plt.show()
