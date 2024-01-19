import dyememorycx.cx as cx

import matplotlib.pyplot as plt
import loguru as lg
import numpy as np
import scipy.signal as sig
import scipy.interpolate as interp
import time

import os

__root_dir__ = os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))
__stat_dir__ = os.path.abspath(os.path.join(__root_dir__, "data", "stats"))

DEFAULT_ACC = 0.15  # a good value because keeps speed under 1
DEFAULT_DRAG = 0.15


def generate_random_route(T=1000, step_size=0.01, mean_acc=DEFAULT_ACC, drag=DEFAULT_DRAG, kappa=100.0,
                          max_acc=DEFAULT_ACC, min_acc=0.0, vary_speed=False, min_homing_distance=4, rng=None):
    """
    Generate a random outbound route using bee_simulator physics.
    The rotations are drawn randomly from a von mises distribution and smoothed
    to ensure the agent makes more natural turns.

    Parameters
    ----------
    T
    mean_acc
    drag
    kappa
    max_acc
    min_acc
    vary_speed
    min_homing_distance

    Returns
    -------

    """

    if rng is None:
        rng = np.random.RandomState()

    # Generate random turns
    mu = 0.0
    vm = rng.vonmises(mu, kappa, T)
    rotation = sig.lfilter([1.0], [1, -0.4], vm)
    rotation[0] = 0.0

    route = np.zeros((T, 3))  # x, y, theta

    # Randomly sample some points within acceptable acceleration and interpolate to create smoothly varying speed.
    if vary_speed:
        if T > 200:
            num_key_speeds = T // 50
        else:
            num_key_speeds = 4

        x = np.linspace(0, 1, num_key_speeds)
        y = np.random.random(num_key_speeds) * (max_acc - min_acc) + min_acc
        f = interp.interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, 1, T, endpoint=True)
        acceleration = f(xnew)
    else:
        acceleration = mean_acc * np.ones(T)

    # Get headings and velocity for each step
    headings = np.zeros(T)
    velocity = np.zeros((T, 2))

    for t in range(1, T):
        headings[t], velocity[t, :] = get_next_state(
            dt=1.0, heading=headings[t-1], velocity=velocity[t-1, :],
            rotation=rotation[t], acceleration=acceleration[t], drag=drag)
        route[t, :2] = route[t-1, :2] + velocity[t, :]

    route[:, 2] = np.angle(np.diff(route[:, 0] + route[:, 1] * 1j, append=True), deg=True)
    route[-1, 2] = route[-2, 2]

    xy = route[:, 0] + route[:, 1] * 1j
    d = np.diff(xy, prepend=0.)
    d *= step_size / np.maximum(abs(d), np.finfo(float).eps)
    xy = np.cumsum(d)

    route[:, 0] = np.real(xy - xy.mean()) + 5
    route[:, 1] = np.imag(xy - xy.mean()) + 5

    if abs(xy[0] - xy[-1]) < min_homing_distance:
        return generate_random_route(T, step_size, mean_acc, drag, kappa, max_acc, min_acc, vary_speed,
                                     min_homing_distance, rng)

    return route


def get_next_state(dt, heading, velocity, rotation, acceleration, drag=0.5):
    """
    Get new heading and velocity, based on relative rotation and acceleration and linear drag.
    Parameters
    ----------
    dt
    heading
    velocity
    rotation
    acceleration
    drag

    Returns
    -------

    """
    theta = rotate(dt, heading, rotation)
    v = velocity + thrust(dt, theta, acceleration)
    v *= (1.0 - drag) ** dt

    return theta, v


def rotate(dt, theta, r):
    """Return new heading after a rotation around Z axis."""
    return (theta + r * dt + np.pi) % (2.0 * np.pi) - np.pi


def thrust(dt, theta, acceleration):
    """
    Thrust vector from current heading and acceleration

    Parameters
    ----------
    dt: float
        delta time
    theta: float
        clockwise radians around z-axis, where 0 is forward
    acceleration: float
        float where max speed is ....?!?

    Returns
    -------

    """
    return np.array([np.sin(theta), np.cos(theta)]) * acceleration * dt


def plot_results(data, name='results', show=True, save=None, save_format=None):
    if "original" in name.lower():
        line_colour = "#ff00ffff"
    else:
        line_colour = "orange"

    x_in, y_in = data['xy_return'].T
    yaw_in = data['theta_return'].T
    x_out, y_out = data['xy'].T
    yaw_out = data['theta'].T
    l_in, l_out = data['L_return'], data['L']
    c_in, c_out = data['C_return'], data['C']

    # tortuosity
    tau_out = l_out / l_in[0]
    tau_in = l_in / l_in[0]
    l_in_ideal = np.maximum(l_in[0] - c_in + c_in[0], 0)
    tau_in_ideal = l_in_ideal / l_in[0]

    epg = data['EPG']
    pfn = data['PFN']
    fc2 = data['FC2']
    pfl3 = data['PFL3']

    mosaic = """
    AB
    AC
    AD
    AE
    AF
    """
    fig = plt.figure(num=name, figsize=(10, 5))
    ax = fig.subplot_mosaic(mosaic,
                            per_subplot_kw={
                                'A': {'aspect': 'equal'}
                            })
    ax['A'].plot(x_out, y_out, color='grey', label='outbound')
    ax['A'].plot(x_in, y_in, color=line_colour, label='inbound')
    ax['A'].plot([x_out[0]], [y_out[0]], color='grey', marker=(3, 1, np.rad2deg(yaw_out[0])), ls='', ms=10)
    ax['A'].plot([x_in[0]], [y_in[0]], color=line_colour, marker=(3, 1, np.rad2deg(yaw_in[0])), ls='', ms=10)

    ax['B'].imshow(epg.T, vmin=0, vmax=1, aspect='auto')
    ax['B'].set_ylabel('EPG')
    ax['B'].set_yticks([0, 15])
    ax['B'].set_xticks([])
    ax['B'].set_ylim([epg.shape[1] - 0.5, -0.5])
    ax['B'].set_xlim([-0.5, epg.shape[0] - 0.5])
    plot_twin_angles(ax['B'], epg, len(l_out), angle_max=4 * np.pi, line_colour=line_colour)

    ax['C'].imshow(pfn.T, vmin=0, vmax=1, aspect='auto')
    ax['C'].set_ylabel('PFN')
    ax['C'].set_yticks([0, 15])
    ax['C'].set_xticks([])
    ax['C'].set_ylim([pfn.shape[1] - 0.5, -0.5])
    ax['C'].set_xlim([-0.5, pfn.shape[0] - 0.5])
    plot_twin_angles(ax['C'], pfn, len(l_out), angle_max=4 * np.pi, line_colour=line_colour)

    ax['D'].imshow(fc2.T, vmin=0, vmax=1, aspect='auto')
    ax['D'].set_ylabel('FC2')
    ax['D'].set_yticks([0, 15])
    ax['D'].set_xticks([])
    ax['D'].set_ylim([fc2.shape[1] - 0.5, -0.5])
    ax['D'].set_xlim([-0.5, fc2.shape[0] - 0.5])
    plot_twin_angles(ax['D'], fc2, len(l_out), angle_max=4 * np.pi, line_colour=line_colour)

    ax['E'].imshow(pfl3.T, vmin=0, vmax=1, aspect='auto')
    ax['E'].set_ylabel('PFL3')
    ax['E'].set_yticks([0, 15])
    ax['E'].set_xticks([])
    ax['E'].set_ylim([pfl3.shape[1] - 0.5, -0.5])
    ax['E'].set_xlim([-0.5, pfl3.shape[0] - 0.5])
    plot_twin_angles(ax['E'], pfl3, len(l_out), angle_max=4 * np.pi, line_colour=line_colour)

    ax['F'].plot(len(tau_out) + np.arange(len(tau_in)), tau_in_ideal * 100, color='red', ls='--')
    ax['F'].plot(np.arange(len(tau_out)), tau_out * 100, color='grey')
    ax['F'].plot(len(tau_out) + np.arange(len(tau_in)), tau_in * 100, color=line_colour)
    ax['F'].set_xlim([0, len(tau_out) + len(tau_in) - 1])
    ax['F'].set_ylim([0, 100])
    ax['F'].set_ylabel('distance')

    fig.tight_layout()
    plt.tight_layout()
    if save is not None:
        plt.savefig(os.path.join(save, f"{name}.{save_format.lower()}"), dpi=600, format=save_format)
    if show:
        plt.show()


def plot_summarised_results(data, name='summarised_results', show=True, save=None, save_format=None):
    if "stone" in name:
        line_colour = "#ff00ffff"
    else:
        line_colour = "orange"

    x_perc = np.linspace(-150, 200, 1051)
    dataset = {"xy": [], "yaw": [], "perc": [], "turn": [], "tau": [], "tau_opt": [], "memory": []}
    for datum in data:
        if datum['xy_return'].shape[0] < 1:
            continue
        x_in, y_in = datum['xy_return'].T
        yaw_in = datum['theta_return'].T
        x_out, y_out = datum['xy'].T
        yaw_out = datum['theta'].T
        xy = np.r_[x_out, x_in] + 1j * np.r_[y_out, y_in]
        yaw = np.r_[yaw_out, yaw_in]
        l_in, c_in = datum['L_return'], datum['C_return']
        l = np.r_[datum['L'], l_in]

        turn_point = len(x_out)

        home_distance = np.argmin(abs(l_in[0] - c_in + c_in[0]))
        t = np.arange(turn_point + len(x_in))
        t_per = (t - turn_point) / home_distance * 100
        dataset["perc"].append(x_perc)
        dataset["turn"].append(np.argmin(abs(x_perc)))

        dataset["xy"].append(np.interp(x_perc, t_per, xy))
        dataset["yaw"].append(np.interp(x_perc, t_per, yaw))

        tau = l / l_in[0]
        tau_opt = np.r_[np.zeros(turn_point), np.maximum(l_in[0] - c_in + c_in[0], 0) / l_in[0]]
        dataset["tau"].append(np.interp(x_perc, t_per, tau))
        dataset["tau_opt"].append(np.interp(x_perc, t_per, tau_opt))

        r_memory = datum['FC2_mem']
        pref_angles = np.linspace(0, 4 * np.pi, r_memory.shape[1], endpoint=False) + np.pi
        memory = np.sum(r_memory * np.exp(1j * pref_angles[None, :]), axis=1)
        dataset["memory"].append(np.interp(x_perc, t_per, memory))

    plt.figure(name, figsize=(2.5, 5))

    turn = int(np.mean(dataset["turn"]))

    # distance from home
    plt.subplot(311)
    x_mean = np.mean(dataset["perc"], axis=0)
    x_min, x_max = x_mean.min(), x_mean.max()

    tau_25 = np.quantile(dataset["tau"], 0.25, axis=0)
    tau_50 = np.quantile(dataset["tau"], 0.50, axis=0)
    tau_75 = np.quantile(dataset["tau"], 0.75, axis=0)

    tau_opt_50 = np.median(dataset["tau_opt"], axis=0)

    plt.plot([x_mean[turn]] * 2, [0, 100], 'k:')

    plt.fill_between(x_mean[:turn], tau_25[:turn] * 100, tau_75[:turn] * 100,
                     color='grey', edgecolor=None, alpha=0.2)
    plt.fill_between(x_mean[turn-1:], tau_25[turn-1:] * 100, tau_75[turn-1:] * 100,
                     color=line_colour, edgecolor=None, alpha=0.2)

    plt.plot(x_mean[turn:], tau_opt_50[turn:] * 100, color='red')
    plt.plot(x_mean[:turn], tau_50[:turn] * 100, color='grey')
    plt.plot(x_mean[turn-1:], tau_50[turn-1:] * 100, color=line_colour)

    plt.xticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500], [""] * 11)
    plt.ylim(0, 100)
    plt.xlim(x_min, x_max)
    plt.ylabel('distance from home [%]', fontsize=8)
    # plt.xlabel('distance travelled\nrelative to turning point [%]', fontsize=8)

    # heading error
    plt.subplot(312)
    locs = np.array(dataset['xy'])
    yaws = np.rad2deg(np.array(dataset['yaw']))
    home_headings = np.angle(locs[:, :1] - locs, deg=True)

    heading_error = abs((home_headings - yaws + 180) % 360 - 180)
    heading_error_25 = np.quantile(heading_error, 0.25, axis=0)
    heading_error_50 = np.quantile(heading_error, 0.50, axis=0)
    heading_error_75 = np.quantile(heading_error, 0.75, axis=0)

    plt.plot([x_mean[turn]] * 2, [0, 180], 'k:')

    plt.fill_between(x_mean[:turn], heading_error_25[:turn], heading_error_75[:turn],
                     color='grey', edgecolor=None, alpha=0.2)
    plt.fill_between(x_mean[turn-1:], heading_error_25[turn-1:], heading_error_75[turn-1:],
                     color=line_colour, edgecolor=None, alpha=0.2)

    plt.plot(x_mean[:turn], heading_error_50[:turn], color='grey')
    plt.plot(x_mean[turn-1:], heading_error_50[turn-1:], color=line_colour)

    plt.xticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500], [""] * 11)
    plt.yticks([0, 45, 90, 135, 180])
    plt.ylim(0, 180)
    plt.xlim(x_min, x_max)
    plt.ylabel(r'heading error [$^o$]', fontsize=8)
    # plt.xlabel('distance travelled relative to turning point [%]', fontsize=8)

    # memory error
    plt.subplot(313)
    v_memory = np.array(dataset['memory'])
    a_memory = np.angle(v_memory, deg=True)

    memory_error = abs((home_headings - a_memory + 180) % 360 - 180)
    memory_error_25 = np.quantile(memory_error, 0.25, axis=0)
    memory_error_50 = np.quantile(memory_error, 0.50, axis=0)
    memory_error_75 = np.quantile(memory_error, 0.75, axis=0)

    plt.plot([x_mean[turn]] * 2, [0, 90], 'k:')

    plt.fill_between(x_mean[:turn], memory_error_25[:turn], memory_error_75[:turn],
                     color='grey', edgecolor=None, alpha=0.2)
    plt.fill_between(x_mean[turn-1:], memory_error_25[turn-1:], memory_error_75[turn-1:],
                     color=line_colour, edgecolor=None, alpha=0.2)

    plt.plot(x_mean[:turn], memory_error_50[:turn], color='grey')
    plt.plot(x_mean[turn-1:], memory_error_50[turn-1:], color=line_colour)

    plt.xticks([-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500],
               [-500, "", -300, "", -100, "", 100, "", 300, "", 500])
    plt.yticks([0, 30, 60, 90])
    plt.ylim(0, 90)
    plt.xlim(x_min, x_max)
    plt.ylabel(r'memory error [$^o$]', fontsize=8)
    plt.xlabel('distance travelled\nrelative to turning point [%]', fontsize=8)

    plt.tight_layout()
    if save is not None:
        plt.savefig(os.path.join(save, f"{name}.{save_format.lower()}"), dpi=600, format=save_format)
    if show:
        plt.show()


def plot_twin_angles(axis, responses, nb_out, angle_min=0, angle_max=2*np.pi, line_colour='orange'):

    bins = responses.shape[1]
    nb_in = responses.shape[0] - nb_out
    ax_twin = axis.twinx()
    pref_angles = np.linspace(angle_min, angle_max, bins, endpoint=False)
    ang = np.angle(np.mean(responses * np.exp(1j * pref_angles)[None, :], axis=1), deg=True)
    ang = (ang + 0.5 / bins * 360) % 360 - 0.5 / bins * 360
    ts_out = np.arange(nb_out)
    ang_out = ang[:nb_out]
    ts_in = nb_out + np.arange(nb_in)
    ang_in = ang[nb_out:]
    for ts_i, ang_i, c in zip([ts_out, ts_in], [ang_out, ang_in], ["grey", line_colour]):

        split_i = np.where(abs(np.diff(ang_i)) > 90)[0] + 1
        ang_s = np.split(ang_i, split_i)
        ts_s = np.split(ts_i, split_i)
        for ts, ang in zip(ts_s, ang_s):
            ax_twin.plot(ts, ang, ls='-', color=c)
            ax_twin.plot(ts, ang + 360, ls='-', color=c)

    ax_twin.set_yticks([0, 180, 360, 15 / 16 * 720])
    ax_twin.set_ylim([15.5 / 16 * 720, -0.5 / 16 * 720])

    return ax_twin


def run_simulation(task):
    simulation_name, route, ts_outbound, ts_inbound, step_size, seed, noise, animation, cx_class, cx_params = task

    lg.logger.info(f"Seed: {seed}")
    rng = np.random.RandomState(seed)

    sim = Simulation(cx_class=eval(f'cx.{cx_class}'), cx_params=cx_params, name=simulation_name, noise=noise, rng=rng)
    sim.speed = step_size

    rt = generate_random_route(step_size=step_size, T=ts_outbound, rng=rng)

    lg.logger.info(f"Running simulation: {simulation_name}")
    sim(rt, duration=ts_inbound+ts_outbound, save=True)

    return simulation_name


class Simulation(object):
    def __init__(self, cx_class=cx.OriginalModel, cx_params=None, name=None, noise=None, rng=None):
        cx_params = {} if cx_params is None else cx_params
        self.noise = 0.0 if noise is None else noise
        self.rng = np.random if rng is None else rng
        cx_params.setdefault('noise', self.noise)
        cx_params.setdefault('rng', self.rng)

        self.cx = cx_class(**cx_params)

        self.xy = np.zeros(2, dtype=float)
        self.theta = 0.0
        self.speed = 0.10  # 10 cm
        self.delta_time = 1.0  # sec

        self.name = name

        self.stats = {}
        self._saved = None

    def reset(self):
        self.cx.reset(self.noise, self.rng)

        self.xy = np.zeros(2, dtype=float)
        self.theta = 0.0

        self.stats["xy"] = []
        self.stats["xy_return"] = []
        self.stats["theta"] = []
        self.stats["theta_return"] = []
        self.stats["L"] = []
        self.stats["L_return"] = []
        self.stats["C"] = []
        self.stats["C_return"] = []
        self.stats["EPG"] = []
        self.stats["PFN"] = []
        self.stats["FC2_mem"] = []
        self.stats["FC2"] = []
        self.stats["PFL3"] = []

    def __call__(self, route, duration=2500, save=False):
        try:
            self.reset()

            str_len = len(f"{duration}")
            steering = 0.0
            for i in range(duration):

                t0 = time.time()
                steering = self.step(i, route, steering)
                et = time.time() - t0

                lg.logger.info(f"Step {i+1:{str_len}d}/{duration} - "
                               f"x: {self.xy[0]:.2f}, y: {self.xy[1]:.2f}, theta: {np.rad2deg(self.theta):.2f} - "
                               f"time: {et:.2f} sec")
        except KeyboardInterrupt:
            lg.logger.error("Simulation interrupted by keyboard!")
        finally:
            if save:
                self.save()

    def step(self, i, route, steering):

        if i < route.shape[0]:
            self.theta = np.deg2rad(route[i, 2])
        else:
            self.theta += steering * self.delta_time
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        velocity = self.speed * np.exp(1j * self.theta)
        self.xy += np.array([velocity.real, velocity.imag]) * self.delta_time

        if i < route.shape[0]:
            self.stats["xy"].append(self.xy.copy())
            self.stats["theta"].append(self.theta.copy())
            self.stats["L"].append(np.linalg.norm(self.xy))
            self.stats["C"].append(self.speed * (i + 1))
        else:
            self.stats["xy_return"].append(self.xy.copy())
            self.stats["theta_return"].append(self.theta.copy())
            self.stats["L_return"].append(np.linalg.norm(self.xy))
            self.stats["C_return"].append(self.speed * (i + 1))

        self.stats["EPG"].append(self.cx.r_EPG.copy())
        self.stats["PFN"].append(self.cx.r_PFN.copy())
        self.stats["FC2_mem"].append(self.cx.c_FC2.copy())
        self.stats["FC2"].append(self.cx.r_FC2.copy())
        self.stats["PFL3"].append(self.cx.r_PFL3.copy())

        steering = self.cx(self.theta, self.speed, self.delta_time)

        return steering

    def save(self, filename=None):
        filename = filename.replace('.npz', '') if filename is not None else self.name
        save_path = os.path.join(__stat_dir__, f"{filename}.npz")
        dir_path = os.path.dirname(save_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        np.savez_compressed(save_path, **self.stats)
        lg.logger.info("Saved stats in: '%s'" % save_path)
        self._saved = save_path
