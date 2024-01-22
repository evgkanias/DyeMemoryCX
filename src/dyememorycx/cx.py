import dyememorycx._cx_weights as wt
import dyememorycx.dye as dye

import numpy as np
import scipy.special as ss


class OriginalModel(object):

    def __init__(self, gain=None, noise=None, rng=None):
        self.gain = gain if gain is not None else 0.025
        self.noise = noise if noise is not None else 0.0
        self.rng = rng if rng is not None else np.random

        self.r_ER = np.zeros(16)
        self.r_LNO1 = np.zeros(2)
        self.r_LNO2 = np.zeros(2)
        self.r_D7 = np.zeros(8)
        self.r_EPG = np.zeros(16)
        self.s_PFN = np.zeros(16)
        self.r_PFN = np.zeros(16)
        self.r_FC2 = np.zeros(16)
        self.c_FC2 = 0.5 * np.ones(16)
        self.r_hD = np.zeros(16)
        self.r_PFL3 = np.zeros(16)

        self.__l_theta = 0.0

    def reset(self, noise=None, rng=None):
        self.noise = noise if noise is not None else self.noise
        self.rng = rng if rng is not None else self.rng

        self.r_ER[:] = 0.0
        self.r_LNO1[:] = 0.0
        self.r_LNO2[:] = 0.0
        self.r_D7[:] = 0.0
        self.r_EPG[:] = 0.0
        self.s_PFN[:] = 0.0
        self.r_PFN[:] = 0.0
        self.r_FC2[:] = 0.0
        self.c_FC2[:] = 0.5
        self.r_hD[:] = 0.0
        self.r_PFL3[:] = 0.0

        self.__l_theta = 0.0

    def __call__(self, theta, speed, dt=1.0):

        omega = speed * np.ones(2, dtype=float) / np.sqrt(2)

        # self.r_LNO1[:] = np.clip(omega, 0, 1)
        # self.r_LNO2[:] = np.clip(0.5 * (1.0 - omega), 0, 1)
        self.r_LNO1[:] = np.clip(omega + uniform_noise(self.noise, self.rng, self.r_LNO1.shape), 0, 1)
        self.r_LNO2[:] = np.clip(0.5 * (1.0 - omega) + uniform_noise(self.noise, self.rng, self.r_LNO2.shape), 0, 1)
        # self.r_LNO1[:] = np.clip(omega + self.rng.normal(scale=self.noise, size=self.r_LNO1.shape), 0, 1)
        # self.r_LNO2[:] = np.clip(0.5 * (1.0 - omega) + self.rng.normal(scale=self.noise, size=self.r_LNO2.shape), 0, 1)
        self.r_ER[:] = self.a_func(6.8 * np.cos(theta - wt.ER_pref) - 3.0)
        self.r_EPG[:] = self.a_func(3.0 * self.r_ER.dot(wt.ER2EPG) + 0.5)
        self.r_D7[:] = self.a_func(5.0 * (0.667 * self.r_EPG.dot(wt.EPG2D7) + 0.333 * self.r_D7.dot(wt.D72D7)))
        self.s_PFN[:] = ((self.r_LNO2 - 0.5).dot(wt.LNO2PFN) * (self.r_D7 - 1.0).dot(wt.D72PFN) +
                         0.25 * self.r_LNO1.dot(wt.LNO2PFN))
        self.r_PFN[:] = self.a_func(200.0 * self.s_PFN - 2.5)

        self.c_FC2[:] = self.update_memory(dt)
        self.r_FC2[:] = self.a_func(5.0 * self.c_FC2 - 2.5)

        self.r_hD[:] = self.a_func(5.0 * self.r_FC2.dot(wt.FC2hD) - 2.5)
        self.r_PFL3[:] = self.a_func(7.5 * (
                0.5 * self.r_FC2.dot(wt.FC2PFL3) +
                0.5 * self.r_hD.dot(wt.hD2PFL3) +
                1.0 * self.r_D7.dot(wt.D72PFL3)) + 1.0)

        # self.__l_theta = theta
        return self.motor_output()

    def update_memory(self, dt=1.0):
        return self.c_FC2 + self.gain * self.s_PFN * dt

    def motor_output(self):
        motor = self.r_PFL3.reshape((2, -1)).sum(axis=1) + uniform_noise(self.noise, self.rng, 2)

        return 0.25 * (motor[0] - motor[1])

    def a_func(self, x):
        return logistic(x, self.noise, self.rng)


class DyeModel(OriginalModel):
    def __init__(self, dye_model=None, **kwargs):
        """

        Parameters
        ----------
        dye_model: dye.Dye
            the dye model.
        """
        kwargs.setdefault('gain', 1.0)
        if dye_model is None:
            dye_kwargs = {}
            keys = list(kwargs.keys())
            for key in keys:
                if key not in ['gain', 'noise', 'rng']:
                    dye_kwargs[key] = kwargs.pop(key)
            dye_model = dye.Dye(**dye_kwargs)
        OriginalModel.__init__(self, **kwargs)

        self.dye = dye_model

    def reset(self, noise=None, rng=None):
        OriginalModel.reset(self, noise, rng)
        self.dye.reset_like(self.r_FC2)

    def update_memory(self, dt=1.0):
        return self.dye(self.r_PFN * self.gain, dt)


def logistic(x, noise=0., rng=np.random):
    return np.clip(ss.expit(x) + uniform_noise(noise, rng, x.shape), 0., 1.)


def uniform_noise(noise=0.0, rng=np.random, shape=None):
    return rng.uniform(low=-noise, high=noise, size=shape)
