import numpy as np

AVOGADRO_CONSTANT = 6.02214076e+23  # /mol
PLANK_CONSTANT = 6.62597915e-34  # J s
SPEED_OF_LIGHT = 299792458.0  # m/s


class Dye(object):
    def __init__(self, epsilon=None, length=None, phi=None, wavelength=None, w_max=None,
                 k=None, c_tot=None, volume=None):
        """

        Parameters
        ----------
        epsilon: float
            the molar absorption coefficient. Default is 1.58e05.
        length: float
            the optical path length through the sample. Default is 2.5e-04.
        phi: float
            the proportion of the absorbed light that leads to switching (quantum yield). Default is 0.2%.
        wavelength: float
            light wavelength. Default is 653 nm.
        w_max: float
            maximum optical effect. Default is 1.3e-07.
        k: np.ndarray[float], float, None
            the rate coefficient (related to half-life as k = log(2) / T_half). Default is 6.28e-05
        c_tot: float
            the total concentration of dye molecules per unit. Default is 6.58e-03.
        volume: float
            the volume of the material. Default is 9.60e-11.
        """
        self.epsilon = epsilon if epsilon else 1.58e05
        self.length = length if length else 2.5e-04
        self.phi = phi if phi else 0.002
        self.wavelength = wavelength if wavelength else 653.0e-09
        self.w_max = w_max if w_max else 1.30e-07
        self.k = k if k else 6.28e-05
        self.c_tot = c_tot if c_tot else 6.58e-06
        self.volume = volume if volume else 9.60e-11

        E = PLANK_CONSTANT * SPEED_OF_LIGHT / self.wavelength
        self.k_phi = self.w_max / (E * self.volume * AVOGADRO_CONSTANT)

        self.last_c = np.zeros(16, dtype=float)

    def __call__(self, light_intensity, dt=1.0):
        self.last_c = np.clip(self.last_c + self.dcdt(light_intensity)(0, self.last_c) * dt, 0, 1)
        return self.transmittance(self.last_c)

    def reset(self, c0=None):
        self.reset_like(self.last_c, c0)

    def reset_like(self, example, c0=None):
        if c0 is None:
            self.last_c[:] = np.zeros_like(example)
        else:
            self.last_c[:] = np.ones_like(example) * c0

    def transmittance(self, c):
        """
        The transmittance corresponds to the weight of the synapse

        Parameters
        ----------
        c: np.ndarray[float]
            the OFF-state concentration (c_OFF)

        Returns
        -------
        np.ndarray[float]
            the transmittance
        """

        return transmittance(c, self.epsilon, self.length, self.c_tot)

    def dcdt(self, u):
        """

        Parameters
        ----------
        u: np.ndarray[float]
            the PFN output, i.e., its normalised activity

        Returns
        -------
        Callable
            the dc/dt function.
        """

        return dcdt(u, self.transmittance, k=self.k, phi=self.phi, k_phi=self.k_phi)


def dcdt(u, transmittance_func, k=0.0, phi=0.00045, k_phi=1.0):
    """

    Parameters
    ----------
    u: np.ndarray[float], float
        the PFN output, i.e., its normalised activity
    transmittance_func
    k: np.ndarray[float], float
    phi: np.ndarray[float], float
    k_phi: np.ndarray[float], float

    Returns
    -------
    Callable
        the dc/dt function.
    """

    def f(t, c):
        """

        Parameters
        ----------
        t: float
            time
        c: np.ndarray[float]
            the OFF-state concentration (c_OFF)

        Returns
        -------
        np.ndarray[float]
            the concentration change (dc/dt)
        """
        T = transmittance_func(c)
        # -k * c: the first-order back-reaction

        return -k * c + u * (1 - T) * phi * k_phi

    return f


def transmittance(c, epsilon, length, c_tot):
    """
    The transmittance corresponds to the weight of the synapse

    Parameters
    ----------
    c: np.ndarray[float]
        the OFF-state concentration (c_OFF)
    epsilon: np.ndarray[float], float
    length: np.ndarray[float], float
    c_tot: np.ndarray[float], float

    Returns
    -------
    np.ndarray[float]
        the transmittance
    """

    return 10 ** -absorbance(c, epsilon, length, c_tot)


def absorbance(c, epsilon, length, c_tot):
    return epsilon * length * (c_tot - c)
