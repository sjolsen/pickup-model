from cmath import exp
import math
from math import log, pi
from operator import attrgetter
from typing import Any, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from si_prefix import si_format

# VIN_PP is actually around 10.1 at 100 Hz and decreases slightly as frequency
# increases. The below models treat the frequency generator as an ideal AC
# voltage source.
VIN_PP = 9.76
R_SENSE = 1000

class Datum:

    v_raw = np.vectorize(attrgetter('raw'))
    v_f_Hz = np.vectorize(attrgetter('f_Hz'))
    v_z = np.vectorize(attrgetter('z'))

class RawDatum(Datum):

    def __init__(self, f_Hz, vpp_mV, phi_us):
        super().__init__()
        self._f_Hz = f_Hz
        self._vpp_mV = vpp_mV
        self._phi_us = phi_us

    @property
    def raw(self):
        return (self._f_Hz, self._vpp_mV, self._phi_us)

    @property
    def f_Hz(self):
        return self._f_Hz

    @property
    def z(self):
        vpp = self._vpp_mV / 1000
        phi = self._phi_us * 1e-6 * self._f_Hz * 2*pi
        return R_SENSE * (VIN_PP / vpp) * exp(1j * phi) - R_SENSE

class ZDatum(Datum):

    def __init__(self, f_Hz, z):
        super().__init__()
        self._f_Hz = f_Hz
        self._z = z

    @property
    def raw(self):
        vout = VIN_PP * R_SENSE / (R_SENSE + self._z)
        vpp_mV = abs(vout) * 1000
        phi_us = -np.angle(vout) / (1e-6 * self._f_Hz * 2*pi)
        return (self._f_Hz, vpp_mV, phi_us)

    @property
    def f_Hz(self):
        return self._f_Hz

    @property
    def z(self):
        return self._z

# Not recorded on paper
empirical_data = np.array([
    # RawDatum(10, 992, -1880),
    RawDatum(100, 992, 290.0),
    RawDatum(200, 944, 280.0),
    RawDatum(300, 872, 266.0),
    RawDatum(400, 796, 250.0),
    RawDatum(500, 728, 236.0),
    RawDatum(600, 660, 220.0),
    RawDatum(700, 596, 206.0),
    RawDatum(800, 548, 192.0),
    RawDatum(900, 500, 180.0),
    RawDatum(1000, 464, 168.0),
    RawDatum(1500, 330, 125.0),
    RawDatum(2000, 246, 99.0),
    RawDatum(3000, 152, 69.2),
    RawDatum(4000, 97.6, 52.0),
    RawDatum(5000, 65.6, 40.0),
    RawDatum(6000, 39.6, 30.8),
    RawDatum(6500, 29.2, 25.6),
    RawDatum(7000, 20.2, 19.2),
    RawDatum(7100, 18.6, 17.6),
    RawDatum(7200, 17.2, 15.6),
    RawDatum(7300, 16.0, 13.2),
    RawDatum(7400, 14.8, 12.4),
    RawDatum(7500, 13.6, 9.2),
    RawDatum(7600, 11.8, 6),
    RawDatum(7700, 11.3, 2.8),
    RawDatum(7800, 10.8, 0),
    RawDatum(7900, 10.6, -2.4),
    RawDatum(8000, 10.6, -4.8),
    RawDatum(8100, 10.7, -7.2),
    RawDatum(8200, 11.0, -9.2),
    RawDatum(8300, 11.5, -11.2),
    RawDatum(8400, 13.0, -12.4),
    RawDatum(8500, 13.8, -14),
    RawDatum(9000, 19.8, -18.8),
    RawDatum(9500, 25.6, -20.4),
    RawDatum(10000, 31.4, -21.2),
    RawDatum(10500, 37.6, -20.8),
    RawDatum(11000, 43.6, -20.4),
    # np.logspace(4, 5, 10, endpoint=False) etc.
    RawDatum(12600, 64.0, -18.4),
    RawDatum(15800, 97.6, -15.3),
    RawDatum(20000, 143, -12.2),
    RawDatum(25100, 192, -9.76),
    RawDatum(31600, 250, -7.72),
    RawDatum(39800, 332, -6.12),
    RawDatum(50100, 416, -4.74),
    RawDatum(63100, 536, -3.74),
    RawDatum(79400, 664, -2.96),
    RawDatum(100000, 848, -2.33),
    RawDatum(126000, 1050, -1.82),
    RawDatum(158000, 1310, -1.43),
    RawDatum(200000, 1680, -1.092),
    RawDatum(251000, 2060, -0.844),
    RawDatum(316000, 2540, -0.644),
    RawDatum(398000, 2900, -0.492),
    RawDatum(501000, 3620, -0.356),
    RawDatum(631000, 4320, -0.256),
    RawDatum(794000, 5000, -0.184),
    RawDatum(1000000, 5640, -0.130),
])

fig, y_mV = plt.subplots()
plt.xlabel('f (Hz)')
y_us = y_mV.twinx()
y_mV.set_xscale('log')
y_us.set_xscale('log')
y_mV.set_ylabel('dBV')
y_us.set_ylabel('us')

def plot(data, style, label):
    f_Hz, vpp_mV, phi_us = Datum.v_raw(data)
    # Since the precision of the scope is relative to the magnitude of the
    # measurement, should the fitting be done against dBV values? Similarly, the
    # relationship between time-domain phase delay and phase angle is
    # frequency-dependent -- a difference of 1 us makes a much bigger difference
    # at 100 kHz than at 1 kHz.
    vpp_dBV = 20 * np.log10(vpp_mV / 1000)
    y_mV.plot(f_Hz, vpp_dBV, style[0], label=f'dBVpp ({label})')
    y_us.plot(f_Hz, phi_us, style[1], label=f'us ({label})')

plot(empirical_data, ('b.', 'bx'), 'empirical')

def z_series(*z):
    return sum(z)

def z_parallel(*z):
    return 1/(sum(1/zi for zi in z))

class Param(NamedTuple):
    name: str
    print_precision: int
    unit: str

class Model:

    def describe(self, *param):
        assert len(param) == len(self.PARAMS)
        return ', '.join(
            f'{p.name} = {si_format(v, precision=p.print_precision)}{p.unit}'
            for p, v in zip(self.PARAMS, param))


class WithoutRp(Model):

    PARAMS = [
        Param('Rl', 2, 'Ω'),
        Param('L', 1, 'H'),
        Param('C', 1, 'F'),
    ]

    def model(self, f, Rl, L, C):
        omega = 2*pi * f
        zl = 1j * omega * L
        zc = -1j / (omega * C)
        zp = z_parallel(z_series(Rl, zl), zc)
        return zp

    def analytical_approximation(self, data):
        data = sorted(data, key=attrgetter('f_Hz'))
        # Measure Rl close to DC
        Rl = data[0].z.real
        # Also measure L close to DC
        dX = (data[1].z - data[0].z).imag
        dω = (2*pi * (data[1].f_Hz - data[0].f_Hz))
        L =  dX / dω
        # Derive C from L using the self-resonant frequency
        srf = max(data, key=lambda x: abs(x.z)).f_Hz
        C = (2*pi * srf)**-2 / L
        return (Rl, L, C)


class WithRp(Model):

    PARAMS = [
        Param('Rl', 2, 'Ω'),
        Param('Rp', 1, 'Ω'),
        Param('L', 1, 'H'),
        Param('C', 1, 'F'),
    ]

    def model(self, f, Rl, Rp, L, C):
        omega = 2*pi * f
        zl = 1j * omega * L
        zc = -1j / (omega * C)
        zp = z_parallel(z_series(Rl, zl), zc, Rp)
        return zp

    def analytical_approximation(self, data):
        Rl, L, C = WithoutRp().analytical_approximation(data)
        # TODO: A better approximation of Rp
        return (Rl, 1.4e6, L, C)


class FitStrategy:

    def compute_bounds(self, model):
        return [
            tuple(0 for p in model.PARAMS),
            tuple(+np.inf for p in model.PARAMS),
        ]

class FitAnalytical(FitStrategy):

    def fit(self, model, data):
        return model.analytical_approximation(data)

class FitResistance(FitStrategy):

    def fit(self, model, data):
        f_Hz = Datum.v_f_Hz(data)
        z = Datum.v_z(data)

        def fit_wrapper(f, *param):
            return model.model(f, *param).real

        param, _ = curve_fit(fit_wrapper, f_Hz, z.real,
                             p0=model.analytical_approximation(data),
                             bounds=self.compute_bounds(model))
        return param

class FitRawData(FitStrategy):

    def fit(self, model, data):
        f_Hz, vpp_mV, phi_us = Datum.v_raw(data)

        def fit_wrapper(f, *param):
            model_data = np.vectorize(ZDatum)(f, model.model(f, *param))
            _, model_vpp, model_phi = Datum.v_raw(model_data)
            return np.concatenate((model_vpp, model_phi))

        param, _ = curve_fit(fit_wrapper, f_Hz,
                             np.concatenate((vpp_mV, phi_us)),
                             p0=model.analytical_approximation(data),
                             bounds=self.compute_bounds(model))
        return param


def run_model(data, model, strategy, color):
    param = strategy.fit(model, data)
    start = log(min(Datum.v_f_Hz(data)), 10)
    stop = log(max(Datum.v_f_Hz(data)), 10)
    decades = stop - start
    f_Hz = np.logspace(start, stop, num=math.ceil(200 * decades))
    model_data = np.vectorize(ZDatum)(f_Hz, model.model(f_Hz, *param))
    plot(model_data, (color + '-', color + '--'), model.describe(*param))

run_model(empirical_data, WithRp(), FitRawData(), 'g')

y_mV.legend()
y_us.legend()
plt.show()
