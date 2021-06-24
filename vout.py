from cmath import exp
import math
from math import pi
from operator import attrgetter
from typing import Any, NamedTuple, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from si_prefix import si_format

VIN_PP = 9.76
R_SENSE = 1000

class Datum:
    pass

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
    RawDatum(100, 1000, 272.0),
    RawDatum(1000, 464, 168.0),
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
])

fig, y_mV = plt.subplots()
plt.xlabel('f (Hz)')
y_mV.set_xscale('log')
y_us = y_mV.twinx()
y_mV.set_ylabel('mVpp')
y_us.set_ylabel('us')

def plot(data, style, label):
    f_Hz, vpp_mV, phi_us = np.vectorize(attrgetter('raw'))(data)
    z = np.vectorize(attrgetter('z'))(data)
    y_mV.plot(f_Hz, vpp_mV, style[0], label=f'mVpp ({label})')
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
        f_Hz = np.vectorize(attrgetter('f_Hz'))(data)
        z = np.vectorize(attrgetter('z'))(data)

        def fit_wrapper(f, *param):
            return model.model(f, *param).real

        param, _ = curve_fit(fit_wrapper, f_Hz, z.real,
                             p0=model.analytical_approximation(data),
                             bounds=self.compute_bounds(model))
        return param

class FitRawData(FitStrategy):

    def fit(self, model, data):
        get_raw = np.vectorize(attrgetter('raw'))
        f_Hz, vpp_mV, phi_us = get_raw(data)

        def fit_wrapper(f, *param):
            model_data = np.vectorize(ZDatum)(f, model.model(f, *param))
            _, model_vpp, model_phi = get_raw(model_data)
            return np.concatenate((model_vpp, model_phi))

        param, _ = curve_fit(fit_wrapper, f_Hz,
                             np.concatenate((vpp_mV, phi_us)),
                             p0=model.analytical_approximation(data),
                             bounds=self.compute_bounds(model))
        return param


def run_model(model, strategy, color):
    param = strategy.fit(model, empirical_data)
    model_data = np.array([
        ZDatum(f_Hz, model.model(f_Hz, *param))
        for f_Hz in range(10, 11001, 10)
    ])
    plot(model_data, (color + '-', color + '--'), model.describe(*param))

run_model(WithRp(), FitRawData(), 'g')

y_mV.legend()
y_us.legend()
plt.show()
