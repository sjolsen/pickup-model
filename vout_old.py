from cmath import exp
import math
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
# from scipy.optimize import curve_fit
from si_prefix import si_format

VIN_PP = 10
R_SENSE = 1000

def empirical_zp(vpp, phi):
    return R_SENSE * (VIN_PP/vpp) * exp(1j*phi) - R_SENSE

def phase_from_us(us, f_Hz):
    return us * 1e-6 * f_Hz * 360

# raw_data = [
#     (10, 1050, 0),
#     (100, 1040, 9),
#     (200, 984, 18),
#     (300, 920, 26),
#     (400, 840, 34),
#     (500, 784, 40),
#     (600, 720, 44),
#     (700, 652, 48),
#     (800, 600, 52),
#     (900, 552, 54),
#     (1000, 512, 56),
#     (1100, 480, 58),
#     (1200, 448, 60),
# ]

# Not recorded on paper
# (f (Hz), Vpp (mV), phase delay (us))
raw_data = [
    (10, 992, -1880),
    (100, 1000, 272.0),
    (1000, 464, 168.0),
    (2000, 246, 99.0),
    (3000, 152, 69.2),
    (4000, 97.6, 52.0),
    (5000, 65.6, 40.0),
    (6000, 39.6, 30.8),
    (6500, 29.2, 25.6),
    (7000, 20.2, 19.2),
    (7100, 18.6, 17.6),
    (7200, 17.2, 15.6),
    (7300, 16.0, 13.2),
    (7400, 14.8, 12.4),
    (7500, 13.6, 9.2),
    (7600, 11.8, 6),
    (7700, 11.3, 2.8),
    (7800, 10.8, 0),
    (7900, 10.6, -2.4),
    (8000, 10.6, -4.8),
    (8100, 10.7, -7.2),
    (8200, 11.0, -9.2),
    (8300, 11.5, -11.2),
    (8400, 13.0, -12.4),
    (8500, 13.8, -14),
    (9000, 19.8, -18.8),
    (9500, 25.6, -20.4),
    (10000, 31.4, -21.2),
    (10500, 37.6, -20.8),
    (11000, 43.6, -20.4),
]

EMPIRICAL_RESISTANCE = 8780
EXPECTED_VOUT_PP_AT_DC = VIN_PP * R_SENSE / (R_SENSE + EMPIRICAL_RESISTANCE)
# SCALE_FACTOR = EXPECTED_VOUT_PP_AT_DC / (raw_data[0][1] / 1000)
SCALE_FACTOR = 1

samples = [(f_Hz, empirical_zp(SCALE_FACTOR * vpp_mV / 1000, 2*pi * phase_from_us(us, f_Hz)/360))
           for f_Hz, vpp_mV, us in raw_data]

EMPIRICAL_RESONANT_FREQUENCY = 7840

def c_from_l(L):
    omega = 2*pi*EMPIRICAL_RESONANT_FREQUENCY
    return (1/omega)**2 / L

def model(f, L):
    omega = 2*pi*f
    zrl = EMPIRICAL_RESISTANCE + 1j * omega * L
    zc = -1j/(omega * c_from_l(L))
    zp = 1/(1/zrl + 1/zc)
    return zp

xdata = np.array([f_Hz for (f_Hz, z) in samples])
ydata_r = np.array([z.real for (f_Hz, z) in samples])
ydata_i = np.array([z.imag for (f_Hz, z) in samples])
plt.plot(xdata, ydata_r, 'b-', label='R')
plt.plot(xdata, ydata_i, 'b--', label='X')

# popt, pcov = curve_fit(model, xdata, ydata)
# plt.plot(xdata, model(xdata, *popt), 'r-', label='fit: L=%5.3f' % tuple(popt))

res = stats.linregress(xdata, ydata_i)
# EMPIRICAL_L = res.slope / (2*pi)
EMPIRICAL_L = 1.75

def format_lc(L):
    fmt_l = f'{si_format(L, precision=1)}H'
    fmt_c = f'{si_format(c_from_l(L), precision=1)}F'
    return f'L = {fmt_l}, C = {fmt_c}'

model_xdata = np.array(range(100, 10000, 50))
def plot_lc(L, color):
    model_ydata = model(model_xdata, L)
    plt.plot(model_xdata, model_ydata.real, color + '-', label=f'R ({format_lc(L)})')
    plt.plot(model_xdata, model_ydata.imag, color + '--', label='X ({format_lc(L)})')

plot_lc(EMPIRICAL_L, 'r')
# plot_lc(math.floor(EMPIRICAL_L), 'g')
# plot_lc(math.ceil(EMPIRICAL_L), 'y')

plt.xlabel('f (Hz)')
plt.ylabel('Z (ohms)')
plt.legend()
plt.show()
