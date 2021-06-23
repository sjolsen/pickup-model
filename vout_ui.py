from cmath import exp
import math
from math import pi

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from scipy.optimize import curve_fit
from si_prefix import si_format

VIN_PP = 9.76
R_SENSE = 1000

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

def raw_to_z(f_Hz, vpp_mV, phi_us):
    vpp = vpp_mV / 1000
    phi = phi_us * 1e-6 * f_Hz * 2*pi
    return R_SENSE * (VIN_PP / vpp) * exp(1j * phi) - R_SENSE

def plot(xdata, ydata, color, label):
    r = plt.plot(xdata, ydata.real, color + '-', label=f'R ({label})')
    x = plt.plot(xdata, ydata.imag, color + '--', label=f'X ({label})')
    return r, x

fig, _ = plt.subplots()

empirical_xdata = np.array([f_Hz for (f_Hz, _, _) in raw_data])
empirical_ydata = np.array([raw_to_z(*data) for data in raw_data])
plot(empirical_xdata, empirical_ydata, 'b', 'empirical')

def model(f, R, L, C):
    omega = 2*pi * f
    zrl = R + 1j * omega * L
    zc = -1j / (omega * C)
    zp = 1 / (1/zrl + 1/zc)
    return zp

def model_r(f, R, L, C):
    return model(f, R, L, C).real

def model_i(f, R, L, C):
    return model(f, R, L, C).imag

estimate = [8780, 2.7, 150e-12]  # Ohm, H, pF
rparam, _ = curve_fit(model_r, empirical_xdata, empirical_ydata.real,
                      p0=estimate)
iparam, _ = curve_fit(model_i, empirical_xdata, empirical_ydata.imag,
                      p0=estimate)

def format_rlc(R, L, C):
    fmt_r = f'{si_format(R, precision=0)}Ohm'
    fmt_l = f'{si_format(L, precision=1)}H'
    fmt_c = f'{si_format(C, precision=1)}F'
    return f'R = {fmt_r}, L = {fmt_l}, C = {fmt_c}'

model_xdata = np.array(range(10, 11001, 10))
((rplt,), (xplt,)) = plot(
    model_xdata, model(model_xdata, *iparam), 'r', format_rlc(*rparam))

plt.xlabel('f (Hz)')
plt.ylabel('Z (Ohm)')
plt.legend()

plt.subplots_adjust(right=0.75)
r_slider = Slider(
    ax=plt.axes([0.75, 0, 0.75 + 0.25/6, 1]),
    label='R (kOhm)',
    valmin=0,
    valmax=50,
    valinit=rparam[0] / 1000,
    orientation='vertical',
)
l_slider = Slider(
    ax=plt.axes([0.75 + 2*0.25/6, 0, 0.75 + 0.25/6, 1]),
    label='L (H)',
    valmin=1,
    valmax=5,
    valinit=rparam[1],
    orientation='vertical',
)
c_slider = Slider(
    ax=plt.axes([0.75 + 4*0.25/6, 0, 0.75 + 0.25/6, 1]),
    label='C (pF)',
    valmin=10,
    valmax=1000,
    valinit=1e12 * rparam[2],
    orientation='vertical',
)

def update(val):
    R = r_slider.val * 1000
    L = l_slider.val
    C = c_slider.val * 1e-12
    ydata = model(model_xdata, R, L, C)
    rplt.set_ydata(ydata.real)
    xplt.set_ydata(ydata.imag)
    rplt.set_label(f'R ({format_rlc(R, L, C)})')
    xplt.set_label(f'X ({format_rlc(R, L, C)})')
    fig.canvas.draw_idle()

r_slider.on_changed(update)
l_slider.on_changed(update)
c_slider.on_changed(update)

plt.show()
