import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def sqamp_1(t, s):  # pi+ pi+ -> pi+ pi+
    u = 4*m_pi**2 - s - t
    return np.abs(gv**2 / (4 * F**4) * ( t**2 * (s-u) / (t - m_rho**2) + u**2 * (s-t) / (u - m_rho**2) ))**2


def sqamp_2(t, s):  # pi+ pi- -> pi+ pi-
    u = 4*m_pi**2 - s - t
    return np.abs(gv**2 / (4 * F**4) * ( s**2 * (t-u) / (s - m_rho**2 + 1j*epsilon) + t**2 * (s-u) / (t - m_rho**2) ))**2


def csec(s, sqamp):
    p_inout = 1 / (2*np.sqrt(s)) * np.sqrt(kallen(s, m_pi**2, m_pi**2))
    t1 = - 4 * p_inout**2  # First term will be zero
    t0 = 0
    result = integrate.quad(sqamp, t1, t0, args=(s))
    sigma = 1 / (64 * np.pi * p_inout**2 * s) * result[0]
    return sigma


def kallen(a,b,c):
    return a**2 + b**2 + c**2 - 2*(a*b + b*c + a*c)


gv = 0.17
m_pi = 140 # MeV
m_rho = 775 # MeV
F = 92 # MeV
epsilon = 50000  # Some "small" number (set to get reasonable width of peak)

csecs_1 = []
csecs_2 = []
s_values = np.linspace(4*m_pi**2 + 1, 1.5e6, 1000)
for s in s_values:
    csecs_1.append(csec(s, sqamp_1) * 1e-6 * 0.389)  # in mbarn
    csecs_2.append(csec(s, sqamp_2) * 1e-6 * 0.389)

fig1, ax1 = plt.subplots()
ax1.plot(s_values*1e-6, csecs_1)
ax1.set_xlabel("s [GeV$^2$]")
ax1.set_ylabel("$\sigma$ [mbarn]")
ax1.grid()
fig1.suptitle(r"$\pi^+ \pi^+ \rightarrow \pi^+ \pi^+$ cross section")

fig2, ax2 = plt.subplots()
ax2.plot(s_values*1e-6, csecs_2)
ax2.set_xlabel("s [GeV$^2$]")
ax2.set_ylabel("$\sigma$ [mbarn]")
ax2.grid()
fig2.suptitle(r"$\pi^+ \pi^- \rightarrow \pi^+ \pi^-$ cross section")
plt.show()