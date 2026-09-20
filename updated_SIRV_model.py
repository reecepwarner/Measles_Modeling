# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:20:47 2026

@author: rpwar
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts,amsmath,amssymb,amsthm,mathpazo, bm,mathabx, tgheros, helvet}'

textwidth = 542.025

figwidth = textwidth/72.27
figheight = figwidth*0.61


def SIRV(t, S, beta, gamma, sigma):
    s, i, r, v = S
    
    return np.array([-beta*s*i,
                     beta*s*i + sigma*beta*v*i - gamma*i,
                     gamma*i,
                     -sigma*beta*v*i])




def R0(v0, s0, beta, gamma, sigma):
    return beta/gamma*(s0 + sigma*v0)




# def V0(s0, sigma, beta, gamma):
#     return (gamma - beta*s0)/(beta*sigma)

def V0(sigma, beta, gamma):
    return (gamma/beta - 1)/(sigma - 1)

# def V0(i0, sigma, beta, gamma):
#     return (gamma + beta*(s0))/(beta*( sigma ))


beta = 0.4
gamma = 0.2
sigma = 0.4

r0 = 0.

i0 = 0.1


v0 = V0( sigma, beta, gamma) - 0.1
s0 = 1-v0-i0


t0 = 0
tf = 50





# v0 = 1/sigma*(gamma/beta - s0)





n = 500


BR = R0(v0, s0, beta, gamma, sigma)


S0 = np.array([s0, i0, r0, v0])

X = solve_ivp(SIRV, [t0, tf], S0, method = 'RK45', args = (beta, gamma, sigma), t_eval = np.linspace(t0, tf, n+1))
t = X.t
X = X.y
S = X[0,:]
I = X[1,:]
R = X[2,:]
V = X[3,:]


num_points = 100

sigmas = np.linspace(0, 1.0, num_points)

v0s = np.linspace(0, 1.0, num_points)

SIG, V0s = np.meshgrid(sigmas, v0s)

R0s = beta/gamma*(1-V0s + SIG*V0s)





V_sig = V0(sigmas[:-1], beta, gamma)


V_sig_trunc = V_sig[:51]

sigmas_trunc = sigmas[:51]

# =============================================================================
# time-series and heat map plots
# =============================================================================


# SIRV time-series
fig1, ax1 = plt.subplots(1,1, figsize = (figwidth, figheight))

ax1.plot(t, S, color = 'cornflowerblue', linewidth = 3, label = r'$S(t)$')
ax1.plot(t, I, color = 'tomato', linewidth = 3, label = r'$I(t)$')
ax1.plot(t, R, color = 'lime', linewidth = 3, label = r'$R(t)$')
ax1.plot(t, V, color = 'mediumorchid', linewidth = 3, label = r'$V(t)$')
leg1 = ax1.legend(loc = 'upper right', fontsize = 15)


textstr = '\n'.join((r"$\mathrm{R}_0 = %1.3f$" %(BR),
                     ))

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


tbbox = leg1.get_window_extent()
inv = plt.gca().transAxes.inverted()
x0, y0 = inv.transform((tbbox.x0, tbbox.y0))

ax1.text(x0 + 0.033, y0-0.2, s = textstr, bbox = props, fontsize = 15, transform = plt.gca().transAxes)


ax1.set_xlim(t0, tf)
ax1.set_xlim(0, tf+1)
ax1.set_ylim(0,1.01)
ax1.set_xlabel(r"$t$ [a.u.]", fontsize = 20)
ax1.set_ylabel("\%pop", fontsize = 20, rotation = 90)
ax1.set_xticks(np.arange(0, tf+1, tf//10))
ax1.set_yticks(np.arange(0.1, 1.1, 0.1))
ax1.tick_params(axis = 'both', labelsize = 15)

fig1.suptitle("SIRV Time-Series Solutions", fontsize = 25)
fig1.tight_layout()
fig1.show()



# R0, v0, sigma heat map

fig2, ax2 = plt.subplots(1,1, figsize = (figwidth, figwidth))

c = ax2.pcolormesh(SIG, V0s, R0s,cmap = 'magma', )

ax2.plot(sigmas_trunc, V_sig_trunc, color = 'cornflowerblue', linewidth = 3, label = r'$\mathrm{R}_0=1$')
ax2.legend(loc = 'upper right', fontsize = 15)

ax2.set_xlim(0, 1.0)
ax2.set_ylim(0, 1.0)
ax2.set_xticks(np.arange(0, 1.04, 1/20), minor = True)
ax2.set_yticks(np.arange(0, 1.04, 1/20), minor = True)
ax2.set_xlabel(r"$\sigma$", fontsize = 20)
ax2.set_ylabel(r"$V_{0}$", fontsize = 20, rotation = 0)
ax2.tick_params(axis = 'both', labelsize = 15, pad = 1)

# colorbar and formatting
cbar = fig2.colorbar(c, ax = ax2)
cbar_ax = cbar.ax
cbar.set_label(label = r'$\mathrm{R}_0$', rotation = 0, fontsize = 20, loc = 'center', labelpad = 10)
cbar.set_ticks(np.arange(0, 2.05, 0.05), minor = True)
#$cbar_ax.tick_params(labelpad = 10)

fig2.suptitle(r"SIRV $\mathrm{R}_0(V_0, \sigma)$ Heat Map", fontsize = 25)
fig2.tight_layout()
fig2.show()






























































