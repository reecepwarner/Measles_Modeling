# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 08:26:33 2026

@author: rpwar
"""

import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts,amsmath,amssymb,amsthm,mathpazo, bm,mathabx, tgheros, helvet}'


def IV(t, S, mu, alpha, beta, sigma, phi, N):
    i, v = S
    return np.array([beta*(N-i-(1-sigma)*v)*i - (mu + alpha)*i,
                     phi*(N-i) - sigma*beta*v*i - (mu + phi)*v])




def J(I, V, alpha, beta, sigma, phi, N):
    return np.array([[-2*beta*I - (1 - sigma)*beta*V - (mu + alpha) + beta*N, -(1 - sigma)*beta*I],
                     [-(phi + sigma*beta*V), -(mu + phi + sigma*beta*I)]])



# population carrying capacity, initial compartment populations

K = 1

s0 = 0.9
i0 = 0.0
v0 = 0.09

I0 = np.array([i0, v0])
# birth, death, and contact rates

beta = 0.5
alpha = 0.2

phi = 0.3
sigma = 0.2

mu = 0.01

R0 = (K*beta)/(mu + alpha)

p = np.array([mu, alpha, beta])
# init conditions


def R_phi(beta):
    return (K*beta)/(mu + alpha)*(mu + sigma*phi)/(mu + phi)


def beta_crit(mu, alpha, sigma, phi, N):
    return (sigma*(mu + alpha) + 2*np.sqrt(sigma*(1 - sigma)*(mu + alpha)*phi) - (mu + sigma*phi))/(sigma*N)

def Beta(R0):
    return R0*(mu + alpha)*(mu + phi)/(K*(mu + sigma*phi))

def I_plus(R0):
    beta = Beta(R0)
    A = sigma*beta
    B = (mu + sigma*phi) + sigma*(mu + alpha) - sigma*beta*K
    C = (mu + alpha)*(mu + phi)/(beta) - (mu + sigma*phi)*K
    
    return (-B + np.sqrt(B**2 - 4*A*C))/(2*A)

def I_minus(R0):
    beta = Beta(R0)
    A = sigma*beta
    B = (mu + sigma*phi) + sigma*(mu + alpha) - sigma*beta*K
    C = (mu + alpha)*(mu + phi)/(beta) - (mu + sigma*phi)*K
    
    return (-B - np.sqrt(B**2 - 4*A*C))/(2*A)    




# simulations


t0 = 0
tf = 1000


R_0 = np.linspace(0.92, 1.2, 8000)

I_p = I_plus(R_0)
I_m = I_minus(R_0)


K = 1


# i0 = (I_p[np.where(~np.isnan(I_p))[0][0]] + I_m[np.where(~np.isnan(I_m))[0][0]])/2
# i0 = (I_p[np.where(~np.isnan(I_p))[0][-1]] )
i0 = 0.1
v0 = 0.1
s0 = K - i0 - v0
I0 = np.array([i0, v0])





beta_critical = beta_crit(mu, alpha, sigma, phi, K )
R_critical = R_phi(beta_critical)




R_min_end = R_critical
beta_end = Beta(R_min_end)
beta_end = Beta(R_min_end)


I_stable = I_plus(R_min_end)



R_min_nil = 0.923

beta_nil = Beta(R_min_nil)





IV_end = solve_ivp(IV, [t0, tf], I0, method = 'RK45', args = (mu, alpha, beta_end, sigma, phi, K))
t_IV_end = IV_end.t
IV_end = IV_end.y
I_end = IV_end[0,:]
V_end = IV_end[1,:]
S_end = K - I_end - V_end


IV_nil = solve_ivp(IV, [t0, tf], I0, method = 'RK45', args = (mu, alpha, beta_nil, sigma, phi, K))
t_IV_nil = IV_nil.t
IV_nil = IV_nil.y
I_nil = IV_nil[0,:]
V_nil = IV_nil[1,:]
S_nil = K - I_nil - V_nil





fig1, ax1 = plt.subplots(1,1, figsize = (8.5, 8.5))
# backward bifurcation diagram
ax1.plot(R_0, I_p, color = 'cornflowerblue', label = 'stable', linewidth = 3)
ax1.plot(np.linspace(0, 1.0, 101), 0*np.linspace(0, 1.0, 101), color = 'cornflowerblue', linewidth = 3 )
ax1.plot(R_0, I_m, color = 'tomato', label = 'unstable', linestyle = 'dashed', linewidth = 3)
ax1.plot(np.linspace(1.0, 1.2, 101), 0*np.linspace(1.0, 1.2, 101) , color = 'tomato', linestyle = 'dashed', linewidth = 3)

textstr = '\n'.join((r"$\mathrm{R}_{c} = $%1.3f" %(R_critical),
                     ))

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)



leg2 = ax1.legend(fontsize = 15, loc = 'upper left')
tbbox = leg2.get_window_extent()
inv = plt.gca().transAxes.inverted()
x1, y1 = inv.transform((tbbox.x0, tbbox.y0))

ax1.text(x1+0.01, y1-0.05, s = textstr, bbox = props, fontsize = 15, transform = plt.gca().transAxes)


ax1.set_xlim(np.floor(100*R_0[np.where(~np.isnan(I_p))[0][0]])/100, 1.2)
ax1.set_ylim(0, np.ceil(100*np.max(I_p[~np.isnan(I_p)]))/100)
ax1.set_xlabel(r"$\mathrm{R}_0$", fontsize = 15)
ax1.set_ylabel(r"$I$", fontsize = 15, rotation = 0)
ax1.tick_params(axis = 'both', labelsize = 10)
fig1.suptitle("SIV Model Backward Bifurcation Diagram", fontsize = 20)




fig2, ax2 = plt.subplots(1,2, figsize = (17, 8.5))

# SIV time-series, infective endemic equlibrium

ax2[0].plot(R_0, I_p, color = 'cornflowerblue', label = 'stable', linewidth = 3)
ax2[0].plot(np.linspace(0, 1.0, 101), 0*np.linspace(0, 1.0, 101), color = 'cornflowerblue', linewidth = 3 )
ax2[0].plot(R_0, I_m, color = 'tomato', label = 'unstable', linestyle = 'dashed', linewidth = 3)
ax2[0].plot(np.linspace(1.0, 1.2, 101), 0*np.linspace(1.0, 1.2, 101) , color = 'tomato', linestyle = 'dashed', linewidth = 3)
ax2[0].plot(R_min_end, i0, linestyle = ' ', marker = '*', markersize = 8, color = 'black', label = 'init')

ax2[0].set_xlim(np.floor(100*R_0[np.where(~np.isnan(I_p))[0][0]])/100, 1.2)
ax2[0].set_ylim(0, np.ceil(100*np.max(I_p[~np.isnan(I_p)]))/100)
ax2[0].set_xlabel(r"$\mathrm{R}_0$", fontsize = 15)
ax2[0].set_ylabel(r"$I$", fontsize = 15, rotation = 0)
ax2[0].tick_params(axis = 'both', labelsize = 10)
ax2[0].legend(fontsize = 15)


ax2[1].plot(t_IV_end, S_end, color = 'cornflowerblue', label = r"$S(t)$", linewidth = 3)
ax2[1].plot(t_IV_end, I_end, color = 'tomato', label = r'$I(t)$', linewidth = 3)
ax2[1].plot(t_IV_end, V_end, color = 'lime', label = r'$V(t)$', linewidth = 3)
ax2[1].plot(np.linspace(t0, tf, 100), I_stable*np.ones(100), color = 'black', linestyle = 'dashed', linewidth = 2, label = r'$I_{\text{stable}}$')



textstr = '\n'.join((r"$\mathrm{R}_{0} = $%1.3f" %(R_min_end),
                     ))

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)



leg2 = ax2[1].legend(fontsize = 15, loc = 'upper left')
tbbox = leg2.get_window_extent()
inv = plt.gca().transAxes.inverted()
x1, y1 = inv.transform((tbbox.x0, tbbox.y0))

ax2[1].text(x1+0.01, y1-0.05, s = textstr, bbox = props, fontsize = 15, transform = plt.gca().transAxes)

ax2[1].set_xlim(0, tf+1)
ax2[1].set_ylim(0,1.01)
ax2[1].set_xlabel(r"$t$ [a.u.]", fontsize = 15)
ax2[1].set_ylabel("\%pop", fontsize = 15, rotation = 90)
ax2[1].set_xticks(np.arange(100, tf+1, 100))
ax2[1].set_yticks(np.arange(0, 1.1, 0.1))
ax2[1].tick_params(axis = 'both', labelsize = 10)
fig2.suptitle("SIV Time-series Endemic Equilibrium", fontsize = 20)
fig2.tight_layout()
fig2.show()

# # SIV time-series, infective nil equilibrium

fig3, ax3 = plt.subplots(1,2, figsize = (17, 8.5))

ax3[0].plot(R_0, I_p, color = 'cornflowerblue', label = 'stable', linewidth = 3)
ax3[0].plot(np.linspace(0, 1.0, 101), 0*np.linspace(0, 1.0, 101), color = 'cornflowerblue', linewidth = 3 )
ax3[0].plot(R_0, I_m, color = 'tomato', label = 'unstable', linestyle = 'dashed', linewidth = 3)
ax3[0].plot(np.linspace(1.0, 1.2, 101), 0*np.linspace(1.0, 1.2, 101) , color = 'tomato', linestyle = 'dashed', linewidth = 3)
ax3[0].plot(R_min_nil, i0, linestyle = ' ', marker = '*', markersize = 8, color = 'black', label = 'init')

ax3[0].set_xlim(np.floor(100*R_0[np.where(~np.isnan(I_p))[0][0]])/100, 1.2)
ax3[0].set_ylim(0, np.ceil(100*np.max(I_p[~np.isnan(I_p)]))/100)
ax3[0].set_xlabel(r"$\mathrm{R}_0$", fontsize = 15)
ax3[0].set_ylabel(r"$I$", fontsize = 15, rotation = 0)
ax3[0].tick_params(axis = 'both', labelsize = 10)
ax3[0].legend(fontsize = 15)


ax3[1].plot(t_IV_nil, S_nil, color = 'cornflowerblue', label = r"$S(t)$", linewidth = 3)
ax3[1].plot(t_IV_nil, I_nil, color = 'tomato', label = r'$I(t)$', linewidth = 3)
ax3[1].plot(t_IV_nil, V_nil, color = 'lime', label = r'$V(t)$', linewidth = 3)
ax3[1].plot(np.linspace(t0, tf, 100), 0*np.ones(100), color = 'black', linestyle = 'dashed', linewidth = 2, label = r'$I_{\text{stable}}$')




textstr = '\n'.join((r"$\mathrm{R}_{0} = $%1.3f" %(R_min_nil),
                     ))

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


leg2 = ax3[1].legend(fontsize = 15, loc = 'upper left')
tbbox = leg2.get_window_extent()
inv = plt.gca().transAxes.inverted()
x1, y1 = inv.transform((tbbox.x0, tbbox.y0))

ax3[1].text(x1+0.01, y1-0.05, s = textstr, bbox = props, fontsize = 15, transform = plt.gca().transAxes)


ax3[1].set_xlim(0, tf+1)
ax3[1].set_ylim(0,1.01)
ax3[1].set_xlabel(r"$t$ [a.u.]", fontsize = 15)
ax3[1].set_ylabel("\%pop", fontsize = 15, rotation = 90)
ax3[1].set_xticks(np.arange(100, tf+1, 100))
ax3[1].set_yticks(np.arange(0, 1.1, 0.1))
ax3[1].tick_params(axis = 'both', labelsize = 10)

fig3.suptitle("SIV Time-series DFE", fontsize = 20)
fig3.tight_layout()
fig3.show()

































