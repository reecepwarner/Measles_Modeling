# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 16:20:47 2026

@author: rpwar
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, LinearConstraint, fmin
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts,amsmath,amssymb,amsthm,mathpazo, bm,mathabx, tgheros, helvet}'

textwidth = 542.025

figwidth = textwidth/72.27
figheight = figwidth*0.61


filename_1 = r"C:\Users\rpwar\Documents\OIT\YEAR FOUR\measles_research\oregon_data\Oregon's_Weekly_Reportable_Disease_Data_20260916.csv"

filename_2 = r"C:\Users\rpwar\Documents\OIT\YEAR FOUR\measles_research\oregon_data\measles_vaccination\T1. Measles Vaccination Data by Age_data.csv"


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

def V0_i(i0, sigma, beta, gamma):
    return (gamma/beta + i0 - 1)/(sigma - 1)

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
# parameter estimation
# =============================================================================

Oregon_Measles_Vaccination = pd.read_csv(filename_2, keep_default_na = True, index_col = 0)


Oregon_Disease_df = pd.read_csv(filename_1, keep_default_na = True, index_col = 0)


Oregon_Measles2020_2026_df = Oregon_Disease_df.loc[Oregon_Disease_df.Disease=="Measles" ]


Oregon_Measles_2026_df = Oregon_Measles2020_2026_df.loc[Oregon_Measles2020_2026_df.MMWRyear==2026]

Oregon_Measles_2026_df = Oregon_Measles_2026_df.sort_values(by = "MMWRweek", ascending = True)


last_date = Oregon_Measles_2026_df.MMWRweek.max()

time_data = Oregon_Measles_2026_df.MMWRweek.to_numpy()
measles_data = Oregon_Measles_2026_df.Cases.to_numpy()



approx_pop = Oregon_Measles_Vaccination.loc["State"].Denominator.sum()
approx_vax = Oregon_Measles_Vaccination.loc["State"].Count.sum()



measles_data = measles_data/approx_pop

v0_oregon = approx_vax/approx_pop

i0_oregon = measles_data[0]

s0_oregon = 1 - v0_oregon - i0_oregon

r0_oregon = 0

S0_oregon = np.array([s0_oregon,
                      i0_oregon,
                      r0_oregon,
                      v0_oregon])

# weeks
time0 = time_data[0]

timef = 144

num_data = 7*(timef-time0)

def L(p):
    p0, p1, p2 = p
    X = solve_ivp(SIRV, [time0, timef], S0_oregon, method = 'RK45', args = (p0, p1, p2), t_eval = np.linspace(time0, timef, num_data + 1))
    X = X.y
    S = X[0,:]
    I = X[1,:]
    
    I_tilde = p1*S[0:36*7:7]*I[0:36*7:7]
    residual = I_tilde - measles_data
    
    return np.dot(residual, residual)





p0 = 0.1*np.ones(3)


constraint1 = LinearConstraint(np.array([1, 0, 0]), lb = 0, ub = 5.)
constraint2 = LinearConstraint(np.array([0, 1, 0]), lb = 0, ub = 5.)
constraint3 = LinearConstraint(np.array([0, 0, 1]), lb = 0, ub = 1.)


pstar = minimize(L, p0, method = 'COBYLA', constraints = (constraint1, constraint2, constraint3) ).x



beta_oregon = pstar[0]
gamma_oregon = pstar[1]
sigma_oregon = pstar[2]

R0_oregon = R0(v0_oregon, s0_oregon, beta_oregon, gamma_oregon, sigma_oregon)

print("beta_OR = %1.3f" %(beta_oregon))
print("gamma_OR = %1.3f" %(gamma_oregon))
print("sigma_OR = %1.3f" %(sigma_oregon))
print("R0_OR = %1.3f" %(R0_oregon))

X = solve_ivp(SIRV, [time0, timef], S0_oregon, method = 'RK45', args = (beta_oregon, gamma_oregon, sigma_oregon), t_eval = np.linspace(time0, timef, num_data + 1))
t_oregon = X.t
X = X.y
S_oregon = X[0,:]
I_oregon = X[1,:]
R_oregon = X[2,:]
V_oregon = X[3,:]

new_cases_oregon = beta_oregon*S_oregon*I_oregon

num_points = 100

sigmas_oregon = np.linspace(0, 1.0, num_points)

v0s_oregon = np.linspace(0, 1.0, num_points)

SIG_oregon, V0s_oregon = np.meshgrid(sigmas_oregon, v0s_oregon)

R0s_oregon = beta_oregon/gamma_oregon*(1 - V0s_oregon - i0_oregon + SIG_oregon*V0s_oregon)




V_sig_oregon = V0_i(i0_oregon, sigmas_oregon[:-1], beta_oregon, gamma_oregon)

V_sig_trunc_oregon = V_sig_oregon[:51]

sigmas_trunc_oregon = sigmas_oregon[:51]



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






# =============================================================================
# Oregon parameter estimation plots
# =============================================================================


# R0, v0, sigma heat map

fig3, ax3 = plt.subplots(1,1, figsize = (figwidth, figwidth))

c2 = ax3.pcolormesh(SIG_oregon, V0s_oregon, R0s_oregon,cmap = 'magma', )

ax3.plot(sigmas_trunc_oregon, V_sig_trunc_oregon, color = 'cornflowerblue', linewidth = 3, label = r'$\mathrm{R}_0=1$')
ax3.plot(sigma_oregon, v0_oregon, linestyle = '', marker = '*', markersize = 10, color =  'lime', label = r'$(\sigma_{\text{OR}},V_{\text{OR}}^{0})$')

ax3.legend(loc = 'upper right', fontsize = 15)


ax3.set_xlim(0, 1.0)
ax3.set_ylim(0, 1.0)
ax3.set_xticks(np.arange(0, 1.04, 1/20), minor = True)
ax3.set_yticks(np.arange(0, 1.04, 1/20), minor = True)
ax3.set_xlabel(r"$\sigma$", fontsize = 20)
ax3.set_ylabel(r"$V_{0}$", fontsize = 20, rotation = 0)
ax3.tick_params(axis = 'both', labelsize = 15, pad = 1)

# colorbar and formatting
cbar2 = fig3.colorbar(c2, ax = ax3)
cbar_ax2 = cbar2.ax
cbar2.set_label(label = r'$\mathrm{R}_0$', rotation = 0, fontsize = 20, loc = 'center', labelpad = 10)
cbar2.set_ticks(np.arange(0, np.ceil(R0s_oregon.max()*100)/100, 0.25), minor = True)


fig3.suptitle("2026 Oregon Data $\mathrm{R}_0(V_0, \sigma)$ Heat Map", fontsize = 25)
fig3.tight_layout()
fig3.show()



# time-series plots

fig4, ax4 = plt.subplots(1, 2, figsize = (figwidth, figheight))

ax4[0].plot(time_data, measles_data, linestyle = ' ', marker = '*', color = 'black', markersize = 10, label = r'$j_{\text{OR}}$')
ax4[0].plot(t_oregon, new_cases_oregon, linewidth = 3, color = 'cornflowerblue', label = r'$\beta SI$')



ax4[0].set_xlim(time0, timef)
ax4[0].set_ylim(0-np.ceil(new_cases_oregon.max()*1000)/100000, np.ceil(new_cases_oregon.max()*1000)/1000 )
ax4[0].set_xticks(np.arange(time0, timef, timef/8))
ax4[0].set_yticks(np.arange(0, np.ceil(new_cases_oregon.max()*1000)/1000,np.ceil(new_cases_oregon.max()*1000)/10000 ))
ax4[0].ticklabel_format(axis = 'y', style = 'scientific', scilimits = (-2, -3))
ax4[0].set_xlabel(r"$t$ [weeks]", fontsize = 20)
ax4[0].set_ylabel("\%pop", fontsize = 20, rotation = 90)


ax4[1].plot(time_data, measles_data, linestyle = ' ', marker = '*', color = 'black', markersize = 10, label = r'$j_{\text{OR}}$')
ax4[1].plot(t_oregon, new_cases_oregon, linewidth = 3, color = 'cornflowerblue', label = r'$\beta SI$')
ax4[1].legend(fontsize = 15)

ax4[1].set_xlim(time_data[0], time_data[-1])
ax4[1].set_ylim(0-np.ceil(measles_data.max()*10000)/100000, np.ceil(measles_data.max()*10000)/10000 )
ax4[1].set_xticks(np.arange(time_data[0], time_data[-1], time_data[-1]/9))
ax4[1].set_yticks(np.arange(0, np.ceil(measles_data.max()*10000)/10000,np.ceil(measles_data.max()*10000)/100000 ))
ax4[1].ticklabel_format(axis = 'y', style = 'scientific', scilimits = (-2, -5))
ax4[1].set_xlabel(r"$t$ [weeks]", fontsize = 20)
ax4[1].set_ylabel("\%pop", fontsize = 20, rotation = 90)


fig4.suptitle(r"2026 OR Measles Data PE New Cases", fontsize = 25)
fig4.tight_layout()
fig4.show()


fig5, ax5 = plt.subplots(1,1, figsize = (figwidth, figheight))

ax5.plot(t_oregon, S_oregon, color = 'cornflowerblue', linewidth = 3, label = r'$S(t)$')
ax5.plot(t_oregon, I_oregon, color = 'tomato', linewidth = 3, label = r'$I(t)$')
ax5.plot(t_oregon, R_oregon, color = 'lime', linewidth = 3, label = r'$R(t)$')
ax5.plot(t_oregon, V_oregon, color = 'mediumorchid', linewidth = 3, label = r'$V(t)$')
leg5 = ax5.legend(loc = 'upper right', fontsize = 15)


textstr = '\n'.join((r"$\mathrm{R}_0 = %1.3f$" %(R0_oregon),
                     ))

props5 = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


tbbox = leg5.get_window_extent()
inv = plt.gca().transAxes.inverted()
x0, y0 = inv.transform((tbbox.x0, tbbox.y0))

ax5.text(x0 + 0.033, y0-0.2, s = textstr, bbox = props, fontsize = 15, transform = plt.gca().transAxes)



ax5.set_xlim(time0, timef+1)
ax5.set_ylim(0,1.01)
ax5.set_xlabel(r"$t$ [weeks]", fontsize = 20)
ax5.set_ylabel("\%pop", fontsize = 20, rotation = 90)
ax5.set_xticks(np.arange(0, timef+1, timef//10))
ax5.set_yticks(np.arange(0.1, 1.1, 0.1))
ax5.tick_params(axis = 'both', labelsize = 15)

fig5.suptitle("OR Measles Data PE SIRV Solutions", fontsize = 25)
fig5.tight_layout()
fig5.show()


































