# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 13:58:50 2026

@author: rpwar
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, fmin, LinearConstraint
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts,amsmath,amssymb,amsthm,mathpazo, bm,mathabx, tgheros, helvet}'

textwidth = 542.025

figwidth = textwidth/72.27
figheight = figwidth*0.61


def SIRV(t, S, beta, gamma, nu):
    s, i, r, v = S
    
    return np.array([-beta*s*i - nu*s,
                     beta*s*i - gamma*i,
                     gamma*i,
                     nu*s])



def SIRV_nd(t, S, k, b):
    s, i, r, v = S
    
    return np.array([-s*i - b*s,
                     s*i - k*i,
                     k*i,
                     b*s])




eta = 0.2

s0 = 1-eta
i0 = eta
r0 = 0.
v0= 0.

t0 = 0
tf = 20
Dt = 0.1

beta = 0.7
gamma = 0.2
nu = 0.5

k = gamma/beta
b = nu/beta


tau0 = 0
tauf = beta*tf
n = 100


S0 = np.array([s0, i0, r0, v0])

X = solve_ivp(SIRV, [t0, tf], S0, method = 'RK45', args = (beta, gamma, nu), t_eval = np.linspace(t0, tf, n+1))
t = X.t
X = X.y
S = X[0,:]
I = X[1,:]
R = X[2,:]
V = X[3,:]

X_nd = solve_ivp(SIRV_nd, [tau0, tauf], S0, method = 'RK45', args = (k, b), t_eval = np.linspace(tau0, tauf, n+1))
t_nd = X_nd.t
X_nd = X_nd.y
S_nd = X_nd[0,:]
I_nd = X_nd[1,:]
R_nd = X_nd[2,:]
V_nd = X_nd[3,:]


fig1, ax1 = plt.subplots(1,1, figsize = (figwidth, figheight))

ax1.plot(t, S, color = 'cornflowerblue', linewidth = 3, label = r'$S(t)$')
ax1.plot(t, I, color = 'tomato', linewidth = 3, label = r'$I(t)$')
ax1.plot(t, R, color = 'lime', linewidth = 3, label = r'$R(t)$')
ax1.plot(t, V, color = 'mediumorchid', linewidth = 3, label = r'$V(t)$')
ax1.legend(loc = 'upper right', fontsize = 15)

ax1.set_xlim(t0, tf)
ax1.set_xlim(0, tf+1)
ax1.set_ylim(0,1.01)
ax1.set_xlabel(r"$t$ [a.u.]", fontsize = 20)
ax1.set_ylabel("\%pop", fontsize = 20, rotation = 90)
ax1.set_xticks(np.arange(0, tf+1, 1))
ax1.set_yticks(np.arange(0.1, 1.1, 0.1))
ax1.tick_params(axis = 'both', labelsize = 15)

fig1.suptitle("SIRV Time-Series Solutions", fontsize = 25)
fig1.tight_layout()
fig1.show()


fig2, ax2 = plt.subplots(1,1, figsize = (figwidth, figheight))

ax2.plot(t_nd, S_nd, color = 'cornflowerblue', linewidth = 3, label = r'$S(\tau)$')
ax2.plot(t_nd, I_nd, color = 'tomato', linewidth = 3, label = r'$I(\tau)$')
ax2.plot(t_nd, R_nd, color = 'lime', linewidth = 3, label = r'$R(\tau)$')
ax2.plot(t_nd, V_nd, color = 'mediumorchid', linewidth = 3, label = r'$V(\tau)$')
ax2.legend(loc = 'upper right', fontsize = 15)

ax2.set_xlim(tau0, tauf)
ax2.set_xlim(0, tauf+1)
ax2.set_ylim(0,1.01)
ax2.set_xlabel(r"$\beta t$", fontsize = 20)
ax2.set_ylabel("\%pop", fontsize = 20, rotation = 90)
ax2.set_xticks(np.arange(0, tauf+1, 1))
ax2.set_yticks(np.arange(0.1, 1.1, 0.1))
ax2.tick_params(axis = 'both', labelsize = 15)

fig2.suptitle("ND SIRV Time-Series Solutions", fontsize = 25)
fig2.tight_layout()
fig2.show()



# =============================================================================
# parameter estimation on number of new infections
# =============================================================================


noise = 0.0001


np.random.seed(100)

I_data = S_nd[0:n:5]*I_nd[0:n:5] + noise*np.random.random(n//5)
t_data = t_nd[0:-1:5]

def L(p):
    p0, p1 = p
    X = solve_ivp(SIRV_nd, [tau0, tauf], S0, method = 'RK45', args = (p0, p1), t_eval = np.linspace(tau0, tauf, n+1))
    X = X.y
    S = X[0,:]
    I = X[1,:]
    
    I_tilde = S[-1:n+1:5]*I[-1:n+1:5]
    residual = I_tilde - I_data
    
    return np.dot(residual, residual)




p0 = np.ones(2)

constraint = LinearConstraint(np.array([[1,0],
                                        [0, 1]]), lb = 0, ub = 1)


p_star = minimize(L, p0, method = 'trust-constr', constraints = (constraint) ).x

X_est = solve_ivp(SIRV_nd, [tau0, tauf], S0, method = 'RK45', args = (p_star[0], p_star[1]), t_eval = np.linspace(tau0, tauf, n+1))
t_est = X_est.t
X_est = X_est.y
S_est = X_est[0,:]
I_est = X_est[1,:]
R_est = X_est[2,:]
V_est = X_est[3,:]


new_infectives = S_est*I_est

max_height = np.ceil(np.max(new_infectives*100))/100
#p_star_fmin = fmin(L, p0)


fig3, ax3 = plt.subplots(1,1, figsize = (figwidth, figheight))

ax3.plot(t_est, S_est*I_est, color = 'cornflowerblue', linewidth = 3, label = r'$S(\tau)I(\tau)$')
ax3.plot(t_data, I_data, color = 'black', linestyle = ' ', marker = '*', markersize = 10, label = 'data')
leg3 = ax3.legend(loc = 'upper right', fontsize = 15)


p_vect_text = (r"$\mathbf{p}^{*}=\begin{bmatrix}"
               r"%1.3f\\"
               r"%1.3f\\"
               r"\end{bmatrix}$" %(p_star[0], p_star[1]))


p_text = (r"$\mathbf{p}_{\text{gen}}=\begin{bmatrix}"
               r"%1.3f\\"
               r"%1.3f\\"
               r"\end{bmatrix}$" %(k, b))

textstr = '\n'.join((r"%s" %(p_vect_text),
                     r"%s" %(p_text),
                     ))

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)


tbbox = leg3.get_window_extent()
inv = plt.gca().transAxes.inverted()
x0, y0 = inv.transform((tbbox.x0, tbbox.y0))

ax3.text(x0+0.03, y0-0.4, s = textstr, bbox = props, fontsize = 12, transform = plt.gca().transAxes)



ax3.set_xlim(tau0, tauf)
ax3.set_xlim(0, tauf+1)
ax3.set_ylim(0,max_height)
ax3.set_xlabel(r"$\beta t$", fontsize = 20)
ax3.set_ylabel("\%pop", fontsize = 20, rotation = 90)
ax3.set_xticks(np.arange(0, tauf+1, 1))
ax3.set_yticks(np.arange(0.005, max_height + 0.005, 0.01))
ax3.tick_params(axis = 'both', labelsize = 15)
fig3.suptitle("New Cases PE\nNoise = %.4f" %(noise), fontsize = 25)
fig3.tight_layout()
fig3.show()

fig4, ax4 = plt.subplots(2, 2, figsize = (figwidth, figheight))

# plot 1

ax4[0,0].plot(t_est, S_est, color = 'cornflowerblue', linewidth = 3, label = r"$S(t)$")
ax4[0,0].plot(t_nd, S_nd, color = 'red', linestyle = 'dashed', label = r'$S_{\text{gen}}$')
ax4[0,0].legend(loc = 'upper right', fontsize = 10)

ax4[0,0].set_xlim(tau0, tauf)
ax4[0,0].set_xlim(0, tauf+1)
ax4[0,0].set_ylim(0,1.1)
ax4[0,0].set_xlabel(r"$\beta t$", fontsize = 15)
ax4[0,0].set_ylabel("\%pop", fontsize = 15, rotation = 90)
ax4[0,0].set_xticks(np.arange(0, tauf+1, 1))
ax4[0,0].set_yticks(np.arange(0.0, 1.1, 0.1))
ax4[0,0].tick_params(axis = 'both', labelsize = 10)
ax4[0,0].set_title("(a)", fontsize = 15)

# plot 2

ax4[0,1].plot(t_est, I_est, color = 'cornflowerblue', linewidth = 3, label = r"$I(t)$")
ax4[0,1].plot(t_nd, I_nd, color = 'red', linestyle = 'dashed', label = r'$I_{\text{gen}}$')
ax4[0,1].legend(loc = 'upper right', fontsize = 10)

ax4[0,1].set_xlim(tau0, tauf)
ax4[0,1].set_xlim(0, tauf+1)
ax4[0,1].set_ylim(0,1.1)
ax4[0,1].set_xlabel(r"$\beta t$", fontsize = 15)
ax4[0,1].set_ylabel("\%pop", fontsize = 15, rotation = 90)
ax4[0,1].set_xticks(np.arange(0, tauf+1, 1))
ax4[0,1].set_yticks(np.arange(0.0, 1.1, 0.1))
ax4[0,1].tick_params(axis = 'both', labelsize = 10)
ax4[0,1].set_title("(b)", fontsize = 15)

# plot 3

ax4[1,0].plot(t_est, R_est, color = 'cornflowerblue', linewidth = 3, label = r"$R(t)$")
ax4[1,0].plot(t_nd, R_nd, color = 'red', linestyle = 'dashed', label = r'$R_{\text{gen}}$')
ax4[1,0].legend(loc = 'upper right', fontsize = 10)

ax4[1,0].set_xlim(tau0, tauf)
ax4[1,0].set_xlim(0, tauf+1)
ax4[1,0].set_ylim(0,1.1)
ax4[1,0].set_xlabel(r"$\beta t$", fontsize = 15)
ax4[1,0].set_ylabel("\%pop", fontsize = 15, rotation = 90)
ax4[1,0].set_xticks(np.arange(0, tauf+1, 1))
ax4[1,0].set_yticks(np.arange(0.0, 1.1, 0.1))
ax4[1,0].tick_params(axis = 'both', labelsize = 10)
ax4[1,0].set_title("(c)", fontsize = 15)

# plot 4

ax4[1,1].plot(t_est, V_est, color = 'cornflowerblue', linewidth = 3, label = r"$V(t)$")
ax4[1,1].plot(t_nd, V_nd, color = 'red', linestyle = 'dashed', label = r'$V_{\text{gen}}$')
ax4[1,1].legend(loc = 'upper right', fontsize = 10)

ax4[1,1].set_xlim(tau0, tauf)
ax4[1,1].set_xlim(0, tauf+1)
ax4[1,1].set_ylim(0,1.1)
ax4[1,1].set_xlabel(r"$\beta t$", fontsize = 15)
ax4[1,1].set_ylabel("\%pop", fontsize = 15, rotation = 90)
ax4[1,1].set_xticks(np.arange(0, tauf+1, 1))
ax4[1,1].set_yticks(np.arange(0.0, 1.1, 0.1))
ax4[1,1].tick_params(axis = 'both', labelsize = 10)
ax4[1,1].set_title("(c)", fontsize = 15)

fig4.suptitle("SIRV Time-Series from PE\nNoise = %.4f" %(noise), fontsize = 25)
fig4.tight_layout()
fig4.show()
# ax3[1].plot(t_est, S_est, color = 'cornflowerblue', linewidth = 3, label = r'$S(\tau)$')
# ax3[1].plot(t_est, I_est, color = 'tomato', linewidth = 3, label = r'$I(\tau)$')
# ax3[1].plot(t_est, R_est, color = 'lime', linewidth = 3, label = r'$R(\tau)$')
# ax3[1].plot(t_est, V_est, color = 'mediumorchid', linewidth = 3, label = r'$V(\tau)$')


# ax3[1].set_xlim(tau0, tauf)
# ax3[1].set_xlim(0, tauf+1)
# ax3[1].set_ylim(0,1.1)
# ax3[1].set_xlabel(r"$\beta t$", fontsize = 20)
# ax3[1].set_ylabel("\%pop", fontsize = 20, rotation = 90)
# ax3[1].set_xticks(np.arange(0, tauf+1, 1))
# ax3[1].set_yticks(np.arange(0.0, 1.1, 0.1))
# ax3[1].tick_params(axis = 'both', labelsize = 15)
# ax3[1].set_title("(b)", fontsize = 20)





































