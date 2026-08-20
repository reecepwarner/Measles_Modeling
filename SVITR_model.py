# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 18:46:12 2026

@author: rpwar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin

kwale_infected = np.array([69, 77, 93, 111, 184, 112, 421, 137, 363, 388], 'double')
kwale_susceptible = np.array([222430, 229047, 235825, 242677, 262032, 269790, 277882, 286313, 295087, 304209], 'double')
kwale_vaccinated = np.array([39455, 40694, 31718, 44714, 41844, 43290, 53325, 51237, 61129, 55322], 'double')
kwale_treated = np.array([66, 71, 93, 108, 183, 110, 418, 136, 360, 386], 'double')
kwale_recovered = np.array([62, 69, 6, 108, 174, 110, 418, 136, 360, 385], 'double')


kilifi_infected = np.array([149, 97, 43, 49, 152, 645, 502, 144, 351, 362], 'double')
kilifi_vaccinated = np.array([45755, 56610, 48479, 68550, 70512, 68171, 75786, 73676, 75038, 79507], 'double')
kilifi_susceptible = np.array([366314, 375614, 385776, 396205, 407016, 417042, 427098, 437295, 447593, 458054], 'double')
kilifi_treated = np.array([147, 96, 42, 46, 149, 637, 499, 143, 351, 362], 'double')
kilifi_recovered = np.array([143, 93, 41, 45, 143, 618, 484, 139, 348, 350], 'double')

def RK4_sys(t0,tn,n,y0,diff, args):
    h = abs((tn-t0)/n)
    rows = len(y0)
    
    t = np.linspace(t0,tn,n+1)
    y = np.zeros((rows, n+1))
    
    y[:,0] = y0[:]
    
    for k in range(n):
        
        A = diff(t[k],y[:,k], args)
        B = diff( t[k] + (h/2), y[:,k] + (A*h)/2, args)
        C = diff( t[k]+ (h/2), y[:,k] + (B*h)/2, args)
        D = diff( t[k] + h, y[:,k] + C*h, args)
        y[:,k+1] = y[:,k] + (h/6)*(A + 2*B + 2*C + D)
    return t,y


def SVITR(t, S, p):
    
    s, v, i, T, r = S
    
    
    return np.array([p[0] + p[1]*v - (p[2]*i + p[3] + p[4])*s,
                     p[3]*s - (p[1] + p[4] + p[5])*v,
                     p[2]*s*i + p[5] - (p[6] + p[7] + p[4])*i,
                     p[6]*i - (p[8] + p[4] + p[7])*T,
                     p[8]*T - p[4]*r
                     ])

kwale0 = np.array([222430, 39455, 69, 66, 62], 'double')
kilifi0 = np.array([366314, 45755, 149, 147, 143 ], 'double')
data_name = ["Kwale 2015-2024","Kilifi 2015-2024"]

t0 = 0
tn = 10 # years
n = 40


# =============================================================================
# training via infective data
# =============================================================================

infective_data = kwale_infected
vax_data = kwale_vaccinated
sus_data = kwale_susceptible
treat_data = kwale_treated
recov_data = kwale_recovered

S0 = kwale0
scale_down = np.sum(S0)
years = len(infective_data)


def I_Loss(p):
    I = RK4_sys(t0, tn, n, S0/scale_down, SVITR, p)[1][2]
    I_tilde = I[0:-1:4]
    data = infective_data/scale_down
    residual = I_tilde-data
    
    return np.dot(residual, residual)


kwale_p = np.array([0.107, 1.186, 0.176, 0.05, 0.03, 0.007, 0.027, 0.988, 0.949])

p0 = kwale_p
I_pstar = fmin(I_Loss, p0)


t, S_opt = RK4_sys(t0, tn, n, S0/scale_down, SVITR, I_pstar)

# plt.figure(1)
# plt.clf()
# plt.plot(t, S_opt[2]*scale_down, label = 'I(t)')
# plt.plot(t[0:-1:4], infective_data, linestyle = ' ', marker = '^', color = 'red', label = data_name[0])
# plt.ylabel("I(t)")
# plt.xlabel("time [years]")
# plt.xticks(np.arange(0,11, 1))
# plt.yticks(np.arange(0,801, 100))
# plt.legend()
# plt.title("%s Measles Parameter Estimate Infectives" %(data_name[0]))
# plt.show()




# plt.figure(2)
# plt.clf()
# plt.plot(t, S_opt[1]*scale_down, label = 'V(t)', color = 'cornflowerblue')
# plt.plot(t[0:-1:4], vax_data, linestyle = ' ', marker = '^', color = 'red', label = data_name[0])
# plt.ylabel("V(t)")
# plt.xlabel("time [years]")
# # plt.xticks(np.arange(0,11, 1))
# # plt.yticks(np.arange(0,801, 100))
# plt.legend()
# plt.title("%s Measles Parameter Estimate Vaccinated" %(data_name[0]))
# plt.show()


# plt.figure(3)
# plt.clf()
# plt.plot(t, S_opt[3]*scale_down, label = 'T(t)')
# plt.plot(t[0:-1:4], treat_data, linestyle = ' ', marker = '^', color = 'red', label = data_name[0])
# plt.ylabel("T(t)")
# plt.xlabel("time [years]")
# # plt.xticks(np.arange(0,11, 1))
# # plt.yticks(np.arange(0,801, 100))
# plt.legend()
# plt.title("%s Measles Parameter Estimate Treated" %(data_name[0]))
# plt.show()


# plt.figure(4)
# plt.clf()
# plt.plot(t, S_opt[0]*scale_down, label = 'S(t)')
# plt.plot(t[0:-1:4], sus_data, linestyle = ' ', marker = '^', color = 'red', label = data_name[0])
# plt.ylabel("S(t)")
# plt.xlabel("time [years]")
# # plt.xticks(np.arange(0,11, 1))
# # plt.yticks(np.arange(0,801, 100))
# plt.legend()
# plt.title("%s Measles Parameter Estimate Susceptibles" %(data_name[0]))
# plt.show()


# plt.figure(5)
# plt.clf()
# plt.plot(t, S_opt[4]*scale_down, label = 'R(t)')
# plt.plot(t[0:-1:4], recov_data, linestyle = ' ', marker = '^', color = 'red', label = data_name[0])
# plt.ylabel("R(t)")
# plt.xlabel("time [years]")
# # plt.xticks(np.arange(0,11, 1))
# # plt.yticks(np.arange(0,801, 100))
# plt.legend()
# plt.title("%s Measles Parameter Estimate Recovered" %(data_name[0]))
# plt.show()

fig1, ax1 = plt.subplots(1,2, figsize = (10,5))
ax1[0].plot(t, S_opt[2]*scale_down, label = r'$\tilde{I}(t)$', color = 'cornflowerblue', linewidth = 3)
ax1[0].plot(t[0:-1:4], infective_data, linestyle = ' ', marker = '^', color = 'red', label = data_name[0], markersize = 8)
ax1[0].set_ylabel(r"Infectives", fontsize = 12)
ax1[0].set_xlabel("time [years]", fontsize = 12)
ax1[0].set_xticks(np.arange(0,11, 1))
ax1[0].set_yticks(np.arange(0,801, 100))
ax1[0].set_ylim(0,850)
ax1[0].legend(fontsize = 12)

ax1[1].plot(t, S_opt[1]*scale_down, label = r'$\tilde{V}(t)$', color = 'mediumorchid', linewidth = 3)
ax1[1].plot(t[0:-1:4], vax_data, linestyle = ' ', marker = '^', color = 'seagreen', label = data_name[0], markersize = 8)
ax1[1].set_ylabel(r"Vaccinated", fontsize = 12)
ax1[1].set_xlabel("time [years]", fontsize = 12)
ax1[1].set_xticks(np.arange(0,11, 1))
ax1[1].set_ylim(20000, 60000)
ax1[1].set_yticks(np.arange(20000, 60001, 5000))
ax1[1].legend(fontsize = 12)
fig1.suptitle("%s Measles Parameter Estimate From Infective Data" %(data_name[0]), fontsize = 15, fontweight = 'semibold')
fig1.tight_layout()
fig1.show()


# =============================================================================
# training via vax data
# =============================================================================

vaccinated_data = kwale_vaccinated

def V_Loss(p):
    V = RK4_sys(t0, tn, n, S0/scale_down, SVITR, p)[1][1]
    V_tilde = V[0:-1:4]
    data = vaccinated_data/scale_down
    residual = V_tilde-data
    
    return np.dot(residual, residual)


kwale_p = np.array([0.107, 1.186, 0.176, 0.05, 0.03, 0.007, 0.027, 0.988, 0.949])

p0 = kwale_p
V_pstar = fmin(V_Loss, p0)


t, S_opt = RK4_sys(t0, tn, n, S0/scale_down, SVITR, V_pstar)

fig2, ax2 = plt.subplots(1,2, figsize = (10,5))
ax2[0].plot(t, S_opt[2]*scale_down, label = r'$\tilde{I}(t)$', color = 'cornflowerblue', linewidth = 3)
ax2[0].plot(t[0:-1:4], infective_data, linestyle = ' ', marker = '^', color = 'red', label = data_name[0], markersize = 8)
ax2[0].set_ylabel(r"Infectives", fontsize = 12)
ax2[0].set_xlabel("time [years]", fontsize = 12)
ax2[0].set_xticks(np.arange(0,11, 1))
ax2[0].set_yticks(np.arange(0,801, 100))
ax2[0].set_ylim(0,850)
ax2[0].legend(fontsize = 12)

ax2[1].plot(t, S_opt[1]*scale_down, label = r'$\tilde{V}(t)$', color = 'mediumorchid', linewidth = 3)
ax2[1].plot(t[0:-1:4], vax_data, linestyle = ' ', marker = '^', color = 'seagreen', label = data_name[0], markersize = 8)
ax2[1].set_ylabel(r"Vaccinated", fontsize = 12)
ax2[1].set_xlabel("time [years]", fontsize = 12)
ax2[1].set_xticks(np.arange(0,11, 1))
ax2[1].set_ylim(20000, 60000)
ax2[1].set_yticks(np.arange(20000, 60001, 5000))
ax2[1].legend(fontsize = 12)
fig2.suptitle("%s Measles Parameter Estimate From Vaccinated Data" %(data_name[0]), fontsize = 15, fontweight = 'semibold')
fig2.tight_layout()
fig2.show()


























