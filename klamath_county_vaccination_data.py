# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:11:52 2026

@author: rpwar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


filename_1 = r"C:\Users\rpwar\Documents\OIT\YEAR FOUR\measles_research\measles_data\klamath_county_data\K12 School Level Imm_Exempt.csv"

K12_imex2026 = pd.read_csv(filename_1, keep_default_na = True,encoding = 'utf-16', index_col = 1, sep = '\t')

K12_imex2026 = K12_imex2026.replace({'\%': '', " , Klamath": ''}, regex=True)

K12_imex2026["%All NME"] = K12_imex2026["%All NME"].astype(float) 

K12_imex2026["%MMR1"] = K12_imex2026["%MMR1"].astype(float) 

K12_imex2026["%MMR2"] = K12_imex2026["%MMR2"].astype(float) 


plt.figure(figsize=(15,7.5))
plt.clf()
ax = K12_imex2026["%All NME"].plot(x = K12_imex2026.iloc[:,0], kind = 'bar', alpha = 0.8, rot=90, width = 0.9, fontsize = 10, linewidth = 0.4, color = 'cornflowerblue')
ax.bar_label(ax.containers[0])
plt.ylim(0,20)
plt.ylabel(K12_imex2026.columns[-2], rotation = 90, fontsize = 10)
plt.title("2025-2026 K-12 %All NME\nKlamath County", fontsize = 20)
plt.tight_layout(pad=0.4, w_pad =2, h_pad=1.0)
plt.show()





























