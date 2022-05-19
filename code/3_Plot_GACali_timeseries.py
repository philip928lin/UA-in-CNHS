import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import HydroCNHS
import HydroCNHS.calibration as cali
import matplotlib.pyplot as plt
from collections import OrderedDict

path = r""
cali_output_path = r""
folder_list = os.listdir(cali_output_path)

#%%
ls_dict = OrderedDict(
    [('solid',               (0, ())),
     ('dotted',              (0, (1, 5))),
     ('densely dashed',      (0, (5, 1))),
     ('densely dotted',      (0, (1, 1))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('dashed',              (0, (5, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
     ('loosely dotted',      (0, (1, 10))),
     ('loosely dashed',      (0, (5, 10)))])
ls_list = list(ls_dict.keys())
#%%
def evaluation(individual, info):
    fitness=0
    return (fitness,)

numiter = 100
name = ["$M_{S}$", "$M_{A,L}$", "$M_{A,Q}$", "$M_{L,L}$", "$M_{L,Q}$"]


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6,4), sharey=True, sharex=True)
plt.tight_layout(w_pad=0, rect=(0,0,1,1))
axes1 = axes.flatten()
axes2 = [ax.twinx() for ax in axes1]
for i, item in enumerate(["Static-Linear", "Adaptive-Linear",
                          "Adaptive-Quadratic", "Learning-Linear",
                          "Learning-Quadratic"]):
    ax1 = axes1[i]
    ax2 = axes2[i]
    l = 0
    for h in [1,2]:
        for seed in [9,28,83]:
            f = "Cali_ABM-{}_KGE_{}_hydro{}".format(item, seed, h)
            cali_save_path = os.path.join(cali_output_path, f, "GA_auto_save.pickle")
            ga = cali.GA_DEAP(evaluation)
            ga.load(cali_save_path, max_gen="")
            fit_list = ga.summary["max_fitness"]
            std_list = ga.summary["std"]
            ax1.plot(fit_list, ls=ls_dict[ls_list[l]], zorder=3, color="k", label="Seed{}-HydroEMR{}".format(seed, h))
            ax2.plot(std_list, ls=ls_dict[ls_list[l]], zorder=-1, color="grey")
            l += 1
    ax1.set_title(name[i])
    ax1.set_ylim([1,1.5])
    ax2.set_ylim([0,0.25])
    ax1.set_xlim([0,numiter])
    ax2.set_xlim([0,numiter])
    if i != 2 and i != 4:
        ax2.set_yticklabels([])

ax1.legend(loc="upper right", bbox_to_anchor=(2.4, 0.9), fontsize=9)
axes1[-1].set_axis_off()
axes2[-1].set_axis_off()
ax = fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False,
                left=False, right=False)
plt.box(False)
plt.xlabel("Generation")
plt.ylabel("Objective value")
plt.xlim([0,70])
plt.ylim([0,1.5])
plt.text(77, -0.12, "Standard deviation of within-population objective values", rotation=90)
# ax = ax.twinx()
# ax.set_yticks([])
# plt.box(False)
# plt.tick_params(labelcolor='none', top=False, bottom=False,
#                 left=False, right=False)
# plt.yticks(color='w')
# plt.ylabel("\nStandard deviation of within-population objective values")


#%%
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
l = 0
for seed in [9, 13, 17, 19, 23, 28, 29, 31, 37, 83]:#[9,28,83]:
    f = "Cali_C1C2G_KGE_{}".format(seed)
    cali_save_path = os.path.join(cali_output_path, f, "GA_auto_save.pickle")
    ga = cali.GA_DEAP(evaluation)
    ga.load(cali_save_path, max_gen="")
    fit_list = ga.summary["max_fitness"]
    std_list = ga.summary["std"]
    ax1.plot(fit_list, ls=ls_dict[ls_list[l]], zorder=3, color="k", label="Seed{}".format(seed))
    ax2.plot(std_list, ls=ls_dict[ls_list[l]], zorder=1, color="grey")
    l += 1
ax1.set_ylim([-20,11])
ax1.set_xlim([0,100])
ax2.set_xlim([0,100])
ax1.set_xlabel("Generation")
ax1.set_ylabel("Objective value")
ax2.set_ylabel("Standard deviation of \nwithin-population objective values")
#ax1.legend(loc="upper right", bbox_to_anchor=(1.01, 1.11), fontsize=9, ncol=3)
ax1.legend(loc="upper right", bbox_to_anchor=(1.07, 1.16), fontsize=9, ncol=5)
#%%

f = folder_list[0]
cali_save_path = os.path.join(cali_output_path, f, "GA_auto_save.pickle")
ga = cali.GA_DEAP(evaluation)
ga.load(cali_save_path, max_gen="")
sol = ga.solution
fit_list = ga.summary["max_fitness"]
std_list = ga.summary["std"]

fig, axes = plt.subplots()
ax1 = axes
ax2 = ax1.twinx()
ax1.plot(fit_list)
ax2.plot(std_list)






































