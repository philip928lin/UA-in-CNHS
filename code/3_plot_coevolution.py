import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
pc = ""
path = r"".format(pc)
os.chdir(os.path.join(path, "NewCode"))
from decomposition_funcs import get_EV, get_VE, get_EVE, get_EV2

sc_result_path = os.path.join(path, "NewYRB_Sc")

# Sc settings
num_iter = 30
rcp_list = ["rcp26", "rcp85"]
q_list = ["10","30","50","70","90"]
num_realization = 20
# EMRs settings
num_hy_emr = 2
num_abm_emr = 4
bt_list = ["Learning", "Adaptive", "Static"]
at_list = ["Quadratic", "Linear"]

agtype_list = [('Static', 'Linear'),
               ('Adaptive', 'Linear'),('Adaptive', 'Quadratic'),
               ('Learning', 'Linear'),('Learning', 'Quadratic')]
name_dict = {("Learning", "Linear"): "$M_{L,L}$",
             ("Learning", "Quadratic"): "$M_{L,Q}$",
             ("Adaptive", "Linear"): "$M_{A,L}$",
             ("Adaptive", "Quadratic"): "$M_{A,Q}$",
             ("Static", "Linear"): "$M_{S}$"}

#%%
# =============================================================================
# Extract values
# =============================================================================
G_Qup = {}
G_Qdown = {}
G_Div = {}
sc_co_dict = {}
for ite in range(1, num_iter+1):
    d1 = pickle.load(open(os.path.join(
        sc_result_path, "Sc_results_rcp_rcp26_iter_{}.p".format(ite)), "rb"))
    d2 = pickle.load(open(os.path.join(
        sc_result_path, "Sc_results_rcp_rcp85_iter_{}.p".format(ite)), "rb"))
    d1.update(d2)
    G_Qup[ite] = {}
    G_Qdown[ite] = {}
    G_Div[ite] = {}
    for label, results in tqdm(d1.items(), desc="Extract [{}]".format(ite)):
        (rcp,prec_q,temp_q,r,hy,abm,bt,at,it) = label
        new_label = (rcp,prec_q,temp_q,r,hy,abm,bt,at)
        rrr = results
        G_Qup[ite][new_label] = results["Qup_G_789_Y"]
        G_Qdown[ite][new_label] = results["Qdown_G_789_Y"]
        G_Div[ite][new_label] = results["div_Y"][["Roza", "Wapato", "Sunnyside"]].sum(axis=1)
sc_co_dict = {"G_Qup": G_Qup, "G_Qdown": G_Qdown, "G_Div": G_Div}
pickle.dump(sc_co_dict, open(os.path.join(
    sc_result_path, "Sc_coevolve_dict_{}iter.p".format(num_iter)), "wb"))
print("Done!")

#%%
item = "G_Qup"
def cal_mean(sc_co_dict, item):
    df_mean = pd.DataFrame()
    for ite in tqdm(range(1, num_iter+1)):
        df = pd.DataFrame(sc_co_dict[item][ite], columns=list(sc_co_dict[item][ite].keys())).T
        df_mean = pd.concat([df_mean, df])
    df_mean = df_mean.groupby(df_mean.index).mean()
    return df_mean

df_Qup = cal_mean(sc_co_dict, "G_Qup")
df_Qdown = cal_mean(sc_co_dict, "G_Qdown")
df_Div = cal_mean(sc_co_dict, "G_Div")

#%%
def cal_var(df):
    df_copy = df.copy()
    df_copy["Model"] = [(bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_copy.index]
    return df_copy.groupby(["Model"]).var()

def cal_cov(df1, df2):
    df1 = df1.copy()
    df2 = df2.copy()
    df1["Model"] = [(bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df1.index]
    df2["Model"] = [(bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df2.index]
    distict_model = list(set(df1["Model"]))

    df_cov = pd.DataFrame()
    for dm in distict_model:
        dff1 = df1[df1["Model"] == dm]
        dff2 = df2[df2["Model"] == dm]
        cov_list = []
        for i in range(80):
            dfff = pd.DataFrame()
            dfff["df1"] = dff1.iloc[:, i]
            dfff["df2"] = dff2.iloc[:, i]
            cov_list.append(dfff.cov().iloc[1,0])
        df_cov[dm] = cov_list
    df_cov.index = df1.columns[:-1]
    return df_cov.T

# This can be use to reply reviewer
df_Qup_var = cal_var(df_Qup).T
df_Qdown_var = cal_var(df_Qdown).T
df_Div_var = cal_var(df_Div).T
df_Qup_Div_cov = cal_cov(df_Qup, df_Div).T
start = 11
df_list = [df_Qdown_var, df_Qup_var, df_Div_var, df_Qup_Div_cov]
df_list = [i.iloc[start:,:] for i in df_list]
df_list_mean = [i[agtype_list].mean() for i in df_list]

l = int(df_list[0].shape[0]/2)
df_list_mean_a = [i[agtype_list].iloc[:l,:].mean() for i in df_list]
df_list_mean_b = [i[agtype_list].iloc[l:l*2,:].mean() for i in df_list]

fig, axes = plt.subplots(ncols=1,nrows=2, sharex=True)
axes = axes.flatten()
labels = ["$Q_{down}$", "$Q_{up}$","$Div$","$(Q_{up},Div)$"]
x = np.arange(len(labels))
width = 0.15
scale = 0.9
xloc = [-2*width, -1*width, 0, 1*width, 2*width]
cm = plt.cm.get_cmap('tab20').colors
colors = [cm[0], cm[2], cm[3], cm[4], cm[5]]
colors = [cm[14], cm[16], cm[17], cm[18], cm[19]]
ax = axes[0]
for i, ag in enumerate(agtype_list):
    y = [mean[ag] for mean in df_list_mean]
    rects1 = ax.bar(x + xloc[i], y, width*scale, label=name_dict[ag],
                    color=colors[i])
    #ax.bar_label(rects1, padding=3)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Averaged (co)variance\n$(m^3/sec)^2$")
ax.legend()

ax = axes[1]
for i, ag in enumerate(agtype_list):
    y = [(b-a)[ag] for a, b in zip(df_list_mean_a, df_list_mean_b)]
    rects1 = ax.bar(x + xloc[i], y, width*scale, label=name_dict[ag],
                    color=colors[i])
    #ax.bar_label(rects1, padding=3)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Difference of averaged\n(co)variance $(m^3/sec)^2$")
#ax.legend()
