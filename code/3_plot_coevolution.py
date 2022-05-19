import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy

path = r""
os.chdir(os.path.join(path, "code"))
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
# G_Qup = {}
# G_Qdown = {}
# G_Div = {}
# sc_co_dict = {}
# for ite in range(1, num_iter+1):
#     d1 = pickle.load(open(os.path.join(
#         sc_result_path, "Sc_results_rcp_rcp26_iter_{}.p".format(ite)), "rb"))
#     d2 = pickle.load(open(os.path.join(
#         sc_result_path, "Sc_results_rcp_rcp85_iter_{}.p".format(ite)), "rb"))
#     d1.update(d2)
#     G_Qup[ite] = {}
#     G_Qdown[ite] = {}
#     G_Div[ite] = {}
#     for label, results in tqdm(d1.items(), desc="Extract [{}]".format(ite)):
#         (rcp,prec_q,temp_q,r,hy,abm,bt,at,it) = label
#         new_label = (rcp,prec_q,temp_q,r,hy,abm,bt,at)
#         rrr = results
#         G_Qup[ite][new_label] = results["Qup_G_789_Y"]
#         G_Qdown[ite][new_label] = results["Qdown_G_789_Y"]
#         G_Div[ite][new_label] = results["div_Y"][["Roza", "Wapato", "Sunnyside"]].sum(axis=1)
# sc_co_dict = {"G_Qup": G_Qup, "G_Qdown": G_Qdown, "G_Div": G_Div}
# pickle.dump(sc_co_dict, open(os.path.join(
#     sc_result_path, "Sc_coevolve_dict_{}iter.p".format(num_iter)), "wb"))
# print("Done!")

#%%
num_iter = 1
sc_co_dict = pickle.load(open(os.path.join(
    sc_result_path, "Sc_coevolve_dict_{}iter.p".format(num_iter)), "rb"))
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
df_list_mean_diff = [i[agtype_list].iloc[l:l*2,:].mean() - i[agtype_list].iloc[:l,:].mean() for i in df_list]
#%%
fig, axes = plt.subplots(ncols=2,nrows=1, sharey=False, figsize=(6,3))
axes = axes.flatten()
plt.tight_layout(w_pad=0)
width = 1
scale = 0.8
cm = plt.cm.get_cmap('tab20').colors
colors = [cm[0], cm[2], cm[3], cm[4], cm[5]]
colors = [cm[14], cm[16], cm[17], cm[18], cm[19]]


name = ["$M_{S}$", "$M_{A,L}$", "$M_{A,Q}$", "$M_{L,L}$", "$M_{L,Q}$"]

item = df_list_mean
ax = axes[0]
y_down = item[0]
y_up = item[1]
y_d = item[2]
y_cov = item[3]
ax.bar(name, y_up, width*scale, label="$Var(Q_{up})$")
ax.bar(name, y_d, width*scale, label="$Var(Div)$", bottom=y_up)
ax.bar(name, -2*y_cov, width*scale, label="$-2Cov(Q_{up},Div)$")
ax.bar(name, y_down, width*scale, fill=False, edgecolor="k", label="$Var(Q_{down})$") # , yerr=0, capsize=10, color="k"
ax.axhline(0, color="k", lw=1, ls="--")

item = df_list_mean_diff
ax = axes[1]
y_down = item[0]
y_up = item[1]
y_d = item[2]
y_cov = item[3]
ax.bar(name, y_up, width*scale, label="$Var(Q_{up})$")
ax.bar(name, y_d, width*scale, label="$Var(Div)$")
ax.bar(name, -2*y_cov, width*scale, label="$-2Cov(Q_{up},Div)$", bottom=y_up)
ax.bar(name, y_down, width*scale, fill=False, edgecolor="k", label="$Var(Q_{down})$") # , yerr=0, capsize=10, color="k"
ax.axhline(0, color="k", lw=1, ls="--")

#%%
fig, axes = plt.subplots(ncols=2,nrows=1, figsize=(5,3), gridspec_kw={'width_ratios': [1, 2]})
axes = axes.flatten()
plt.tight_layout(w_pad=-1)
width = 1
scale = 0.6
cm = plt.cm.get_cmap('tab20').colors
colors = [cm[0], cm[2], cm[3], cm[4], cm[5]]
colors = [cm[14], cm[16], cm[17], cm[18], cm[19]]


name = ["$M_{S}$", "$M_{A,L}$", "$M_{A,Q}$", "$M_{L,L}$", "$M_{L,Q}$"]

item = df_list_mean
ax = axes[0]
y_down = item[0]
y_up = item[1]
y_d = item[2]
y_cov = item[3]
ax.bar(name, y_up, width*scale, label="$Var(Q_{up})$")
ax.bar(name, y_d, width*scale, label="$Var(Div)$", bottom=y_up)
ax.bar(name, -2*y_cov, width*scale, label="$-2Cov(Q_{up},Div)$")
ax.bar([0.2,1.2,2.2,3.2,4.2], y_down, width*scale*0.5, fill=True, edgecolor="k", label="$Var(Q_{down})$") 
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("$(m^3/sec)^2$")
ax.tick_params(axis='y', labelsize=9, labelrotation=90)
ax.tick_params(axis='x', labelrotation=40, labelsize=10)
ax.set_title("Averaged\n(co)variance", fontsize=10)
ax.legend(fontsize=9, labelspacing=0.1, handlelength=0.9, handletextpad=0.2, loc="upper center")
ax.set_ylim([-130,800])

ax = axes[1]
#ax = ax.twinx()
labels = ["$Var(Q_{down})$", "$Var(Q_{up})$","$Var(Div)$","$Cov(Q_{up},Div)$"]
x = np.arange(len(labels))
width = 0.15
scale = 0.9
xloc = [-2*width, -1*width, 0, 1*width, 2*width]
cm = plt.cm.get_cmap('tab20').colors
colors = [cm[0], cm[2], cm[3], cm[4], cm[5]]
colors = [cm[14], cm[16], cm[17], cm[18], cm[19]]
ax.axhline(0, color="k", lw=0.5, ls="--", zorder=-10)
for i, ag in enumerate(agtype_list):
    y = [(b-a)[ag] for a, b in zip(df_list_mean_a, df_list_mean_b)]
    rects1 = ax.bar(x + xloc[i], y, width*scale, label=name_dict[ag],
                    color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(axis='y', labelsize=9, labelrotation=90)
ax.tick_params(axis='x', labelrotation=15, labelsize=10)
ax.set_title("Difference of\naveraged (co)variance", fontsize=10)
ax.legend(fontsize=9)
# axes[1].tick_params(top=False, bottom=True,
#                     left=False, right=False)
# axes[1].set_yticklabels([])
# axes[1].tick_params(axis='x', labelrotation=20)
#axes[1].set_axis_off()
# y_down = item[0]
# y_up = item[1]
# y_d = item[2]
# y_cov = item[3]
# ax.bar(name, y_up, width*scale, label="$Var(Q_{up})$")
# ax.bar(name, y_d, width*scale, label="$Var(Div)$")
# ax.bar(name, -2*y_cov, width*scale, label="$-2Cov(Q_{up},Div)$", bottom=y_up)
# ax.bar(name, y_down, width*scale, fill=False, edgecolor="k", label="$Var(Q_{down})$") # , yerr=0, capsize=10, color="k"
# ax.axhline(0, color="k", lw=1, ls="--")

#%% Reviewer ask

fig, axes = plt.subplots(ncols=3,nrows=1, sharey=True, figsize=(6,3))
axes = axes.flatten()
plt.tight_layout(w_pad=0)
width = 1
scale = 0.8
cm = plt.cm.get_cmap('tab20').colors
colors = [cm[0], cm[2], cm[3], cm[4], cm[5]]
colors = [cm[14], cm[16], cm[17], cm[18], cm[19]]


name = ["$M_{S}$", "$M_{A,L}$", "$M_{A,Q}$", "$M_{L,L}$", "$M_{L,Q}$"]
for i, item in enumerate([df_list_mean, df_list_mean_a, df_list_mean_b]):
    ax = axes[i]
    y_down = item[0]
    y_up = item[1]
    y_d = item[2]
    y_cov = item[3]
    ax.bar(name, y_up, width*scale, label="$Var(Q_{up})$")
    ax.bar(name, y_d, width*scale, label="$Var(Div)$", bottom=y_up)
    ax.bar(name, -2*y_cov, width*scale, label="$-2Cov(Q_{up},Div)$")
    ax.bar(name, y_down, width*scale, fill=False, edgecolor="k", label="$Var(Q_{down})$") # , yerr=0, capsize=10, color="k"
    ax.axhline(0, color="k", lw=1, ls="--")

#ax.set_ylabel("Averaged (co)variance\n$(m^3/sec)^2$")
#ax.legend()

#%% MR1 plot
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
labels = ["$Var(Q_{down})$", "$Var(Q_{up})$","$Var(Div)$","$Cov(Q_{up},Div)$"]
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

#%%
df_Qup_var.T.plot()
df_Qdown_var.T.plot()
df_Div_var.T.plot()
df_Qup_Div_cov.T.plot()

# To better visualize => use tw=10 to smooth the trend
def cal_mean_over_df(df, tw=None):
    """df, tw=None"""
    arr = df.T.values
    df_q = pd.DataFrame()
    for y in tqdm(range(1, arr.shape[0]+1), desc="mean"):
        if tw is None:
            df_q[y] = np.mean(arr[:y, :], axis=0)
        else:
            df_q[y] = np.mean(arr[max(0,y-tw):y, :], axis=0)
    df_q.columns = df.columns
    df_q.index = df.index
    return df_q

tw=None
cal_mean_over_df(df_Qup_var, tw).T.plot()
cal_mean_over_df(df_Qdown_var, tw).T.plot()
cal_mean_over_df(df_Div_var, tw).T.plot()
cal_mean_over_df(df_Qup_Div_cov, tw).T.plot()