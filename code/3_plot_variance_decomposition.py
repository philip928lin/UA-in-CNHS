import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
path = r""
os.chdir(os.path.join(path, "code"))
from decomposition_funcs import get_EV, get_VE, get_EVE, get_EV2, get_VE_EV, get_VE_VE

sc_result_path = r""

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

def cal_q_over_df(df, q=0.5, tw=None):
    """df, tw=None, q=0.5"""
    arr = df.T.values
    df_q = pd.DataFrame()
    for y in tqdm(range(1, arr.shape[0]+1), desc="q={}".format(q)):
        if tw is None:
            df_q[y] = np.quantile(arr[:y, :], q, axis=0)
        else:
            df_q[y] = np.quantile(arr[max(0,y-tw):y, :], q, axis=0)
    df_q.columns = df.columns
    df_q.index = df.index
    return df_q
#%%
# =============================================================================
# Extract values
# =============================================================================
sc_dict = {}
D = {}  # Annual total diversion [cms]
R = {}
Q = {}  # 789 streamflow [cms]
S = {}  # Annual shortage [m3/86400]
for ite in range(1, num_iter+1):
    d1 = pickle.load(open(os.path.join(
        sc_result_path, "Sc_results_rcp_rcp26_iter_{}.p".format(ite)), "rb"))
    d2 = pickle.load(open(os.path.join(
        sc_result_path, "Sc_results_rcp_rcp85_iter_{}.p".format(ite)), "rb"))
    d1.update(d2)
    D[ite] = {}
    R[ite] = {}
    Q[ite] = {}
    S[ite] = {}
    for label, results in tqdm(d1.items(), desc="Extract [{}]".format(ite)):
        (rcp,prec_q,temp_q,r,hy,abm,bt,at,it) = label
        new_label = (rcp,prec_q,temp_q,r,hy,abm,bt,at)
        D[ite][new_label] = results["div_Y"].sum(axis=1)
        R[ite][new_label] = results["div_req_Y"].sum(axis=1)
        Q[ite][new_label] = results["sim789"]
        S[ite][new_label] = results["shortage_Y"].sum(axis=1)
sc_dict = {"D": D,
           "R": R,
           "Q": Q,
           "S": S}
pickle.dump(sc_dict, open(os.path.join(
    sc_result_path, "Sc_dict_{}iter.p".format(num_iter)), "wb"))
print("Done!")
#%%
# =============================================================================
# Calculate indicators (rcp)
# =============================================================================
#sc_dict = pickle.load(open(os.path.join(sc_result_path, "Sc_dict_{}iter.p".format(num_iter)), "rb"))
indicator = {}
for ite in range(1, num_iter+1):
    df_D = pd.DataFrame(sc_dict["D"][ite], columns=list(sc_dict["D"][ite].keys())).T
    df_R = pd.DataFrame(sc_dict["R"][ite], columns=list(sc_dict["R"][ite].keys())).T
    df_Q = pd.DataFrame(sc_dict["Q"][ite], columns=list(sc_dict["Q"][ite].keys())).T
    df_S = pd.DataFrame(sc_dict["S"][ite], columns=list(sc_dict["S"][ite].keys())).T
    # Count shortage event over 20 realizations
    df_S[df_S>0] = 1
    df_S["Inter"] = [(rcp,prec_q,temp_q,hy,abm,bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_S.index]
    df_S = df_S.groupby(["Inter"]).sum()

    df_Dm = cal_q_over_df(df_D)
    df_Rm = cal_q_over_df(df_R)
    df_Qm = cal_q_over_df(df_Q)
    df_Sm = cal_q_over_df(df_S)

    indicator[ite] = {"Dm": df_Dm,
                      "Rm": df_Rm,
                      "Qm": df_Qm,
                      "Sm": df_Sm}
pickle.dump(indicator, open(os.path.join(
    sc_result_path, "Sc_indicator_{}iter.p".format(num_iter)), "wb"))
print("Done!")

#%%

def vd(indicator, item, save=True):
    vd = {}
    df_cli = pd.DataFrame()
    df_int = pd.DataFrame()
    df_equ = pd.DataFrame()
    df_equ_abm = pd.DataFrame()
    df_equ_hy = pd.DataFrame()
    for ite in tqdm(range(1, num_iter+1), desc="vd_{}".format(item)):
        # Didn't use copy(). This will override the data in indicator but it is fine.
        df = indicator[ite][item]
        df.columns = list(np.arange(1, 81, 1).astype(int))
        df_cli = pd.concat([df_cli, get_EV2(df)])
        df_int = pd.concat([df_int, get_EVE(df)])
        df_equ = pd.concat([df_equ, get_VE(df)])
        df_equ_abm = pd.concat([df_equ_abm, get_VE_EV(df)])
        df_equ_hy = pd.concat([df_equ_hy, get_VE_VE(df)])
    vd = {"cli": df_cli, "int": df_int, "equ": df_equ, "equ_abm": df_equ_abm, "equ_hy": df_equ_hy}
    if save:
        pickle.dump(vd, open(os.path.join(
            sc_result_path, "Sc_vd_{}_{}iter.p".format(item, num_iter)), "wb"))
    return vd

Qm_vd = vd(indicator, "Qm")
Dm_vd = vd(indicator, "Dm")
Rm_vd = vd(indicator, "Rm")

Sm_mean = pd.DataFrame()
for ite in range(1, num_iter+1):
    df = indicator[ite]["Sm"]
    Sm_mean = pd.concat([Sm_mean, indicator[ite]["Sm"]])
Sm_mean = Sm_mean.groupby(Sm_mean.index).mean()
Sm_mean.columns = list(np.arange(1, 81, 1).astype(int))
#%%
# =============================================================================
# Plot Dm + Sm
# =============================================================================
vd = deepcopy(Qm_vd)
start = 11
rcp_label = ""
ylim=[0,200]

cli_mean = vd["cli"].groupby(vd["cli"].index).mean()
int_mean = vd["int"].groupby(vd["int"].index).mean()
equ_mean = vd["equ"].groupby(vd["equ"].index).mean()
equ_abm_mean = vd["equ_abm"].groupby(vd["equ_abm"].index).mean()
equ_hy_mean = vd["equ_hy"].groupby(vd["equ_hy"].index).mean()

# cli_std = vd["cli"].groupby(vd["cli"].index).std()
# int_std = (vd["cli"]+vd["int"]).groupby(vd["int"].index).std()
# equ_std = (vd["cli"]+vd["int"]+vd["equ"]).groupby(vd["equ"].index).std()

total = cli_mean + int_mean + equ_mean
cli_p = cli_mean / total * 100
int_p = int_mean / total * 100
equ_p = equ_mean / total * 100
equ_abm_p = equ_abm_mean / total * 100
equ_hy_p = equ_hy_mean / total * 100

fig, axes = plt.subplots(nrows=3, ncols=5, sharey='row', sharex=False,
                         figsize=(8,7.5), gridspec_kw={
                             'width_ratios': [1, 1, 1, 1, 1],
                             'height_ratios': [2, 1, 1]})
axes = axes.flatten()

cm = plt.cm.get_cmap('tab20c').colors
# 0: royalblue, 5: orange, 10: lightgreen
#colors = [cm[0], cm[5], cm[10]
colors = [cm[0], cm[5], cm[10], cm[9]]
plt.subplots_adjust(wspace=0.1, hspace=0.05)
axes = axes.flatten()

x = list(np.arange(start, 81, 1).astype(int))
for i, agtype in enumerate(agtype_list):
    ax = axes[i]
    # ax.stackplot(x,
    #              cli_mean.loc[[agtype], x],
    #              int_mean.loc[[agtype], x],
    #              equ_mean.loc[[agtype], x],
    #              labels=[r'Climate change scenario {}'.format(rcp_label),
    #                      'Internal climate variability', 'Equifinality'],
    #              colors=colors)
    ax.stackplot(x,
                 cli_mean.loc[[agtype], x],
                 int_mean.loc[[agtype], x],
                 equ_hy_mean.loc[[agtype], x],
                 equ_abm_mean.loc[[agtype], x],
                 labels=[r'Climate change scenario {}'.format(rcp_label),
                         'Internal climate variability',
                         '$Config_{HydroEMR}$',
                         '$Config_{ABMEMR}$'],
                 colors=colors)
    ax.set_xlim([start,80])
    # ax.set_xticks([11,31,51,71])
    # ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation=23)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Model uncertainty\nvariance of $Q_M$\n$(m^3/sec)^2$",
            fontsize = 14)
    if agtype == ('Learning', 'Quadratic'):
        handles, labels = ax.get_legend_handles_labels()

    ax.set_title(name_dict[agtype], fontsize=14)
    ax.set_ylim(ylim)

    # Percentage
    ax = axes[i+5]
    # ax.stackplot(x,
    #              cli_p.loc[[agtype], x],
    #              int_p.loc[[agtype], x],
    #              equ_p.loc[[agtype], x],
    #              labels=[r'Climate change scenario {}'.format(rcp_label),
    #                      'Internal climate variability', 'Equifinality'],
    #              colors=colors)
    ax.stackplot(x,
                 cli_p.loc[[agtype], x],
                 int_p.loc[[agtype], x],
                 equ_hy_p.loc[[agtype], x],
                 equ_abm_p.loc[[agtype], x],
                 labels=[r'Climate change scenario {}'.format(rcp_label),
                         'Internal climate variability',
                         '$Config_{HydroEMR}$',
                         '$Config_{ABMEMR}$'],
                 colors=colors)
    ax.set_xlim([start,80])
    # ax.set_xticks([11,31,51,71])
    # ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation = 23)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Fractional\ncontribution\n(%)",
            fontsize = 14)
    ax.set_ylim([0,100])


    ax = axes[i+10]
    bt, at = agtype
    mask = [True if (bt0==bt and rcp0==rcp and at0==at) else False \
        for (rcp0,prec_q,temp_q,hy,abm,bt0,at0) in Sm_mean.index]
    df_Sm_ag = Sm_mean.loc[mask,start:]
    df_Sm_ag.columns = list(np.arange(-0, 80-start+1, 1).astype(int))
    df_Sm_ag.plot.box(legend=False, ylim=[0,20], ax=ax,
                           flierprops=dict(markersize=0.2, linewidth=0.1),
                           boxprops=dict(linestyle='-', linewidth=0.1),
                           medianprops=dict(linestyle='-', linewidth=0.1),
                           whiskerprops=dict(linestyle='-', linewidth=0.1),
                           capprops=dict(linestyle='-', linewidth=0.1))

    ax.plot(df_Sm_ag.mean())
    ax.axhline(10, color="grey", lw=0.5)
    ax.set_xlim([0,80-start])
    ax.set_xticks([0,20,40,60])
    ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation=30)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Shortage\nfrequecy, $S_M$\n(max=20)",
            fontsize = 14)
fig.add_subplot(111, frameon=False)
fig.align_ylabels(axes)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False,
                left=False, right=False)
plt.grid(False)
le = plt.legend(handles[::-1], labels[::-1], title="Uncertainty Sources",
                loc="upper right", framealpha=1, fontsize=12, ncol=2)
le.get_title().set_fontsize('12')
plt.xlabel("\nYear", fontsize=14)
#%%
vd = deepcopy(Qm_vd)
cli_mean = vd["cli"].groupby(vd["cli"].index).mean()
int_mean = vd["int"].groupby(vd["int"].index).mean()
equ_mean = vd["equ"].groupby(vd["equ"].index).mean()
total_Qm = cli_mean + int_mean + equ_mean

vd = deepcopy(Dm_vd)
start = 11
rcp_label = ""
ylim=[0,30]

cli_mean = vd["cli"].groupby(vd["cli"].index).mean()
int_mean = vd["int"].groupby(vd["int"].index).mean()
equ_mean = vd["equ"].groupby(vd["equ"].index).mean()
equ_abm_mean = vd["equ_abm"].groupby(vd["equ_abm"].index).mean()
equ_hy_mean = vd["equ_hy"].groupby(vd["equ_hy"].index).mean()

# cli_std = vd["cli"].groupby(vd["cli"].index).std()
# int_std = (vd["cli"]+vd["int"]).groupby(vd["int"].index).std()
# equ_std = (vd["cli"]+vd["int"]+vd["equ"]).groupby(vd["equ"].index).std()

total = cli_mean + int_mean + equ_mean
cli_p = cli_mean / total * 100
int_p = int_mean / total * 100
equ_p = equ_mean / total * 100
equ_abm_p = equ_abm_mean / total * 100
equ_hy_p = equ_hy_mean / total * 100

fig, axes = plt.subplots(nrows=3, ncols=5, sharey='row', sharex=False,
                         figsize=(8,7.5), gridspec_kw={
                             'width_ratios': [1, 1, 1, 1, 1],
                             'height_ratios': [2, 1, 1]})
axes = axes.flatten()

cm = plt.cm.get_cmap('tab20c').colors
# 0: royalblue, 5: orange, 10: lightgreen
#colors = [cm[0], cm[5], cm[10]
colors = [cm[0], cm[5], cm[10], cm[9]]
plt.subplots_adjust(wspace=0.1, hspace=0.05)
axes = axes.flatten()

x = list(np.arange(start, 81, 1).astype(int))
for i, agtype in enumerate(agtype_list):
    ax = axes[i]
    # ax.stackplot(x,
    #              cli_mean.loc[[agtype], x],
    #              int_mean.loc[[agtype], x],
    #              equ_mean.loc[[agtype], x],
    #              labels=[r'Climate change scenario {}'.format(rcp_label),
    #                      'Internal climate variability', 'Equifinality'],
    #              colors=colors)
    ax.stackplot(x,
                 cli_mean.loc[[agtype], x],
                 int_mean.loc[[agtype], x],
                 equ_hy_mean.loc[[agtype], x],
                 equ_abm_mean.loc[[agtype], x],
                 labels=[r'Climate change scenario {}'.format(rcp_label),
                         'Internal climate variability',
                         '$Config_{HydroEMR}$',
                         '$Config_{ABMEMR}$'],
                 colors=colors)
    ax.set_xlim([start,80])
    # ax.set_xticks([11,31,51,71])
    # ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation=23)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Model uncertainty\nvariance of $D_M$\n$(m^3/sec)^2$",
            fontsize = 14)
    if agtype == ('Learning', 'Quadratic'):
        handles, labels = ax.get_legend_handles_labels()

    ax.set_title(name_dict[agtype], fontsize=14)
    ax.set_ylim(ylim)

    # Percentage
    ax = axes[i+5]
    # ax.stackplot(x,
    #              cli_p.loc[[agtype], x],
    #              int_p.loc[[agtype], x],
    #              equ_p.loc[[agtype], x],
    #              labels=[r'Climate change scenario {}'.format(rcp_label),
    #                      'Internal climate variability', 'Equifinality'],
    #              colors=colors)
    ax.stackplot(x,
                 cli_p.loc[[agtype], x],
                 int_p.loc[[agtype], x],
                 equ_hy_p.loc[[agtype], x],
                 equ_abm_p.loc[[agtype], x],
                 labels=[r'Climate change scenario {}'.format(rcp_label),
                         'Internal climate variability',
                         '$Config_{HydroEMR}$',
                         '$Config_{ABMEMR}$'],
                 colors=colors)
    ax.set_xlim([start,80])
    # ax.set_xticks([11,31,51,71])
    # ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation = 23)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Fractional\ncontribution\n(%)",
            fontsize = 14)
    ax.set_ylim([0,100])


    ax = axes[i+10]
    ax.fill_between(x, total_Qm.loc[[agtype], x].values.flatten(), color="grey")
    ax.set_xticks([11,31,51,71])
    ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation=30)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Total variance\nof $Q_M$",
            fontsize = 14)

fig.add_subplot(111, frameon=False)
fig.align_ylabels(axes)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False,
                left=False, right=False)
plt.grid(False)
le = plt.legend(handles[::-1], labels[::-1], title="Uncertainty Sources",
                loc="upper left", framealpha=1, fontsize=12, ncol=1)
le.get_title().set_fontsize('12')
plt.xlabel("\nYear", fontsize=14)


#%%
# For proposal
vd = deepcopy(Qm_vd)
cli_mean_Qm = vd["cli"].groupby(vd["cli"].index).mean()
int_mean_Qm = vd["int"].groupby(vd["int"].index).mean()
equ_mean_Qm = vd["equ"].groupby(vd["equ"].index).mean()
equ_abm_mean_Qm = vd["equ_abm"].groupby(vd["equ_abm"].index).mean()
equ_hy_mean_Qm = vd["equ_hy"].groupby(vd["equ_hy"].index).mean()
total_Qm = cli_mean_Qm + int_mean_Qm + equ_mean_Qm

vd = deepcopy(Dm_vd)
start = 11
rcp_label = ""

cli_mean_Dm = vd["cli"].groupby(vd["cli"].index).mean()
int_mean_Dm = vd["int"].groupby(vd["int"].index).mean()
equ_mean_Dm = vd["equ"].groupby(vd["equ"].index).mean()
equ_abm_mean_Dm = vd["equ_abm"].groupby(vd["equ_abm"].index).mean()
equ_hy_mean_Dm = vd["equ_hy"].groupby(vd["equ_hy"].index).mean()
total_Dm = cli_mean_Dm + int_mean_Dm + equ_mean_Dm


fig, axes = plt.subplots(nrows=3, ncols=5, sharey='row', sharex=False,
                         figsize=(8,7.5), gridspec_kw={
                             'width_ratios': [1, 1, 1, 1, 1],
                             'height_ratios': [1.5, 1.5, 1]})
axes = axes.flatten()

cm = plt.cm.get_cmap('tab20c').colors
# 0: royalblue, 5: orange, 10: lightgreen
# colors = [cm[0], cm[5], cm[10]]
colors = [cm[0], cm[5], cm[10], cm[9]]
plt.subplots_adjust(wspace=0.1, hspace=0.05)
axes = axes.flatten()

x = list(np.arange(start, 81, 1).astype(int))
for i, agtype in enumerate(agtype_list):
    ax = axes[i]
    ax.stackplot(x,
                 cli_mean_Qm.loc[[agtype], x],
                 int_mean_Qm.loc[[agtype], x],
                 equ_hy_mean_Qm.loc[[agtype], x],
                 equ_abm_mean_Qm.loc[[agtype], x],
                 labels=[r'Climate change scenario {}'.format(rcp_label),
                         'Internal climate variability',
                         '$Equifinality_{HydroEMR}$',
                         '$Equifinality_{ABMEMR}$'],
                 colors=colors)

    ax.set_xlim([start,80])
    # ax.set_xticks([11,31,51,71])
    # ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation=23)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Model uncertainty\nvariance of $Q_M$\n$(m^3/sec)^2$",
            fontsize = 14)
    if agtype == ('Learning', 'Quadratic'):
        handles, labels = ax.get_legend_handles_labels()

    ax.set_title(name_dict[agtype], fontsize=14)
    ylim=[0,150]
    ax.set_ylim(ylim)

    # Dm
    ax = axes[i+5]
    ax.stackplot(x,
                 cli_mean_Dm.loc[[agtype], x],
                 int_mean_Dm.loc[[agtype], x],
                 equ_hy_mean_Dm.loc[[agtype], x],
                 equ_abm_mean_Dm.loc[[agtype], x],
                 labels=[r'Climate change scenario {}'.format(rcp_label),
                         'Internal climate variability',
                         '$Equifinality_{HydroEMR}$',
                         '$Equifinality_{ABMEMR}$'],
                 colors=colors)

    ax.set_xlim([start,80])
    # ax.set_xticks([11,31,51,71])
    # ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation=23)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Model uncertainty\nvariance of $D_M$\n$(m^3/sec)^2$",
            fontsize = 14)
    if agtype == ('Learning', 'Quadratic'):
        handles, labels = ax.get_legend_handles_labels()

    #ax.set_title(name_dict[agtype], fontsize=14)
    ylim=[0,30]
    ax.set_ylim(ylim)


    ax = axes[i+10]
    bt, at = agtype
    mask = [True if (bt0==bt and rcp0==rcp and at0==at) else False \
        for (rcp0,prec_q,temp_q,hy,abm,bt0,at0) in Sm_mean.index]
    df_Sm_ag = Sm_mean.loc[mask,start:]
    df_Sm_ag.columns = list(np.arange(-0, 80-start+1, 1).astype(int))
    df_Sm_ag.plot.box(legend=False, ylim=[0,20], ax=ax,
                           flierprops=dict(markersize=0.2, linewidth=0.1),
                           boxprops=dict(linestyle='-', linewidth=0.1),
                           medianprops=dict(linestyle='-', linewidth=0.1),
                           whiskerprops=dict(linestyle='-', linewidth=0.1),
                           capprops=dict(linestyle='-', linewidth=0.1))

    ax.plot(df_Sm_ag.mean())
    ax.axhline(10, color="grey", lw=0.5)
    ax.set_xlim([0,80-start])
    ax.set_xticks([0,20,40,60])
    ax.set_xticklabels(['2030s','2050s','2070s','2090s'], rotation=30)
    if agtype == ('Static', 'Linear'):
        ax.set_ylabel(
            "Shortage\nfrequecy, $S_M$\n(max=20)",
            fontsize = 14)

fig.add_subplot(111, frameon=False)
fig.align_ylabels(axes)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False,
                left=False, right=False)
plt.grid(False)
le = plt.legend(handles[::-1], labels[::-1], title="Uncertainty Sources",
                loc="center left", framealpha=1, fontsize=10, ncol=1)
le.get_title().set_fontsize('12')
plt.xlabel("\nYear", fontsize=14)









