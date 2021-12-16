import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import HydroCNHS
import HydroCNHS.calibration as cali

pc = ""
path = r"".format(pc)
bound_path = os.path.join(path, r"")
csv_path = os.path.join(path, "")
input_path = os.path.join(path, r"")
wd = r"".format(pc)
module_path = os.path.join(path, r"")
emr_path = os.path.join(path, r"")
with open(os.path.join(input_path, "YRB_cali_inputs.pickle"),
          "rb") as file:
    (temp, prec, pet, obv_M, obv_Y) = pickle.load(file)
mask = [True if i.month in [7,8,9] else False for i in obv_M.index]
obv789 = obv_M.loc[mask, "G"].resample("YS").mean()
obv_Y["G789"] = obv789

assigned_Q_res_inflow = pd.read_csv(
    os.path.join(csv_path, "Inflow_D_R1R2R3_cms.csv"),
    parse_dates=True, index_col=[0])["1959/1/1":]
assigned_Q = {}
for sub in ["S1","S2","S3"]:
    assigned_Q[sub] = np.array(assigned_Q_res_inflow["R"+sub[1]])

#%% Plot the calibrated quadratic functions.
ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
bt_list = ["Adaptive", "Learning"]
at_list = ["Quadratic"]
num_hy_emr = 2
num_abm_emr = 4
par_dict = {}
for bt in bt_list:
    for at in at_list:
        for hy in range(1, num_hy_emr+1):
            for abm in range(1, num_abm_emr+1):
                ty = "{}-{}".format(bt, at)
                emr_name = "{}-EMR{}-{}.yaml".format(ty, hy, abm)
                model_dict = HydroCNHS.load_model(
                    os.path.join(emr_path, emr_name), print_summary=False)
                abm_par = HydroCNHS.write_model_to_df(model_dict)[0][2]
                abm_par.columns = [i[0] for i in abm_par.columns]
                par_dict[(bt,at,hy,abm)] = abm_par


x = np.arange(0, 2, 0.01)
def quadratic(x, a,b,c):
    return a*x**2 + b*x + c

fig, axes = plt.subplots(ncols=5,nrows=2, sharex=True)
for i, bt in enumerate(bt_list):
    for at in at_list:
        for hy in range(1, num_hy_emr+1):
            for abm in range(1, num_abm_emr+1):
                for j, ag in enumerate(ag_list):
                    ax = axes[i, j]
                    par = par_dict[(bt,at,hy,abm)]
                    y = quadratic(x, par.loc["a",ag], par.loc["b",ag], par.loc["c",ag])
                    if hy == 1:
                        c = "C0"
                    else:
                        c = "C1"
                    ax.plot(x,y, color=c)
                    if i == 0:
                        ax.set_title(ag)
                        if j == 4 and abm==1:
                            ax.plot(x,y,label="HydroEMR{}".format(hy), color=c)
                            ax.legend()
                    if j == 0:
                        ax.set_ylabel(bt)
plt.tight_layout()
#%%    
r"""
ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
bt_list = ["Static", "Adaptive", "Learning"]
at_list = ["Linear", "Quadratic"]
num_hy_emr = 2
num_abm_emr = 4
gauges = ["C1", "C2", "G"]
mask2 = [True if i.month in [7,8,9] else False \
         for i in pd.date_range("1959-1-1","2013-12-31",freq="M")]

def cal_batch_indicator(period, target, df_obv, df_sim):
        df_obv = df_obv[period[0]:period[1]]
        df_sim = df_sim[period[0]:period[1]]
        Indicator = HydroCNHS.Indicator()
        df = pd.DataFrame()
        for item in target:
            df_i = Indicator.cal_indicator_df(df_obv[item], df_sim[item],
                                              index_name=item)
            df = pd.concat([df, df_i], axis=0)
        df_mean = pd.DataFrame(df.mean(axis=0), columns=["Mean"]).T
        df = pd.concat([df, df_mean], axis=0)
        return df    

emr_results = {}
for bt in bt_list:
    for at in at_list:
        if bt == "Static" and at == "Quadratic":
            continue
        for hy in range(1, num_hy_emr+1):
            for abm in range(1, num_abm_emr+1):
                ty = "{}-{}".format(bt, at)
                emr_name = "{}-EMR{}-{}.yaml".format(ty, hy, abm)
                model_dict = HydroCNHS.load_model(
                    os.path.join(emr_path, emr_name), print_summary=False)
                model_dict["Path"]["WD"] = wd
                model_dict["Path"]["Modules"] = module_path
                model_dict["ABM"]["Inputs"]["Modules"] = ['YRB_ABM_Philip.py']
                model_dict["ABM"]["Inputs"]["Database"] = \
                    os.path.join(csv_path, "Database_1959_2013.csv")
                model = HydroCNHS.Model(model_dict)
                
                if bt == "Static":
                    num_iter = 1
                else:
                    num_iter = 10                                               # check
                df_cali_Q_M = pd.DataFrame()
                df_vali_Q_M = pd.DataFrame()
                df_cali_Q_Y = pd.DataFrame()
                df_vali_Q_Y = pd.DataFrame()
                div_KGE_temp = []

                annual_ts_dict = {}
                monthly_ts_dict = {}
                for it in range(num_iter):
                    Q = model.run(temp, prec, pet, assigned_Q=assigned_Q)

                    # Get simulation data
                    cali_target = ["C1", "C2", "G"]
                    cali_period = ("1960-1-1", "1999-12-31")
                    vali_period = ("2000-1-1", "2013-12-31")

                    ag_list = ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']
                    shortage_M = []
                    div_D = []
                    for ag in ag_list:
                        shortage_M.append(model.data_collector.get_field(ag)["Shortage_M"])
                        div_D.append(model.data_collector.get_field(ag)["Div"])
                    df_div = pd.DataFrame(div_D, index=ag_list, columns=model.pd_date_index).T
                    df = pd.DataFrame(
                        shortage_M, index=ag_list,
                        columns=pd.date_range("1959-1-1", "2013-12-31", freq="MS")).T
                    df = df[cali_period[0]: cali_period[1]]
                    mean_Y_shortage = df.groupby(df.index.year).mean().mean().sum()

                    sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[cali_target]
                    sim_Q_D = pd.concat([sim_Q_D, df_div], axis=1)
                    sim_Q_M = sim_Q_D[cali_target + ag_list].resample("MS").mean()
                    sim_Q_Y = sim_Q_D[cali_target + ag_list].resample("YS").mean()
                    sim789 = sim_Q_M.loc[mask2, "G"].resample("YS").mean()
                    sim_Q_Y["G789"] = sim789
                    annual_ts_dict[it] = sim_Q_Y
                    monthly_ts_dict[it] = sim_Q_M
                    
                    df_cali_Q_M_t = cal_batch_indicator(
                        cali_period, cali_target + ag_list, obv_M, sim_Q_M)
                    df_cali_Q_Y_t = cal_batch_indicator(
                        cali_period, cali_target + ag_list, obv_Y, sim_Q_Y)

                    df_vali_Q_M_t = cal_batch_indicator(
                        vali_period, cali_target + ag_list, obv_M, sim_Q_M)
                    df_vali_Q_Y_t = cal_batch_indicator(
                        vali_period, cali_target + ag_list, obv_Y, sim_Q_Y)

                    df_cali_Q_M = pd.concat([df_cali_Q_M, df_cali_Q_M_t])
                    df_vali_Q_M = pd.concat([df_vali_Q_M, df_vali_Q_M_t])
                    df_cali_Q_Y = pd.concat([df_cali_Q_Y, df_cali_Q_Y_t])
                    df_vali_Q_Y = pd.concat([df_vali_Q_Y, df_vali_Q_Y_t])

                    # Check early stop for stochastic modeling (break for loop)
                    div_KGE_temp.append(
                        df_cali_Q_Y.groupby(df_cali_Q_Y.index).mean().loc[ag_list,"KGE"].sum())

                df_cali_Q_M = df_cali_Q_M.groupby(df_cali_Q_M.index).mean()
                df_vali_Q_M = df_vali_Q_M.groupby(df_vali_Q_M.index).mean()
                df_cali_Q_Y = df_cali_Q_Y.groupby(df_cali_Q_Y.index).mean()
                df_vali_Q_Y = df_vali_Q_Y.groupby(df_vali_Q_Y.index).mean()

                flow_KGE = df_cali_Q_M.loc[cali_target, "KGE"].sum()
                div_KGE = df_cali_Q_Y.loc[ag_list, "KGE"].sum()
                # We weight mean_Y_shortage to 10. Should not have shortage.
                fitness = (flow_KGE + div_KGE)/8 + (1 - mean_Y_shortage)
                
                # Record results
                emr_results[(bt, at, hy, abm, "EMR{}-{}".format(hy, abm))] = \
                    {"Cali": [fitness, df_cali_Q_M, df_vali_Q_M, df_cali_Q_Y, df_vali_Q_Y],
                     "TS_Y": annual_ts_dict,
                     "TS_M": monthly_ts_dict}
                    
# Calulate average TS
for i, v in emr_results.items():
    df = pd.concat(list(v["TS_M"].values()), axis=1).T
    emr_results[i]["TS_M_avg"] = df.groupby(df.index).mean().T
    df = pd.concat(list(v["TS_Y"].values()), axis=1).T
    emr_results[i]["TS_Y_avg"] = df.groupby(df.index).mean().T
    df = pd.concat(list(v["TS_M"].values()), axis=1).T
    emr_results[i]["TS_M_std"] = df.groupby(df.index).std().T
    df = pd.concat(list(v["TS_Y"].values()), axis=1).T
    emr_results[i]["TS_Y_std"] = df.groupby(df.index).std().T

# Overall indicator
target = ["C1", "C2", "G"]
ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
period_ts = ("1960-1-1", "2013-12-31")
KGE_indicator_M = pd.DataFrame()
KGE_indicator_Y = pd.DataFrame()
for i, v in emr_results.items():
    df_indicator_M = pd.DataFrame()
    df_indicator_Y = pd.DataFrame()
    for j, df in v["TS_M"].items():
        df_indicator_M_t = cal_batch_indicator(
            period_ts, target + ag_list, obv_M, df)
        df_indicator_M = pd.concat([df_indicator_M, df_indicator_M_t])
    for j, df in v["TS_Y"].items():
        df_indicator_Y_t = cal_batch_indicator(
            period_ts, target + ag_list, obv_Y, df)
        df_indicator_Y = pd.concat([df_indicator_Y, df_indicator_Y_t])
        
    df_indicator_M = df_indicator_M.groupby(df_indicator_M.index).mean()
    df_indicator_Y = df_indicator_Y.groupby(df_indicator_Y.index).mean()
    
    emr_results[i]["TS_M_indicator"] = df_indicator_M
    emr_results[i]["TS_Y_indicator"] = df_indicator_Y
    KGE_indicator_M[i] = df_indicator_M["KGE"]
    KGE_indicator_Y[i] = df_indicator_Y["KGE"]
    
pickle.dump(emr_results, open(os.path.join(emr_path, "EMR_sim_results_10.p"), "wb"))  
KGE_indicator_Y = KGE_indicator_Y.round(3)    
KGE_indicator_Y.to_csv(os.path.join(emr_path, "ABMEMR_KGE_Y.csv"))  
"""
#%%
emr_results = pickle.load( open(os.path.join(emr_path, "EMR_sim_results_10.p"), "rb")) 

#%%
ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
bt_list = ["Static", "Adaptive", "Learning"]
at_list = ["Linear", "Quadratic"]
num_hy_emr = 2
num_abm_emr = 4

name_dict = {("Learning", "Linear"): "$M_{L,L}$",
             ("Learning", "Quadratic"): "$M_{L,Q}$",
             ("Adaptive", "Linear"): "$M_{A,L}$",
             ("Adaptive", "Quadratic"): "$M_{A,Q}$",
             ("Static", "Linear"): "$M_{S}$"}


fig, axes = plt.subplots(nrows=5, ncols=5, sharex=True, figsize=(6.4,4.8))
ls_list = ["dotted", "dashed", "dashdot", "solid"]
    
ag_y_limits = [ [3, 20], [1, 6], [3, 20], [10, 30], [9.5, 25] ]      
            
for col, ag in enumerate(ag_list):
    row = 0
    for bt in bt_list:
        for at in at_list:
            if bt == "Static" and at == "Quadratic":
                continue
            row += 1
            for hy in range(1, num_hy_emr+1):
                for abm in range(1, num_abm_emr+1):
                    ax = axes[row-1, col]
                    label = (bt, at, hy, abm, "EMR{}-{}".format(hy, abm))
                    div_Y = emr_results[label]["TS_Y_avg"][ag]
                    ci = emr_results[label]["TS_Y_std"][ag]*2
                    ax.fill_between(div_Y.index, (div_Y-ci), (div_Y+ci),
                                    color='grey', alpha=.1)
            for hy in range(1, num_hy_emr+1):
                for abm in range(1, num_abm_emr+1):
                    ax = axes[row-1, col]
                    label = (bt, at, hy, abm, "EMR{}-{}".format(hy, abm))
                    name = name_dict[(bt, at)]
                    div_Y = emr_results[label]["TS_Y_avg"][ag]
                    ax.plot(div_Y.index, div_Y, color="C{}".format(hy-1), lw=0.6,
                            ls=ls_list[abm-1], label="EMR{}.{}".format(hy,abm))
                    ax.set_ylim(ag_y_limits[col])
                    ax.tick_params(length=2)
            
            # Obeservation
            ax.plot(obv_Y[ag], color = "red", lw=0.8, label="Obv")
            ax.tick_params(axis='x', which='major', labelsize=8, rotation=70)
            ax.tick_params(axis='y', which='major', labelsize=7, pad=1)
            ax.yaxis.get_major_locator().set_params(integer=True)
            
            # Add cali and vali line
            ax.axvline(pd.to_datetime("2000-1-1"), color="black", ls="--", lw=0.5)
            
            # Subtitle and legend
            if row == 1:
                ax.set_title(ag, fontsize=11)
                if col == 4:
                    leg = ax.legend(
                        bbox_to_anchor=(1, 0.1), fontsize=8, handletextpad=0.2,
                        labelspacing=1, columnspacing=1, handlelength=2)
                    for line in leg.get_lines():
                        line.set_linewidth(1.5)
            # Y axis sublabel
            if col == 0:
                ax.set_ylabel(name, labelpad=2)

# Add labels for the entire figure.
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.ylabel("Annual irrigation diversion ($m^3/sec$)", fontsize=12, labelpad=8)
plt.xlabel("\nYear", fontsize=12)
fig.tight_layout()
plt.show()

