import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import HydroCNHS
import HydroCNHS.calibration as cali
# =============================================================================
# Path setting
# =============================================================================
pc = ""
path = r"".format(pc)
wd = r"".format(pc)
module_path = os.path.join(path, r"")
csv_path = os.path.join(path, "")
input_path = os.path.join(path, r"")
bound_path = os.path.join(path, r"")
best_model_output_path = os.path.join(wd, "")
cali_path = os.path.join(path,r"")
# =============================================================================
# Load Weather Data & Obv_M & ObvY
# =============================================================================
with open(os.path.join(input_path, "YRB_cali_inputs.pickle"),
          "rb") as file:
    (temp, prec, pet, obv_M, obv_Y) = pickle.load(file)
mask = [True if i.month in [7,8,9] else False for i in obv_M.index]
obv789 = obv_M.loc[mask, "G"].resample("YS").mean()
obv_Y["G789"] = obv789
mask2 = [True if i.month in [7,8,9] else False \
         for i in pd.date_range("1959-1-1","2013-12-31",freq="M")]
# =============================================================================
# Load AssignQ
# =============================================================================
assigned_Q_res_inflow = pd.read_csv(
    os.path.join(csv_path, "Inflow_D_R1R2R3_cms.csv"),
    parse_dates=True, index_col=[0])["1959/1/1":]
assigned_Q = {}
for sub in ["S1","S2","S3"]:
    assigned_Q[sub] = np.array(assigned_Q_res_inflow["R"+sub[1]])

model_path = os.path.join(path, r"NewModel\Template", "Cali_C1C2G.yaml")
# model_path = os.path.join(path, r"NewModel", "ccg.yaml")
model_dict = HydroCNHS.load_model(model_path)
model_dict["Path"]["WD"] = wd
model_dict["Path"]["Modules"] = module_path
# =============================================================================
# Create Formatter & Calibration Inputs.
# =============================================================================
df_list, df_name = HydroCNHS.write_model_to_df(model_dict, key_option=["Pars"])
par_bound_df_list = [
    pd.read_csv(os.path.join(bound_path, "Cali_ParBound_C1C2G-LSM_GWLF.csv"), index_col=[0]),
    pd.read_csv(os.path.join(bound_path, "Cali_ParBound_C1C2G-Routing_Lohmann.csv"), index_col=[0]),
    pd.read_csv(os.path.join(bound_path, "Cali_ParBound_C1C2G-ABM.csv"), index_col=[0])]
converter = cali.Convertor()
converter.gen_cali_inputs(wd, df_list, par_bound_df_list,
                          par_type_df_list=["real"]*3)
formatter = converter.formatter
cali_inputs = converter.inputs
# =============================================================================
# Function to load pickle
# =============================================================================
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

def evaluation(individual, info):
    cali_wd, current_generation, ith_individual, formatter, rn_gen_c = info
    name = "{}-{}".format(current_generation, ith_individual)

    ##### individual -> model
    df_list = cali.Convertor.to_df_list(individual, formatter)
    # ModelDict is from Model Builder (template with -99).
    model = deepcopy(model_dict)
    for i, df in enumerate(df_list):
        s = df_name[i].split("_")[0]
        model = HydroCNHS.load_df_to_model_dict(model, df, s, "Pars")

    ##### Run simuluation
    model = HydroCNHS.Model(model, name)
    #HydroCNHS.write_model(model, os.path.join(path, r"NewModel\ccg.yaml"))
    try:
        Q = model.run(temp, prec, pet, assigned_Q=assigned_Q, disable=True)
    except:
        return (-100,)
    # Get simulation data
    cali_target = ["C1", "C2", "G"]
    cali_period = ("1960-1-1", "1999-12-31")
    vali_period = ("2000-1-1", "2013-12-31")

    sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[cali_target]
    sim_Q_M = sim_Q_D[cali_target].resample("MS").mean()
    sim_Q_Y = sim_Q_D[cali_target].resample("YS").mean()

    sim789 = sim_Q_M.loc[mask2, "G"].resample("YS").mean()
    sim_Q_Y["G789"] = sim789

    df_cali_Q_M = cal_batch_indicator(cali_period, cali_target, obv_M, sim_Q_M)
    df_cali_Q_Y = cal_batch_indicator(cali_period, cali_target + ["G789"], obv_Y, sim_Q_Y)

    df_vali_Q_M = cal_batch_indicator(vali_period, cali_target, obv_M, sim_Q_M)
    df_vali_Q_Y = cal_batch_indicator(vali_period, cali_target + ["G789"], obv_Y, sim_Q_Y)

    ag_list = ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']
    shortage_M = []
    for ag in ag_list:
        shortage_M.append(model.data_collector.get_field(ag)["Shortage_M"])
    df = pd.DataFrame(
        shortage_M, index=ag_list,
        columns=pd.date_range("1959-1-1", "2013-12-31", freq="MS")).T
    df = df[cali_period[0]: cali_period[1]]
    mean_Y_shortage = df.groupby(df.index.year).mean().mean().sum()

    ##### Save output.txt
    with open(os.path.join(cali_wd, "cali_indiv_" + name + ".txt"), 'w') as f:
        f.write("Annual cali/vali result\n")
        f.write(df_cali_Q_Y.round(3).to_csv(sep='\t').replace("\n", ""))
        f.write("\n")
        f.write(df_vali_Q_Y.round(3).to_csv(sep='\t').replace("\n", ""))
        f.write("\n\nMonthly cali/vali result\n")
        f.write(df_cali_Q_M.round(3).to_csv(sep='\t').replace("\n", ""))
        f.write("\n")
        f.write(df_vali_Q_M.round(3).to_csv(sep='\t').replace("\n", ""))
        f.write("\n=========================================================\n")
        f.write("Sol:\n" )
        df = pd.DataFrame(individual, index=cali_inputs["par_name"]).round(4)
        f.write(df.to_string(header=False, index=True))

    # We weight mean_Y_shortage to 10. Should not have shortage.
    #fitness = df_cali_Q_M.loc["Mean", "KGE"] + df_vali_Q_Y.loc["G789", "KGE"] + (1 - mean_Y_shortage)*10
    fitness = df_cali_Q_M.loc["Mean", "KGE"] + (1 - mean_Y_shortage)*10
               #+ df_cali_Q_M.loc["Mean", "iKGE"]) / 2
    return (fitness,)
#%%
df_all_individuals = pd.DataFrame()
for seed in [83, 9, 28]:#, 2, 3]:
    cali_save_path = os.path.join(cali_path, "Cali_C1C2G_KGE_{}".format(seed),
                                  "GA_auto_save.pickle")
    ga = cali.GA_DEAP(evaluation)
    ga.load(cali_save_path, max_gen="")
    print("Seed {}: {}".format(seed, ga.summary["max_fitness"][-1]))
    all_indiv = []
    for i, v in ga.records.items():
        all_indiv += v
    all_indiv_fitness = [i.fitness.values[0] for i in all_indiv]
    df_ga = pd.DataFrame(all_indiv)
    df_ga["fitness"] = all_indiv_fitness
    df_ga = df_ga.drop_duplicates()
    df_all_individuals = pd.concat([df_all_individuals, df_ga])
mask = df_all_individuals["fitness"] > df_all_individuals["fitness"].quantile(0.99)
df_ccg_q99 = df_all_individuals[mask]

df_ccg_q99.iloc[:,:-1].T.plot(legend=False, lw=0.1)
df_ccg_q99.iloc[:,-1].hist(bins=50)
visual = HydroCNHS.Visual()

#%%
# =============================================================================
# Run Kmeans & Identify EMRs
# =============================================================================
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(df, k_min=1, k_max=10):
    df_km = df.iloc[:,:-1]
    kmeans_models = {}
    k_distortions = []
    k_explained_var = []
    k_silhouette_avg = []
    sse = np.sum(np.var(df_km, axis=0))*df_km.shape[0]
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=0).fit(df_km)
        kmeans_models[k] = km
        k_distortions.append(km.inertia_)
        k_explained_var.append((sse - k_distortions[-1])/sse)
        cluster_labels = km.labels_
        if k == 1:  # If given k == 1, then assign the worst value.
            k_silhouette_avg.append(-1)
        else:
            si_avg = silhouette_score(df_km, cluster_labels)
            k_silhouette_avg.append(si_avg)
    fig, ax = plt.subplots()
    x = np.arange(k_min, k_max+1)
    ax.plot(x, k_silhouette_avg, "+-", label = "Silhouette Score")
    ax.plot(x, k_explained_var, "+-", label = "Explained Var")
    ax.legend()
    plt.show()
    return kmeans_models, k_explained_var

def extract_EMRs_from_kmeans(df, kmeans_models, k, min_or_max="max"):
    df = df.copy()
    km = kmeans_models[k]
    df["EMR"] = km.labels_
    df_EMRs = pd.DataFrame()
    for g in range(k):
        dff = df[df["EMR"] == g]
        if min_or_max == "max":
            opt_fitness = max(dff["fitness"])
        elif min_or_max == "max":
            opt_fitness = min(dff["fitness"])
        dff = dff[dff["fitness"] == opt_fitness]
        df_EMRs = pd.concat([df_EMRs, dff.iloc[[0],:]])
    df_EMRs = df_EMRs.sort_values("fitness", ascending=False)
    df_EMRs["EMR"] = np.arange(1, k+1)
    return df_EMRs

def plot_EMRs(df_hydroEMRs, df=None):
    k = df_hydroEMRs.shape[0]
    fig, ax = plt.subplots()
    if df is not None:
        ax.plot(df.iloc[:,:-1].T, lw=0.2, alpha=0.3)
    for i in range(k):
        ax.plot(df_hydroEMRs.iloc[i,:-2], label="HydroEMR{} ({})".format(str(i+1),
            round(df_hydroEMRs.iloc[i,-2],2)), color="C{}".format(i%10))
    ax.axhline(1, ls="dashed", color="black", lw=0.5)
    ax.axhline(0, ls="dashed", color="black", lw=0.5)
    ax.set_ylim([-0.05,1.15])
    ax.set_ylabel("Normalized parameter's value.")
    ax.set_xlabel("Parameter index")
    ax.legend(ncol=k, loc="upper right")
    plt.show()

kmeans_models, k_explained_var = run_kmeans(df_ccg_q99, k_min=1, k_max=10)
df_hydroEMRs = extract_EMRs_from_kmeans(df_ccg_q99, kmeans_models, k=2)
plot_EMRs(df_hydroEMRs, df_ccg_q99)

fig, ax = plt.subplots()
ax.plot(k_explained_var, label="Hydrological model", marker="o")
ax.axvline(2, color="black", ls="--", lw=1)
ax.legend(ncol=1, fontsize=8)
ax.set_ylabel("Explained variance")
ax.set_xlabel("Number of Kmeans clusters")
ax.set_ylim([0,0.9])
#%%
# =============================================================================
# Evaluate EMRs
# =============================================================================
def scale(individual, bound_scale, lower_bound):
    """individual is 1d ndarray."""
    individual = individual.reshape(bound_scale.shape)
    scaled_individual = np.multiply(individual, bound_scale)
    scaled_individual = np.add(scaled_individual, lower_bound)
    return scaled_individual.flatten()
bound_path = os.path.join(path, r"NewModel\EMRs_Output")

fitness_hydroEMRs = []
for k in df_hydroEMRs["EMR"]:
    individual = np.array(df_hydroEMRs[df_hydroEMRs["EMR"]==k].iloc[:,:-2])
    individual = scale(individual, ga.bound_scale, ga.lower_bound)
    df_list = cali.Convertor.to_df_list(individual, formatter)
    model = deepcopy(model_dict)
    for i, df in enumerate(df_list):
        s = df_name[i].split("_")[0]
        model = HydroCNHS.load_df_to_model_dict(model, df, s, "Pars")
    model_name = os.path.join(bound_path, "HydroEMR{}.yaml".format(k))
    HydroCNHS.write_model(model, model_name)
    info = (bound_path, "HydroEMR{}".format(k), "best", formatter, None)
    fitness_hydroEMRs.append(evaluation(individual, info)[0])

#%%
# =============================================================================
# Prepare flow target for ABM calibration
# =============================================================================
emr_path = os.path.join(path, r"")
cali_target = ["C1", "C2", "G"]
k = 1
hydroEMR_model_path = os.path.join(emr_path, "HydroEMR{}.yaml".format(k))
model_dict = HydroCNHS.load_model(hydroEMR_model_path)
model_dict["Path"]["Modules"] = module_path
model_dict["ABM"]["Inputs"]["Modules"] = ['YRB_ABM_C1C2G_Philip.py']
model = HydroCNHS.Model(model_dict, "HydroEMR{}".format(k))
Q1 = model.run(temp, prec, pet, assigned_Q=assigned_Q)
sim_Q_D1 = pd.DataFrame(Q1, index=model.pd_date_index)[cali_target]["1960-1-1":"2013-12-31"]
sim_Q_M1 = sim_Q_D1[cali_target].resample("MS").mean()

k = 2
hydroEMR_model_path = os.path.join(emr_path, "HydroEMR{}.yaml".format(k))
model_dict = HydroCNHS.load_model(hydroEMR_model_path)
model_dict["Path"]["Modules"] = module_path
model_dict["ABM"]["Inputs"]["Modules"] = ['YRB_ABM_C1C2G_Philip.py']
model = HydroCNHS.Model(model_dict, "HydroEMR{}".format(k))
Q2 = model.run(temp, prec, pet, assigned_Q=assigned_Q)

sim_Q_D2 = pd.DataFrame(Q2, index=model.pd_date_index)[cali_target]["1960-1-1":"2013-12-31"]
sim_Q_M2 = sim_Q_D2[cali_target].resample("MS").mean()

minor_div = (((sim_Q_M1["G"] - obv_M["G"]).groupby(obv_M.index.month).mean() 
              + (sim_Q_M2["G"] - obv_M["G"]).groupby(obv_M.index.month).mean()) / 2).round(2)

for m in range(4, 11):
    mask = [True if i.month == m else False for i in obv_M.index]
    sim_Q_M1.loc[mask,"G"] = sim_Q_M1.loc[mask,"G"] - minor_div[m]
    sim_Q_M2.loc[mask,"G"] = sim_Q_M2.loc[mask,"G"] - minor_div[m]

#%%
fig, axes = plt.subplots(nrows=3)
axes = axes.flatten()
name_list = ["Umtanum", "Naches", "Parker"]
for i, c in enumerate(cali_target):
    ax = axes[i]
    ax.plot(sim_Q_M1[c], color="C1", lw=1, label="HydroEMR1")
    ax.plot(sim_Q_M2[c], color="C2", ls="dashed", lw=1, label="HydroEMR2")
    ax.plot(obv_M[c], color="black", ls="dotted", lw=1, label="Obv")
    ax.axvline(pd.to_datetime("2000-1-1"), color="black", ls="--", lw=1)
    ax.set_ylabel(name_list[i], labelpad=2)
    if i == 0:
        ax.legend(ncol=3,
            bbox_to_anchor=(0.4, 1), fontsize=8, handletextpad=0.2,
            labelspacing=1, columnspacing=1, handlelength=2)
# Add labels for the entire figure.
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.ylabel("Monthly streamflow ($m^3/sec$)", fontsize=11, labelpad=15)
plt.xlabel("\nYear", fontsize=11, labelpad=-5)
fig.tight_layout()
plt.show()


#%%
# mask = [True if i.month in [7,8,9] else False for i in obv_M.index]
# obv789 = obv_M.loc[mask, c].resample("YS").mean()
# sim789 = sim_Q_M.loc[mask, c].resample("YS").mean()
# sim789.plot()
# obv789.plot()
# visual.plot_timeseries(obv789, sim789, title="789 "+c)

# sim789[sim789.index.year <1985].quantile(0.5)
# sim789[sim789.index.year >= 1985].quantile(0.5)

# Flow target note:
# We set up the flow target based on the calibrated hydroEMRs to avoid the system bias in the later
# ABM calibration.
r"""
(With new bounds)
HydroEMR1: [46.96, 54.97]
HydroEMR2: [44.98, 53.99]
"""

