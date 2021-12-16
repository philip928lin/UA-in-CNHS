import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import HydroCNHS
import HydroCNHS.calibration as cali

pc = ""
path = r"".format(pc)
cali_output_path = os.path.join(path, "")
emr_path = os.path.join(path, "")
folder_list = os.listdir(cali_output_path)

# =============================================================================
# GA statistic
# =============================================================================
def evaluation(individual, info):
    fitness=0
    return (fitness,)

def descale(individual, bound_scale, lower_bound):
    """individual is 1d array ndarray."""
    individual = individual.reshape(bound_scale.shape)
    descaled_individual = np.subtract(individual, lower_bound)
    descaled_individual = np.divide(descaled_individual, bound_scale)
    return descaled_individual.flatten()


ga_statistic = {}
ga_sol = {}
ga_sol_name = {}
for f in folder_list:
    cali_save_path = os.path.join(cali_output_path, f, "GA_auto_save.pickle")
    ga = cali.GA_DEAP(evaluation)
    ga.load(cali_save_path, max_gen="")
    sol = ga.solution
    dsol = descale(sol, ga.bound_scale, ga.lower_bound)
    ga_sol[f] = dsol
    ga_sol_name[f] = ga.inputs["par_name"]
    summary = ga.summary    
    best_fitness = summary["max_fitness"][-1]
    avg_improve_rate = (best_fitness - summary["max_fitness"][-6])/5
    avg_pop_std = np.mean(summary["std"][-5:])
    ga_statistic[f] = [best_fitness, avg_improve_rate, avg_pop_std]

#%%
df_ga_statistic = pd.DataFrame()
for s in ["S1", "S2", "S3"]:
    fitness = [0,0,0]
    avg_improve_rate = [0,0,0]
    avg_pop_std = [0,0,0] 
    for i, seed in enumerate([9,28,83]):
        a, b, c = ga_statistic["Cali_{}_KGE_{}".format(s, seed)]
        fitness[i] = round(a, 3)
        avg_improve_rate[i] = round(b, 3)
        avg_pop_std[i] = round(c, 3)
    df_ga_statistic[s] = [max(fitness), tuple(fitness), tuple(avg_improve_rate), tuple(avg_pop_std)]

for s in ["C1C2G"]:
    fitness = [0,0,0]
    avg_improve_rate = [0,0,0]
    avg_pop_std = [0,0,0] 
    for i, seed in enumerate([9,28,83]):
        a, b, c = ga_statistic["Cali_{}_KGE_{}".format(s, seed)]
        fitness[i] = round(a, 3)
        avg_improve_rate[i] = round(b, 3)
        avg_pop_std[i] = round(c, 3)
    df_ga_statistic["Hydrological model"] = [max(fitness), tuple(fitness), tuple(avg_improve_rate), tuple(avg_pop_std)]

bt_list = ["Static", "Adaptive", "Learning"]
at_list = ["Linear", "Quadratic"]
for hy in [1,2]:
    for bt in bt_list:
        for at in at_list:
            if bt == "Static" and at == "Quadratic":
                continue
            fitness = [0,0,0]
            avg_improve_rate = [0,0,0]
            avg_pop_std = [0,0,0] 
            for i, seed in enumerate([9,28,83]):
                a, b, c = ga_statistic["Cali_ABM-{}-{}_KGE_{}_hydro{}".format(bt, at, seed, hy)]
                fitness[i] = round(a, 3)
                avg_improve_rate[i] = round(b, 3)
                avg_pop_std[i] = round(c, 3)
            df_ga_statistic["{}-{}-{}".format(bt, at, hy)] = \
                [max(fitness), tuple(fitness), tuple(avg_improve_rate), tuple(avg_pop_std)]
df_ga_statistic.index = ["Best fitness", "Fitness", "Average improve rate", "Average within-population standard deviation"]
df_ga_statistic = df_ga_statistic.T
df_ga_statistic.to_csv(os.path.join(emr_path, "ga_statistic.csv"))
#%%
# =============================================================================
# Stochastic test
# =============================================================================
# wd = r"".format(pc)
# module_path = os.path.join(path, r"")
# csv_path = os.path.join(path, "")
# input_path = os.path.join(path, r"")
# with open(os.path.join(input_path, ""),
#           "rb") as file:
#     (temp, prec, pet, obv_M, obv_Y) = pickle.load(file)
# mask = [True if i.month in [7,8,9] else False for i in obv_M.index]
# obv789 = obv_M.loc[mask, "G"].resample("YS").mean()
# obv_Y["G789"] = obv789
# mask2 = [True if i.month in [7,8,9] else False \
#          for i in pd.date_range("1959-1-1","2013-12-31",freq="M")]

# assigned_Q_res_inflow = pd.read_csv(
#     os.path.join(csv_path, "Inflow_D_R1R2R3_cms.csv"),
#     parse_dates=True, index_col=[0])["1959/1/1":]
# assigned_Q = {}
# for sub in ["S1","S2","S3"]:
#     assigned_Q[sub] = np.array(assigned_Q_res_inflow["R"+sub[1]])

# Qm_std = pd.DataFrame()
# for emr_name in tqdm([i for i in os.listdir(emr_path) if "Learning" in i or "Adaptive" in i]):
#     flow = pd.DataFrame()
#     # shortage = pd.DataFrame()
#     # div = pd.DataFrame()
#     for i in range(50):
#         model_dict = HydroCNHS.load_model(os.path.join(emr_path, emr_name))
#         model_dict["Path"]["WD"] = wd
#         model_dict["Path"]["Modules"] = module_path
#         model_dict["ABM"]["Inputs"]["Modules"] = ['YRB_ABM_Philip.py']
#         model_dict["ABM"]["Inputs"]["Database"] = os.path.join(csv_path, "Database_1959_2013.csv")
#         model = HydroCNHS.Model(model_dict)
#         Q = model.run(temp, prec, pet, assigned_Q=assigned_Q, disable=True)
#         sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)
#         sim_Q_M = sim_Q_D.resample("MS").mean()
#         sim_Q_Y = sim_Q_D.resample("YS").mean()
#         sim789 = sim_Q_M.loc[mask2, "G"].resample("YS").mean()
#         flow[i] = sim789
        
#         # ag_list = ['Kittitas', 'Roza', 'Wapato', 'Sunnyside', 'Tieton']
#         # shortage_M = []
#         # div_D = []
#         # for ag in ag_list:
#         #     shortage_M.append(model.data_collector.get_field(ag)["Shortage_M"])
#         #     div_D.append(model.data_collector.get_field(ag)["Div"])
#         # df_div = pd.DataFrame(div_D, index=ag_list, columns=model.pd_date_index).T
#         # div[i] = df_div.resample("YS").mean().sum(axis=1)
#         # df = pd.DataFrame(
#         #     shortage_M, index=ag_list,
#         #     columns=pd.date_range("1959-1-1", "2013-12-31", freq="MS")).T
#         # shortage[i] = df.resample("YS").mean().sum(axis=1)
#     Qm = pd.DataFrame()
#     for y in range(flow.shape[0]):
#         Qm[y] = flow.iloc[:y, :].quantile(0.5)
#     Qm = Qm.T
#     Qm_std[emr_name] = Qm.std(axis=1)
# Qm_std.to_csv(os.path.join(emr_path, "Qm_std_50.csv"))

# Qm_std.plot(legend=True)
# # flow.std(axis=1).plot()
# # div.std(axis=1).plot()
# # shortage.std(axis=1).plot()
