import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
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

#%%
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
    fitness = df_cali_Q_M.loc["Mean", "KGE"] + (1 - mean_Y_shortage)*10
    return (fitness,)

#%% =============================================================================
# Calibration Preparation
# =============================================================================
config = {'min_or_max': 'max',
         'pop_size': 100,
         'num_ellite': 1,
         'prob_cross': 0.5,
         'prob_mut': 0.1,
         'stochastic': False,
         'max_gen': 100,
         'sampling_method': 'LHC',
         'drop_record': False,
         'paral_cores': -1,
         'paral_verbose': 1,
         'auto_save': True,
         'print_level': 1,
         'plot': True}
for seed in [83, 9, 28]:
    rn_gen = HydroCNHS.create_rn_gen(seed)
    ga = cali.GA_DEAP(evaluation, rn_gen)
    ga.set(cali_inputs, config, formatter, name="Cali_C1C2G_KGE_{}".format(seed))
    ga.run()
    ga.run_individual(ga.solution)
    # Save model
    df_list = cali.Convertor.to_df_list(ga.solution, formatter)
    model = deepcopy(model_dict)
    for i, df in enumerate(df_list):
        s = df_name[i].split("_")[0]
        model = HydroCNHS.load_df_to_model_dict(model, df, s, "Pars")
    HydroCNHS.write_model(model, os.path.join(best_model_output_path,
                                              "C1C2G_KGE_{}.yaml".format(seed)))

