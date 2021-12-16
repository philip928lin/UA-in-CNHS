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
best_model_output_path = os.path.join(wd, "Cali_best")

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

model_path = os.path.join(path, r"\Template", "Cali_ABM.yaml")

for hy in ["hydro1", "hydro2"]:
    for bt in ["Learning", "Adaptive", "Static"]:
        for at in ["Quadratic", "Linear"]:
            if hy == "hydro1":
                hydro_model_path = os.path.join(path, r"NewModel", "HydroEMR1.yaml")
            elif hy == "hydro2":
                hydro_model_path = os.path.join(path, r"NewModel", "HydroEMR2.yaml")

                             # Check
            hydro_model_dict = HydroCNHS.load_model(hydro_model_path)
            model_dict = HydroCNHS.load_model(model_path)
            model_dict["Path"]["WD"] = wd
            model_dict["Path"]["Modules"] = module_path
            model_dict["LSM"] = hydro_model_dict["LSM"]
            model_dict["Routing"] = hydro_model_dict["Routing"]
            # Assign calibrated return factors
            model_dict["ABM"]["IrrDiv_AgType"]["Kittitas"]["Inputs"]["Links"]["C1"] = \
                hydro_model_dict["ABM"]["IrrDiv_AgType"]["Kittitas"]["Pars"]["DivFactor"][0]
            model_dict["ABM"]["IrrDiv_AgType"]["Tieton"]["Inputs"]["Links"]["G"] = \
                hydro_model_dict["ABM"]["IrrDiv_AgType"]["Tieton"]["Pars"]["ReturnFactor"][0]


            db_path = os.path.join(csv_path, "Database_1959_2013.csv")                       # Check
            model_dict["ABM"]["Inputs"]["BehaviorType"] = bt
            model_dict["ABM"]["Inputs"]["AdaptiveType"] = at
            if hy == "hydro1":
                model_dict["ABM"]["Inputs"]["FlowTarget"] = [46.96, 54.97]
            elif hy == "hydro2":
                model_dict["ABM"]["Inputs"]["FlowTarget"] = [44.98, 53.99]
            model_dict["ABM"]["Inputs"]["Database"] = db_path
            df_list, df_name = HydroCNHS.write_model_to_df(model_dict, key_option=["Pars"])
            abm_par_df = df_list[-1]
            if bt == "Static":
                abm_par_df.loc[['L_U', 'L_L', 'Lr_c', 'Sig', 'a', 'c'], :] = np.nan
            else:
                if at == "Linear":
                    abm_par_df.loc[['c'], :] = np.nan
                if bt == "Adaptive":
                    abm_par_df.loc[['L_U', 'L_L', 'Lr_c'], :] = np.nan
            model_dict = HydroCNHS.load_df_to_model_dict(model_dict, abm_par_df, "ABM", "Pars")

            #%
            # =============================================================================
            # Create Formatter & Calibration Inputs.
            # =============================================================================
            df_list, df_name = HydroCNHS.write_model_to_df(model_dict, key_option=["Pars"])
            par_bound_df_list = [
                pd.read_csv(os.path.join(
                    bound_path, "Cali_ParBound_ABM-ABM-{}.csv".format(at)),
                    index_col=[0])]
            converter = cali.Convertor()
            converter.gen_cali_inputs(wd, df_list[-1:], par_bound_df_list,
                                      par_type_df_list=["real"])
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
                model = HydroCNHS.load_df_to_model_dict(model, df_list[-1], "ABM", "Pars")

                ##### Run simuluation
                model = HydroCNHS.Model(model, name, rn_gen_c)
                #HydroCNHS.write_model(model, os.path.join(path, r"NewModel\ccg_abm.yaml"))
                if bt == "Static":
                    num_iter = 1
                else:
                    num_iter = 10
                df_cali_Q_M = pd.DataFrame()
                df_vali_Q_M = pd.DataFrame()
                df_cali_Q_Y = pd.DataFrame()
                df_vali_Q_Y = pd.DataFrame()
                div_KGE_temp = []
                for it in range(num_iter):
                    try:
                        Q = model.run(temp, prec, pet, assigned_Q=assigned_Q, disable=True)
                    except:
                        return (-100,)

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
                    # if it >= 2: # at least three times of simulation.
                    #     if np.std(div_KGE_temp) <= 0.1:
                    #         break

                df_cali_Q_M = df_cali_Q_M.groupby(df_cali_Q_M.index).mean()
                df_vali_Q_M = df_vali_Q_M.groupby(df_vali_Q_M.index).mean()
                df_cali_Q_Y = df_cali_Q_Y.groupby(df_cali_Q_Y.index).mean()
                df_vali_Q_Y = df_vali_Q_Y.groupby(df_vali_Q_Y.index).mean()

                flow_KGE = df_cali_Q_M.loc[cali_target, "KGE"].sum()
                div_KGE = df_cali_Q_Y.loc[ag_list, "KGE"].sum()
                # We weight mean_Y_shortage to 10. Should not have shortage.
                fitness = (flow_KGE + div_KGE)/8 + (1 - mean_Y_shortage)

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
                    f.write("Number of iterations: {}, actual = {}\n".format(
                        num_iter, len(div_KGE_temp)))
                    f.write("Fitness: {}\n".format(round(fitness, 5)))
                    f.write("Mean monthly flow KGE: {}\n".format(round(flow_KGE/3, 5)))
                    f.write("Mean annual div KGE: {}\n".format(round(div_KGE/5, 5)))
                    f.write("Mean annual div shortage: {}\n".format(round(mean_Y_shortage, 5)))
                    f.write("\n=========================================================\n")
                    f.write("Sol:\n" )
                    df = pd.DataFrame(individual, index=cali_inputs["par_name"]).round(4)
                    f.write(df.to_string(header=False, index=True))

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
                     'max_gen': 70,
                     'sampling_method': 'LHC',
                     'drop_record': False,
                     'paral_cores': -1,
                     'paral_verbose': 1,
                     'auto_save': True,
                     'print_level': 1,
                     'plot': True}

            if at == "Quadratic" and bt == "Static":
                continue

            for seed in [28, 83, 9]: #[2,3]
                rn_gen = HydroCNHS.create_rn_gen(seed)
                ga = cali.GA_DEAP(evaluation, rn_gen)
                # Check status
                folder_name = "Cali_ABM-{}-{}_KGE_{}_{}".format(bt, at, seed, hy)
                if os.path.isdir(os.path.join(wd, folder_name)):
                    max_gen = 70
                    ga.load(os.path.join(wd, folder_name, "GA_auto_save.pickle"),
                            max_gen)
                    ga.cali_wd = os.path.join(wd, folder_name)
                    print(folder_name)
                else:
                    ga.set(cali_inputs, config, formatter,
                           name="Cali_ABM-{}-{}_KGE_{}_{}".format(bt, at, seed, hy))
                ga.run()
                ga.run_individual(ga.solution)
                # Save model
                df_list = cali.Convertor.to_df_list(ga.solution, formatter)
                model = deepcopy(model_dict)
                model = HydroCNHS.load_df_to_model_dict(model, df_list[-1], "ABM", "Pars")
                HydroCNHS.write_model(model, os.path.join(
                    best_model_output_path, "ABM-{}-{}_KGE_{}_{}".format(bt, at, seed, hy)))



