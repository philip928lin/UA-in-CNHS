import os
import pickle
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
wd_done_cali = os.path.join(path, r"")
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

model_path = os.path.join(path, r"", "Cali_ABM.yaml")

bt_list = ["Learning", "Adaptive", "Static"]
at_list = ["Linear", "Quadratic"]
hy_list = ["hydro1", "hydro2"]
emr_info = {}
df_explained_var = pd.DataFrame()
for h, hy in enumerate(hy_list):
    for bt in bt_list:
        for at in at_list:
            if bt == "Static" and at == "Quadratic":
                continue

            hydro_model_path = os.path.join(
                path, r"NewModel", "HydroEMR{}.yaml".format(h+1))                 # Check
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

            db_path = os.path.join(csv_path, "Database_1959_2013.csv")               # Check
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


            #model = HydroCNHS.Model(model_dict)
            #Q = model.run(temp, prec, pet, assigned_Q=assigned_Q)

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
                    if it >= 2: # at least three times of simulation.
                        if np.std(div_KGE_temp) <= 0.1:
                            break

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
            #%%

            df_all_individuals = pd.DataFrame()
            for seed in [83, 9, 28]:#, 2, 3]:
                folder_name = "Cali_ABM-{}-{}_KGE_{}_{}".format(bt, at, seed, hy)
                cali_save_path = os.path.join(wd_done_cali, folder_name,
                                              "GA_auto_save.pickle")
                ga = cali.GA_DEAP(evaluation)
                ga.load(cali_save_path, max_gen="")
                print("{}: {}".format(folder_name, round(ga.summary["max_fitness"][-1], 3)))
                all_indiv = []
                for i, v in ga.records.items():
                    all_indiv += v
                all_indiv_fitness = [i.fitness.values[0] for i in all_indiv]
                df_ga = pd.DataFrame(all_indiv)
                df_ga["fitness"] = all_indiv_fitness
                df_ga = df_ga.drop_duplicates()
                df_all_individuals = pd.concat([df_all_individuals, df_ga])
            mask = df_all_individuals["fitness"] > df_all_individuals["fitness"].quantile(0.99)
            df_abm_q99 = df_all_individuals[mask]

            df_abm_q99.iloc[:,:-1].T.plot(legend=False, lw=0.1)
            # df_abm_q99.iloc[:,-1].hist(bins=50)

            #%%
            # =============================================================================
            # Run Kmeans & Identify EMRs
            # =============================================================================
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

            def plot_EMRs(df_hydroEMRs, df=None, title=None):
                k = df_hydroEMRs.shape[0]
                fig, ax = plt.subplots()
                if df is not None:
                    ax.plot(df.iloc[:,:-1].T, lw=0.2, alpha=0.3)
                for i in range(k):
                    ax.plot(df_hydroEMRs.iloc[i,:-2], label="EMR{} ({})".format(str(i+1),
                        round(df_hydroEMRs.iloc[i,-2],2)), color="C{}".format(i%10))
                ax.axhline(1, ls="dashed", color="black", lw=0.5)
                ax.axhline(0, ls="dashed", color="black", lw=0.5)
                ax.set_ylim([-0.05,1.15])
                ax.set_ylabel("Normalized parameter's value.")
                ax.set_xlabel("Parameter index")
                ax.legend(ncol=k, loc="upper right")
                if title is not None:
                    ax.set_title(title)
                plt.show()

            kmeans_models, explained_var = run_kmeans(df_abm_q99, k_min=1, k_max=10)
            df_explained_var["{}_{}_{}".format(bt, at, hy)] = explained_var
            df_EMRs = extract_EMRs_from_kmeans(df_abm_q99, kmeans_models, k=4)
            plot_EMRs(df_EMRs, df_abm_q99, "{}_{}_{}".format(bt, at, hy))
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
            emr_path = os.path.join(path, r"NewModel\EMRs_Output")

            for k in df_EMRs["EMR"]:
                individual = np.array(df_EMRs[df_EMRs["EMR"]==k].iloc[:,:-2])
                individual = scale(individual, ga.bound_scale, ga.lower_bound)
                df_list = cali.Convertor.to_df_list(individual, formatter)
                model = deepcopy(model_dict)
                model = HydroCNHS.load_df_to_model_dict(model, df_list[-1], "ABM", "Pars")
                model_name = os.path.join(
                    emr_path, "{}-{}-EMR{}-{}.yaml".format(bt, at, h+1, k))
                HydroCNHS.write_model(model, model_name)
# Elbow plot!!!!!!
df_explained_var.to_csv(os.path.join(emr_path, "Explained_var_for_abmEMR.csv"))
df_explained_var.plot()
