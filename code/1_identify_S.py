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
input_path = os.path.join(path, r"")
bound_path = os.path.join(path, r"")
s_list = ["S1", "S2", "S3"]
# =============================================================================
# Load Weather Data & Obv_M & ObvY
# =============================================================================
with open(os.path.join(input_path, "YRB_cali_inputs.pickle"),
          "rb") as file:
    (temp, prec, pet, obv_M, obv_Y) = pickle.load(file)

for s in s_list:
    wd_done_cali = os.path.join(path, r"NewCaliResult")
    df_all_individuals = pd.DataFrame()
    for seed in [83, 9, 28]:#, 2, 3]:
        model_path = os.path.join(path, r"NewModel\Template", "Cali_{}.yaml".format(s))
        model_dict = HydroCNHS.load_model(model_path)
        model_dict["Path"]["WD"] = wd

        # =============================================================================
        # Create Formatter & Calibration Inputs.
        # =============================================================================
        df_list, df_name = HydroCNHS.write_model_to_df(model_dict, key_option=["Pars"])
        par_bound_df_list = [
            pd.read_csv(os.path.join(bound_path, "Cali_ParBound_S_Pars_LSM_GWLF.csv"), index_col=[0]),
            pd.read_csv(os.path.join(bound_path, "Cali_ParBound_S_Pars_Routing_Lohmann.csv"), index_col=[0])]
        converter = cali.Convertor()
        converter.gen_cali_inputs(wd, df_list, par_bound_df_list,
                                  par_type_df_list=["real"]*3)
        formatter = converter.formatter
        cali_inputs = converter.inputs

        #%
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
                section = df_name[i].split("_")[0]
                model = HydroCNHS.load_df_to_model_dict(model, df, section, "Pars")

            ##### Run simuluation
            model = HydroCNHS.Model(model, name)
            try:
                Q = model.run(temp, prec, pet)
            except:
                return (-100,)
            # Get simulation data
            cali_target = [s]
            cali_period = ("1960-1-1", "1999-12-31")
            vali_period = ("2000-1-1", "2013-12-31")

            sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)[cali_target]
            sim_Q_M = sim_Q_D[cali_target].resample("MS").mean()
            sim_Q_Y = sim_Q_D[cali_target].resample("YS").mean()

            df_cali_Q_M = cal_batch_indicator(cali_period, cali_target, obv_M, sim_Q_M)
            df_cali_Q_Y = cal_batch_indicator(cali_period, cali_target, obv_Y, sim_Q_Y)

            df_vali_Q_M = cal_batch_indicator(vali_period, cali_target, obv_M, sim_Q_M)
            df_vali_Q_Y = cal_batch_indicator(vali_period, cali_target, obv_Y, sim_Q_Y)

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

            # fitness = (df_cali_Q_M.loc["Mean", "KGE"]
            #            + df_cali_Q_M.loc["Mean", "iKGE"]) / 2
            fitness = df_cali_Q_M.loc["Mean", "KGE"]
            return (fitness,)

        folder_name = "Cali_{}_KGE_{}".format(s, seed)
        cali_save_path = os.path.join(wd_done_cali, folder_name,
                                      "GA_auto_save.pickle")
        ga = cali.GA_DEAP(evaluation)
        ga.load(cali_save_path, max_gen="")
        print("\n{}: {}\n".format(folder_name, round(ga.summary["max_fitness"][-1], 3)))
        all_indiv = []
        for i, v in ga.records.items():
            all_indiv += v
        all_indiv_fitness = [i.fitness.values[0] for i in all_indiv]
        df_ga = pd.DataFrame(all_indiv)
        df_ga["fitness"] = all_indiv_fitness
        df_ga = df_ga.drop_duplicates()
        df_all_individuals = pd.concat([df_all_individuals, df_ga])
    mask = df_all_individuals["fitness"] == df_all_individuals["fitness"].max()
    df_max = df_all_individuals[mask]

    # =============================================================================
    # To yaml
    # =============================================================================
    def scale(individual, bound_scale, lower_bound):
        """individual is 1d ndarray."""
        individual = individual.reshape(bound_scale.shape)
        scaled_individual = np.multiply(individual, bound_scale)
        scaled_individual = np.add(scaled_individual, lower_bound)
        return scaled_individual.flatten()
    emr_path = os.path.join(path, r"NewModel\EMRs_Output")

    individual = np.array(df_max.iloc[:,:-1])
    individual = scale(individual, ga.bound_scale, ga.lower_bound)
    df_list = cali.Convertor.to_df_list(individual, formatter)
    model = deepcopy(model_dict)
    for i, df in enumerate(df_list):
        section = df_name[i].split("_")[0]
        model = HydroCNHS.load_df_to_model_dict(model, df, section, "Pars")
    model_name = os.path.join(
        emr_path, "{}_KGE_best.yaml".format(s))
    HydroCNHS.write_model(model, model_name)

