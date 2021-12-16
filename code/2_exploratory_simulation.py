import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import HydroCNHS

pc = ""
path = r"".format(pc)
wd = r"".format(pc)
module_path = os.path.join(path, r"")
emr_path = os.path.join(path, r"")
sc_path = os.path.join(path, r"")
output_path = wd
#==============================================================================
# Check YRB_ABM_sim.py before running this.
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
#==============================================================================
assigned_Q = {}     # doesn't matter. This will be overwrited by res.
for sub in ["S1","S2","S3"]:
    assigned_Q[sub] = [0]*29219

def run_emr(input_list, seed):
    [temp, prec, pet, assigned_Q, db_path, res_path, label] = input_list
    (rcp,prec_q,temp_q,r,hy,abm,bt,at,ite) = label

    # Modify model dict.
    emr_name = "{}-{}-EMR{}-{}.yaml".format(bt, at, hy, abm)
    model_dict = HydroCNHS.load_model(os.path.join(emr_path, emr_name))
    model_dict["WaterSystem"]["StartDate"] = "2021/1/1"
    model_dict["WaterSystem"]["EndDate"] = "2100/12/31"
    model_dict["WaterSystem"]["DataLength"] = 29219
    model_dict["ABM"]["Inputs"]["Database"] = db_path
    model_dict["Path"]["WD"] = wd
    model_dict["Path"]["Modules"] = module_path
    model_dict["ABM"]["Inputs"]["Modules"] = ['YRB_ABM_sim.py']
    model_dict["ABM"]["Inputs"]["FlowTarget"] = [54.48, 54.48]
    for res_ag in ["R1", "R2", "R3"]:
        model_dict["ABM"]["ResDam_AgType"][res_ag]["Inputs"]["ResPath"] = res_path

    rn_gen = HydroCNHS.create_rn_gen(seed)
    model = HydroCNHS.Model(model_dict, rn_gen=rn_gen)
    Q = model.run(temp, prec, pet, assigned_Q=assigned_Q, disable=True)

    sim_Q_D = pd.DataFrame(Q, index=model.pd_date_index)
    sim_Q_M = sim_Q_D.resample("MS").mean()
    mask = [True if i.month in [7,8,9] else False for i in sim_Q_M.index]
    sim789 = sim_Q_M.loc[mask, "G"].resample("YS").mean()

    ag_list = ['Kittitas', 'Tieton', 'Roza', 'Wapato', 'Sunnyside']
    shortage_M = []
    div_D = []
    Qup = []
    Qdown = []
    div_req_Y = []
    div_ref_Y = []
    for ag in ag_list:
        shortage_M.append(model.data_collector.get_field(ag)["Shortage_M"])
        div_D.append(model.data_collector.get_field(ag)["Div"])
        Qup.append(model.data_collector.get_field(ag)["Qup"])
        Qdown.append(model.data_collector.get_field(ag)["Qdown"])
        div_req_Y.append(model.data_collector.get_field("DivDM")[ag]["DivReq_Y"])
        div_ref_Y.append(model.data_collector.get_field("DivDM")[ag]["DivReqRef"])
    div = pd.DataFrame(div_D, index=ag_list, columns=model.pd_date_index).T
    shortage = pd.DataFrame(
        shortage_M, index=ag_list,
        columns=pd.date_range("2021-1-1", "2100-12-31", freq="MS")).T
    Qup = pd.DataFrame(Qup, index=ag_list, columns=model.pd_date_index).T
    Qup = Qup[['Kittitas', 'Tieton', 'Roza']].resample("MS").mean()
    Qup_G_789 = Qup.loc[mask, 'Roza']
    Qdown = pd.DataFrame(Qdown, index=ag_list, columns=model.pd_date_index).T
    Qdown = Qdown[['Kittitas', 'Tieton', 'Roza']].resample("MS").mean()
    Qdown_G_789 = Qdown.loc[mask, 'Roza']
    div_req_Y = pd.DataFrame(
        div_req_Y, index=ag_list,
        columns=pd.date_range("2021-1-1", "2100-12-31", freq="YS")).T
    div_ref_Y = pd.DataFrame(
        div_ref_Y, index=ag_list,
        columns=pd.date_range("2021-1-1", "2100-12-31", freq="YS")).T

    div_Y = div.resample("YS").mean()
    shortage_Y = shortage.resample("YS").sum() #m3/86400
    Qup_Y = Qup.resample("YS").mean()
    Qdown_Y = Qdown.resample("YS").mean()
    Qup_G_789_Y = Qup_G_789.resample("YS").mean()
    Qdown_G_789_Y = Qdown_G_789.resample("YS").mean()

    # We store main output for UA in result_dict.
    # We pickle the entire model in other place.
    result_dict = {"sim789": sim789,
                   "div_Y": div_Y,
                   "shortage_Y": shortage_Y,
                   "Qup_Y": Qup_Y,
                   "Qdown_Y": Qdown_Y,
                   "Qup_G_789_Y": Qup_G_789_Y,
                   "Qdown_G_789_Y": Qdown_G_789_Y,
                   "div_req_Y": div_req_Y,
                   "div_ref_Y": div_ref_Y}

    # (rcp,prec_q,temp_q,r,hy,abm,bt,at,ite) = label
    # filename = "Sc_model_{}-{}-{}-{}-{}-{}-{}-{}-{}.p".format(
    #     rcp,prec_q,temp_q,r,hy,abm,bt,at,ite)
    # pickle.dump(model, open(os.path.join(output_path, filename)), "wb")

    return result_dict

# Prepare random seeds
seed = 12
rn_gen = HydroCNHS.create_rn_gen(seed)
ss = rn_gen.bit_generator._seed_seq
seeds = ss.spawn(num_iter * len(rcp_list) * len(q_list)**2 * num_realization * len(bt_list) * len(at_list) * num_hy_emr * num_abm_emr)

seed_count = 0
for ite in range(11, num_iter+1):
    for rcp in rcp_list:
        sim_inputs_dict = {}
        sim_list = []
        for prec_q in q_list:
            for temp_q in q_list:
                for r in tqdm(range(num_realization),
                              desc="{}_{}_P{}_T{}".format(ite,rcp,prec_q,temp_q)):
                    # Sc file path
                    prec_path = os.path.join(
                        sc_path, "R{}_BCWth_D_Prec_cm_{}_P{}_T{}.csv".format(
                            r, rcp, prec_q, temp_q))
                    temp_path = os.path.join(
                        sc_path, "R{}_BCWth_D_Tavg_degC_{}_P{}_T{}.csv".format(
                            r, rcp, prec_q, temp_q))
                    pet_path = os.path.join(
                        sc_path, "R{}_BCWth_D_Pet_cm_{}_P{}_T{}.csv".format(
                            r, rcp, prec_q, temp_q))
                    db_path = os.path.join(
                        sc_path, "R{}_Database_2021_2100_{}_P{}_T{}.csv".format(
                            r, rcp, prec_q, temp_q))
                    res_path = os.path.join(
                        sc_path, "R{}_Release_D_R1R2R3_cms_{}_P{}_T{}.csv".format(
                            r, rcp, prec_q, temp_q))
                    # Load data
                    temp = pd.read_csv(temp_path, parse_dates=True, index_col=[0]).to_dict(orient='list')
                    prec = pd.read_csv(prec_path, parse_dates=True, index_col=[0]).to_dict(orient='list')
                    pet = pd.read_csv(pet_path, parse_dates=True, index_col=[0]).to_dict(orient='list')

                    for bt in bt_list:
                        if bt == "Static" and ite != 1:
                            # No stochastic component in Static. Need only one iteration.
                            continue
                        for at in at_list:
                            if bt == "Static" and at == "Quadratic":
                                # No difference in Linear or Quadratic for Static. Run one is fine.
                                continue
                            for hy in range(1, num_hy_emr+1):
                                for abm in range(1, num_abm_emr+1):
                                    label = (rcp,prec_q,temp_q,r,hy,abm,bt,at,ite)
                                    sim_inputs_dict[label] = [temp, prec, pet,
                                                              assigned_Q,
                                                              db_path,
                                                              res_path, label]
                                    sim_list.append(label)

        sim_results = {}
        Parel = Parallel(n_jobs=-1, verbose=2)( delayed(run_emr)(sim_inputs_dict[n], seeds[seed_count + s]) for s, n in enumerate(sim_list) )
        for i, v in enumerate(sim_list):
            sim_results[v] = Parel[i]
        pickle.dump(sim_results, open(
            os.path.join(output_path, "Sc_results_rcp_{}_iter_{}.p".format(rcp, ite)),
            "wb"))
        seed_count += 24000
