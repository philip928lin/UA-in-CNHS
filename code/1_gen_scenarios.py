import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import HydroCNHS
from joblib import Parallel, delayed
from dateutil.relativedelta import relativedelta

pc = ""
path = r"".format(pc)
csv_path = os.path.join(path, "")
input_path = os.path.join(path, "")
scenario_path = os.path.join(csv_path, "")

quantile_list = ["10","30","50","70","90"]

#%% =============================================================================
# Load Original BC Weather
# =============================================================================
gcm_path = os.path.join(path, "")
temp = pd.read_csv(os.path.join(csv_path, "BCWth_D_Tavg_degC.csv"),
                   parse_dates=True, index_col=[0])
prec = pd.read_csv(os.path.join(csv_path, "BCWth_D_Prec_cm.csv"),
                   parse_dates=True, index_col=[0])

#%% =============================================================================
# Bootstrapping the weather data to address the internal variation
# Gen 2021 ~ 2100 for 2030s 2050s 2070s 2090s
# =============================================================================
def is_leap_and_29Feb(s):
    return (s.index.year % 4 == 0) & \
           ((s.index.year % 100 != 0) | (s.index.year % 400 == 0)) & \
           (s.index.month == 2) & (s.index.day == 29)
mask = is_leap_and_29Feb(prec)
prep_no29Feb = prec[~mask]      # Eliminate Feb29
temp_no29Feb = temp[~mask]      # Eliminate Feb29

gen_rng = pd.date_range("2021/1/1", "2100/12/31")
year_pool = np.arange(1960,2014).astype(int)     # Same as the baseline
# We gen 30 but we only use 20 later. (num_realization, TotalMonth)
selected_year = np.empty((30, 12*(2100-2021+1)))
np.random.seed(9)  # Seed
# Bootstrapping
for i in range(selected_year.shape[0]):
    for j in range(selected_year.shape[1]):
        selected_year[i,j] = np.random.choice(year_pool)

# Form scenario data file
for i in tqdm(range(selected_year.shape[0]), desc="ShuffleWth"):
    df_P = pd.DataFrame()
    df_T = pd.DataFrame()
    for j in range(selected_year.shape[1]):
        # Year and Month
        y = 2021 + int(j/12)
        m = j%12 + 1

        # prec
        prep_m = prep_no29Feb[prep_no29Feb.index.month == m]
        prep_d = prep_m[prep_m.index.year == selected_year[i,j]]
        prep_d.index = prep_d.index.map(lambda x: x.replace(year=y))
        df_P = pd.concat([df_P, prep_d])

        # temp
        temp_m = temp_no29Feb[temp_no29Feb.index.month == m]
        temp_d = temp_m[temp_m.index.year == selected_year[i,j]]
        temp_d.index = temp_d.index.map(lambda x: x.replace(year=y))
        df_T = pd.concat([df_T, temp_d])

    # Put back 29Feb using 28Feb data
    df_P = df_P.reindex(gen_rng, method='backfill')
    df_T = df_T.reindex(gen_rng, method='backfill')
    df_P.to_csv(os.path.join(scenario_path, "R{}_BCWth_D_Prec_cm.csv".format(i)))
    df_T.to_csv(os.path.join(scenario_path, "R{}_BCWth_D_Tavg_degC.csv".format(i)))

#%% =============================================================================
# Set P T range
# =============================================================================
def interpolateSc(df, PorT):
    df = df.set_index(["Period"])
    df = df.loc[['2030s', '2050s', '2070s', '2090s'],:]
    df.index = [int(i[0:4]) for i in df.index]
    df = pd.concat(
        [pd.DataFrame(index=np.arange(2021,2101,1).astype(int)), df], axis=1)
    if PorT == "P":
        df.columns = ["P"]
        df.loc[2021, "P"] = 1
    elif PorT == "T":
        df.columns = ["T"]
        df.loc[2021, "T"] = 0
    df = df.interpolate(method='linear', axis=0)
    df.index = pd.to_datetime(df.index, format="%Y")
    df = df.reindex(gen_rng, method='ffill')
    return df
cli_rng_prep = pd.read_csv(
    os.path.join(gcm_path, "ClimateRange_P_Y_CYLin.csv"), index_col=["RCP"])
cli_rng_temp = pd.read_csv(
    os.path.join(gcm_path, "ClimateRange_T_Y_CYLin.csv"), index_col=["RCP"])
gen_rng = pd.date_range("2021/1/1", "2100/12/31")

num_realization = 20
# NumScFile = 5*5*2*20 = 1000 (pq, tq, RCP, num_realization)
for rcp in ["rcp26", "rcp85"]:
    for pq in quantile_list:
        for tq in quantile_list:
            PRatio = interpolateSc(cli_rng_prep.loc[rcp, ["Period", pq]], "P")
            TDelta = interpolateSc(cli_rng_temp.loc[rcp, ["Period", tq]], "T")
            for r in tqdm(range(num_realization),
                          desc="{}_P{}_T{}".format(rcp, pq, tq)):
                # prec
                df_P = pd.read_csv(
                    os.path.join(scenario_path, "R{}_BCWth_D_Prec_cm.csv".format(r)),
                    parse_dates=True, index_col=0)
                df_P = pd.DataFrame(np.multiply(df_P, PRatio),
                                    index=df_P.index, columns=df_P.columns)
                # # temp
                # df_T = pd.read_csv(
                #     os.path.join(scenario_path, "R{}_BCWth_D_Tavg_degC.csv".format(r)),
                #     parse_dates=True, index_col=0)
                # df_T = pd.DataFrame(np.add(df_T, TDelta),
                #                     index=df_T.index, columns=df_T.columns)

                # # Output Wth
                # df_P.to_csv(
                #     os.path.join(scenario_path,
                #                  "R{}_BCWth_D_Prec_cm_{}_P{}_T{}.csv".format(r, rcp, pq, tq)))
                # df_T.to_csv(
                #     os.path.join(scenario_path,
                #                  "R{}_BCWth_D_Tavg_degC_{}_P{}_T{}.csv".format(r, rcp, pq, tq)))
                # # pet
                # df_Pet = pd.DataFrame(index = df_T.index)
                # ID = ["S1","S2","S3","C1","C2","G"]
                # Lats = [47.416,46.814,46.622,47.145,46.839,46.682]

                # def calPet(Tt, lat):
                #     return HydroCNHS.calPEt_Hamon(Tt = Tt, Lat = lat, StartDate = "2021/1/1")
                # QParel = Parallel(n_jobs = -2)( delayed(calPet)(df_T[v], Lats[i]) for i, v in enumerate(ID) )
                # for i, v in enumerate(ID):
                #     df_Pet[v] = QParel[i]
                # df_Pet.to_csv(
                #     os.path.join(scenario_path,
                #                  "R{}_BCWth_D_Pet_cm_{}_P{}_T{}.csv".format(r, rcp, pq, tq)))

                # Database
                df_P_db = df_P.resample("MS").mean()
                df_P_db.index = [i+relativedelta(months=2) for i in df_P_db.index]
                mask_db = [True if i.month in [1,2,3,4,5,6,7,8] else False for i in df_P_db.index]
                df_P_db = df_P_db[mask_db].resample("YS").mean()
                df_P_db["TotalS"] = ( df_P_db["S1"]*83014.25 + df_P_db["S1"]*11601.47 + df_P_db["S1"]*28016.2 )  \
                                    /(83014.25+11601.47+28016.2)
                df_P_db.index = df_P_db.index.year
                df_P_db[["TotalS"]].to_csv(
                    os.path.join(scenario_path,
                                 "R{}_Database_2021_2100_{}_P{}_T{}.csv".format(r, rcp, pq, tq)))




#%% ===========================================================================
# Inflow Simulation
# =============================================================================
gen_rng = pd.date_range("2021/1/1", "2100/12/31")
num_realization = 20
for rcp in ["rcp26", "rcp85"]:
    for pq in quantile_list:
        for tq in quantile_list:
            for r in tqdm(range(num_realization), desc="{}_P{}_T{}".format(rcp, pq, tq)):
                prec = pd.read_csv(
                    os.path.join(
                        scenario_path, "R{}_BCWth_D_Prec_cm_{}_P{}_T{}.csv".format(r, rcp, pq, tq)),
                        parse_dates=True, index_col=[0]).to_dict(orient='list')
                temp = pd.read_csv(
                    os.path.join(
                        scenario_path, "R{}_BCWth_D_Tavg_degC_{}_P{}_T{}.csv".format(r, rcp, pq, tq)),
                        parse_dates=True, index_col=[0]).to_dict(orient='list')
                pet = pd.read_csv(
                    os.path.join(
                        scenario_path, "R{}_BCWth_D_Pet_cm_{}_P{}_T{}.csv".format(r, rcp, pq, tq)),
                    parse_dates=True, index_col=[0]).to_dict(orient='list')

                def calInflow(s, temp, prec, pet):
                    modelname = os.path.join(path, r"NewModel\EMRs_Output",
                                             "{}_KGE_best.yaml".format(s))
                    model_dict = HydroCNHS.load_model(modelname)
                    model_dict["WaterSystem"]["StartDate"] = "2021/1/1"
                    model_dict["WaterSystem"]["EndDate"] = "2100/12/31"
                    model_dict["WaterSystem"]["DataLength"] = 29219
                    model = HydroCNHS.Model(model_dict, modelname)
                    Q = model.run(temp, prec, pet, assigned_Q={}, disable=True)
                    return Q[s]
                QParel = Parallel(n_jobs=-2)(
                    delayed(calInflow)(s, temp, prec, pet) for s in ["S1","S2","S3"] )
                df = pd.DataFrame()
                for i, QS in enumerate(QParel):
                    df["R"+str(i+1)] = QS
                df.index = gen_rng
                df.to_csv(os.path.join(scenario_path, "R{}_Inflow_D_R1R2R3_cms_{}_P{}_T{}.csv".format(r, rcp, pq, tq)))


#%% ===========================================================================
# Reservoir Release Simulation
# =============================================================================
quantile_list = ["10","30","50","70","90"]
num_realization = 20
for rcp in ["rcp26", "rcp85"]:
    for pq in quantile_list:
        for tq in tqdm(quantile_list, desc = "{}_P{}".format(rcp, pq)):
            # for r in tqdm(range(num_realization), desc = "{}_P{}_T{}".format(rcp, pq, tq)):
            def calRes(r, rcp, pq, tq):
                InflowS = pd.read_csv(
                    os.path.join(scenario_path, "R{}_Inflow_D_R1R2R3_cms_{}_P{}_T{}.csv".format(r, rcp, pq, tq)),
                                 parse_dates=True, index_col=[0])
                Database = pd.read_csv(
                    os.path.join(scenario_path, "R{}_Database_2021_2100_{}_P{}_T{}.csv".format(r, rcp, pq, tq)),
                                 index_col=0)
                Dates = pd.date_range("2021/1/1", "2100/12/31", freq="M")

                SimR1_Res = []
                SimR1_S = [668.584161735199]
                SimR2_Res = []
                SimR2_S = [17.4044287624]
                SimR3_Res = []
                SimR3_S = [164.3244507248]
                for CurrentDate in Dates:
                    mask = [True if i.month == CurrentDate.month and i.year == CurrentDate.year else False for i in InflowS.index]
                    if CurrentDate.month == 12:
                        if CurrentDate.year == 2100:
                            mask_fore = mask
                        else:
                            mask_fore = [True if i.month == 1 and i.year == CurrentDate.year + 1 else False for i in InflowS.index]
                    else:
                        mask_fore = [True if i.month == CurrentDate.month + 1 and i.year == CurrentDate.year else False for i in InflowS.index]
                    Inflows_R1 = InflowS[mask]["R1"]
                    Inflows_R2 = InflowS[mask]["R2"]
                    Inflows_R3 = InflowS[mask]["R3"]
                    Inflows_next_R3 = InflowS[mask_fore]["R3"]
                    S_pre_R1 = SimR1_S[-1]
                    S_pre_R2 = SimR2_S[-1]
                    S_pre_R3 = SimR3_S[-1]

                    x = Database.loc[CurrentDate.year, "TotalS"]
                    if x <= 0.583:
                        DroughtYear = True
                    else:
                        DroughtYear = False
                    res, newS = R1(CurrentDate, Inflows_R1, S_pre_R1, DroughtYear, Future = True)
                    SimR1_Res.append(res)
                    SimR1_S.append(newS)
                    res, newS = R2(CurrentDate, Inflows_R2, S_pre_R2, Future = True)
                    SimR2_Res.append(res)
                    SimR2_S.append(newS)
                    res, newS = R3(CurrentDate, Inflows_R3, Inflows_next_R3, S_pre_R3, Future = True)
                    SimR3_Res.append(res)
                    SimR3_S.append(newS)
                SimResDf = pd.DataFrame(index = Dates)
                SimResDf["R1"] = SimR1_Res
                SimResDf["R2"] = SimR2_Res
                SimResDf["R3"] = SimR3_Res
                SimResDf_S = pd.DataFrame(index = Dates)
                SimResDf_S["R1S"] = SimR1_S[1:]
                SimResDf_S["R2S"] = SimR2_S[1:]
                SimResDf_S["R3S"] = SimR3_S[1:]

                # To Daily
                SimResDf = SimResDf.resample("MS").mean()
                SimResDf.index = list(SimResDf.index[:-1]) + [datetime(2100, 12, 31)]
                SimResDf = SimResDf.resample("D").ffill()
                SimResDf.to_csv(os.path.join(scenario_path, "R{}_Release_D_R1R2R3_cms_{}_P{}_T{}.csv".format(r, rcp, pq, tq)))
                return None

            QParel = Parallel(n_jobs = -2)(
                delayed(calRes)(r, rcp, pq, tq) for r in range(num_realization) )

#%% Reservoir functions
def R1(CurrentDate, Inflows, S_pre, DroughtYear, Future = False):
    # Determine MaxSP & MinSP and assign Capacity.
    if Future:
        y = 2014
    else:
        y = CurrentDate.year
    m = CurrentDate.month
    DoM = CurrentDate.daysinmonth

    # New obv min
    MinFlow_set1 = [1.96, 2.11, 0.18, 2.97, 15.47, 40.51,
                85.95 , 88.56, 24.25, 7.48, 2.10, 2.02]           # cms ~1996
    MinFlow_set2 = [5.16, 6.95, 6.77, 8.43, 15.47, 40.51,
                    85.95 , 88.56, 24.25, 7.48, 7.22, 6.22]           # cms ~1996

    # MinFlow_set1 = [3.4, 2.2,  2, 5, 16.5, 30,
    #                   0,   0,  0, 0,  2.1, 2]           # cms ~1996
    # MinFlow_set2 = [8.17, 7.7, 7.8, 11, 16.5, 30,
    #                     0,   0,   0,  0,  9.6, 9.5]      # cms 1996~
    MaxSP = [0.8, 0.896, 0.914, 0.959,    1,    1,
               1,     1,     1,  0.75, 0.8, 0.75]
    MinSP = [0.108, 0.156, 0.255, 0.407, 0.586, 0.523,
             0.307, 0.092, 0.045, 0.038, 0.047, 0.073]
    Capacity = 1028.353823                          # km2-m

    InflowAmount = np.sum(np.array(Inflows)*0.0864)     # km2-m
    Inflow = np.mean(Inflows)
    NewS = S_pre + InflowAmount

    def toCMS(x):
        return x*11.5741/DoM    # x*1000000/(86400*DoM)
    def toV(x):
        return x*0.0864*DoM     # x*(86400*DoM)/1000000

    def checkS(NewS, Res, m):
        MaxS = Capacity*MaxSP[m-1]
        MinS = Capacity*MinSP[m-1]
        if y < 1996:
            MinFlow = MinFlow_set1
        else:
            MinFlow = MinFlow_set2
        if NewS > MaxS:
            Res += toCMS(NewS - MaxS)
            NewS = MaxS
        if NewS < MinS:
            dS = MinS - NewS
            if dS >= 0:
                Res = max(Res-toCMS(dS), min(Inflow, MinFlow[m-1]))
                NewS = S_pre + InflowAmount - toV(Res)

            else:
                NewS += toV(Res)
                Res = 0
        return Res, NewS

    #--- 6
    # S_pre to identify drought
    # Inflow => release storage or not.
    # Otherwise accumulate Storage
    if m == 6:  # No MinS
        if S_pre/Capacity <= 0.76:  # Drought and no water.
            Res = max(0, 110 - Inflow)
            NewS = NewS - toV(Res)
        else:
            if Inflow >= 100:   # Accumulate S
                Res = 85
            else:   # Not enough inflow but has storage to release.
                Res = max(0, 150 - Inflow)
                NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)

    #--- 7
    # Control by three constant under three conditions.
    # (1) Full storage => what inflow
    # (2) Not full storage
    elif m == 7:
        SP = S_pre/Capacity
        if SP <= 0.9:
            Res = 110
        else:
            if Inflow > 55:
                Res = 102
            elif Inflow > 39:
                Res = 110
            else:
                Res = 132
        NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)

    # 8
    # Constant S_pre - S limited by Max Res and Min S
    elif m == 8:
        SDiff = 0.302
        NewS = S_pre - SDiff*Capacity
        Res = toCMS(InflowAmount + S_pre - NewS)
        if Res > 135:
            Res = 135
            NewS = NewS - toV(Res)
        # Min S
        #Res, NewS = checkS(NewS, Res, m)

    # 9
    elif m == 9:
        if DroughtYear:
            Res = 30
            NewS = NewS - toV(Res)
        else:
            Res = 55
            NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)

    # 10
    elif m == 10:
        if S_pre <= 0.2*Capacity:
            Res = 8
            NewS = NewS - toV(Res)
        else:
            Res = 22
            NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)    # In case too much inflow

    #--- 11-3 + 4,5 Flood control
    # Meet minimum flow, no storage control, limited by Max S.
    # MinFlow 11-3: 1988, 2006
    # MinFlow  4-5: 1988, 2001
    # No MinS
    else:   # if m in [11,12,1,2,3, 4,5]:
        if y < 1997:
            MF = MinFlow_set1[m-1]
        else:
            MF = MinFlow_set2[m-1]

        Res = min(MF, toCMS(InflowAmount))
        NewS = NewS - toV(Res)
    Res, NewS = checkS(NewS, Res, m)
    return Res, NewS

def R2(CurrentDate, Inflows, S_pre, Future = False):
    MinFlow_set1 = [1.6, 1.7, 1.9, 2, 6, 5,
                    4. , 1.8, 3.3, 1.5, 1, 1.3]           # cms ~1996
    MinFlow_set2 = [1.6, 1.7, 1.9, 2, 6, 5,
                    5.5 , 5, 3.3, 1.5, 1, 1.3]           # cms 1996~
    MaxSP = [0.8, 1, 1, 1, 1, 1,
             1, 0.9, 0.7 , 0.8, 1, 1]
    MinSP = [0.06, 0.06, 0.06, 0.15, 0.65, 0.75,
             0.55, 0.3, 0.15, 0.06, 0.06, 0.06]
    Capacity = 41.90137863

    def toCMS(x):
        return x*11.5741/DoM    # x*1000000/(86400*DoM)
    def toV(x):
        return x*0.0864*DoM     # x*(86400*DoM)/1000000

    pars = [ [0.6321304183940335,  14.59795659964575,  -0.04135652906787122,  -0.5899534357236593],
             [0.6310875353498168,  10.428258949602393,  -0.028264948274726243,  -0.03335240639099488],
             [0.4823489122163609,  1.1903321412442667,  -0.0015855877065889849,  1.4778645268068662],
             [0.8642808758537905,  10.646158052724626,  -0.0491654969769099,  -0.031015896047420055],
             [-0.08966075245318983,  -4.326419306045811,  0.0016941213094068264,  4.577045379730967],
             [-0.39559333485389025,  -10.287527379746674,  0.008344080391207487,  11.58773739161099],
             [-0.4614050949711562,  -8.96088018489783,  0.0027458988714777724,  13.191646660463716],
             #down[0.42320496864982887, -8.36171921997348, -0.01422777376881184, 11.929198159625129],
             [4.286901128880376,  0.7606178123762414,  -0.10471162453819798,  2.008876320827349],
             [3.998474270565911,  12.513130493341999,  -0.17328895981365228,  -1.7760441510517153],
             [1.850460821965776,  20.23227131674294,  -0.11944460411198518,  -2.7546007935567194],
             [0.614328433316427,  19.76714999040931,  -0.046928703091247734,  -1.1668247398369687],
             [0.11240020929163057,  12.954043963261553,  -0.014018862635680468,  1.346825137921167]]

    if Future:
        y = 2014
    else:
        y = CurrentDate.year
    m = CurrentDate.month
    DoM = CurrentDate.daysinmonth
    InflowAmount = np.sum(np.array(Inflows)*0.0864)     # km2-m
    Inflow = np.mean(Inflows)

    IS = Inflow*S_pre

    NewS = S_pre + InflowAmount
    a, b, c, d = pars[m-1]
    res = a*Inflow + b*S_pre/Capacity + c*IS + d

    # Constraint on Res
    if y < 1996:
        MinFlow_set = MinFlow_set1
    else:
        MinFlow_set = MinFlow_set2
    res = max(res, MinFlow_set[m-1])

    # if m == 5:
    #     res = res - 5
    # elif m == 6:
    #     res = res - 4
    # elif m == 7:
    #     res = res - 2
    #     if res < 4:
    #         res = 4

    NewS = NewS - toV(res)
    # Constraint on S
    MaxS = Capacity*MaxSP[m-1]
    MinS = Capacity*MinSP[m-1]

    if NewS > MaxS:
        res += toCMS(NewS - MaxS)
        NewS = MaxS
    if NewS < MinS:
        dS = MinS - NewS
        if dS >= 0:
            res = max(res-toCMS(dS), min(Inflow, MinFlow_set[m-1]))
            NewS = S_pre + InflowAmount - toV(res)
        else:
            NewS += toV(res)
            res = 0

    return res, NewS

def R3(CurrentDate, Inflows, Inflows_next, S_pre, Future = False):
    # Determine MaxSP & MinSP and assign Capacity.
    if Future:
        y = 2014
    else:
        y = CurrentDate.year
    m = CurrentDate.month
    DoM = CurrentDate.daysinmonth

    # MinFlow_set1 = [0.5, 0.8,  0.7, 0.83, 4.6, 0,
    #                   0,   0,  0, 0,  0.9, 0.6]           # cms ~1996
    # MinFlow_set2 = [2, 1.6, 1.2, 2, 4.6, 0,
    #                    0,   0,   0,  0, 2.16, 2.54]      # cms 1996~

    MinFlow_set1 = [0.5, 0.8,  0.7, 0.83, 4.6, 6.49,
                  11.23,   14.47,  29.34, 2.74,  0.9, 0.6]    # cms ~1996
    MinFlow_set2 = [1.16, 1.1, 1.1, 2.31, 4.45, 8.42,
                   13.45,   14.47,   38.67,  6.80, 2, 2]      # cms 1996~

    MaxSP = [0.82, 0.85,  0.9, 0.97,    1,    1,
                1, 0.98, 0.75,  0.7, 0.75, 0.75]
    MinSP = [0.168, 0.206, 0.298, 0.467, 0.63 , 0.697,
             0.674, 0.436, 0.101, 0.044, 0.081, 0.112]


    Capacity = 244.2294074                          # km2-m
    InflowAmount = np.sum(np.array(Inflows)*0.0864)     # km2-m
    Inflow = np.mean(Inflows)
    Inflow_next = np.mean(Inflows_next)
    NewS = S_pre + InflowAmount

    def toCMS(x):
        return x*11.5741/DoM    # x*1000000/(86400*DoM)
    def toV(x):
        return x*0.0864*DoM     # x*(86400*DoM)/1000000

    def checkS(NewS, Res, m):
        MaxS = Capacity*MaxSP[m-1]
        MinS = Capacity*MinSP[m-1]
        if y < 1996:
            MinFlow = MinFlow_set1
        else:
            MinFlow = MinFlow_set2
        if NewS > MaxS:
            Res += toCMS(NewS - MaxS)
            NewS = MaxS
        if NewS < MinS:
            dS = MinS - NewS
            if dS >= 0:
                Res = max(Res-toCMS(dS), min(Inflow, MinFlow[m-1]))
                NewS = S_pre + InflowAmount - toV(Res)
            else:
                NewS += toV(Res)
                Res = 0
        return Res, NewS

    #--- 4
    # Inflow_next => prepaution release
    # Minimun release -> MaxS
    if m == 4:
        if Inflow_next > 50 and S_pre/Capacity > 0.5:
            Res = 37.5
        elif Inflow_next > 33.8 and S_pre/Capacity > 0.5: #cms
            Res = 20
        elif y < 1996:
            if S_pre/Capacity < 0.5:
                Res = 0
            else:
                Res = 4
        else:
            Res = 3.3
        if y >= 2005:
            MaxSP[m-1] = 0.90
        NewS = NewS - toV(Res)

    #--- 5
    # Inflow_next => prepaution release
    # Minimun release -> MaxS
    if m == 5:
        if Inflow_next > 40 and S_pre/Capacity > 0.5:
            Res = 0 # Let MaxS decide
            MaxSP[m-1] = 0.8
        else:
            Res = 5
        NewS = NewS - toV(Res)
    #--- 6
    # S_pre to identify drought
    # Inflow => release storage or not.
    # Otherwise accumulate Storage
    if m == 6:  # No MinS
        Res = 6.5
        NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)

    #--- 7
    # Control by three constant under three conditions.
    # (1) Full storage => what inflow
    # (2) Not full storage
    elif m == 7:
        SP = S_pre/Capacity
        if SP <= 0.9:
            Res = 13.5
        else:
            Res = 20
        NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)

    # 8
    # Constant S_pre - S limited by Max Res and Min S
    elif m == 8:
        SP = S_pre/Capacity
        if Inflow > 10 or SP < 0.75:
            Res = 18
        else:
            Res = 23
        NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)

    # 9
    # No pattern
    elif m == 9:
        Res = 49.15 # 1988-2013 mean
        NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)

    # 10
    # No pattern
    elif m == 10:
        Res = 14.5
        NewS = NewS - toV(Res)
        #Res, NewS = checkS(NewS, Res, m)    # In case too much inflow

    #--- 11-3 + 4,5 Flood control
    # Meet minimum flow, no storage control, limited by Max S.
    # MinFlow 11-3: 1988, 2006
    # MinFlow  4-5: 1988, 2001
    # No MinS
    else:   # if m in [11,12,1,2,3, 4,5]:
        if y < 1997:
            MF = MinFlow_set1[m-1]
        else:
            MF = MinFlow_set2[m-1]

        Res = min(MF, toCMS(InflowAmount))
        NewS = NewS - toV(Res)
    Res, NewS = checkS(NewS, Res, m)
    return Res, NewS
