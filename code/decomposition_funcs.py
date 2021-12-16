def get_EV(df):
    """E[Var(X|Y)]: Inputs"""
    df_copy = df.copy()
    df_copy["Model"] = [(hy,abm,bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_copy.index]
    df_EV = df_copy.groupby(["Model"]).var()
    df_EV["AgType"] = [(bt,at) for (hy,abm,bt,at) in df_EV.index]
    df_EV = df_EV.groupby(["AgType"]).mean()
    return df_EV

def get_VE(df):
    """Var(E[X|Y]): Equifinality"""
    df_copy = df.copy()
    df_copy["Model"] = [(hy,abm,bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_copy.index]
    df_VE = df_copy.groupby(["Model"]).mean()
    df_VE["AgType"] = [(bt,at) for (hy,abm,bt,at) in df_VE.index]
    df_VE = df_VE.groupby(["AgType"]).var() #+ 0.0001 # to plot in log scale
    return df_VE

def get_EVE(df):
    """E[Var(E[X|Y1,Y2])|Y1]: Wth internal variability; Y1: EMR,
    Y2: Realization"""
    df_copy = df.copy()
    df_copy["Model"] = [(hy,abm,bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_copy.index]
    df_copy["Realization"] = [r for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_copy.index]
    df_EVE = df_copy.groupby(["Model", "Realization"]).mean().reset_index()
    df_EVE = df_EVE.groupby(["Model"]).var()
    df_EVE["AgType"] = [(bt,at) for (hy,abm,bt,at) in df_EVE.index]
    df_EVE = df_EVE.groupby(["AgType"]).mean() #+ 0.0001 # to plot in log scale
    df_EVE = df_EVE.drop("Realization", axis = 1)
    return df_EVE

def get_EV2(df):
    """E[Var(X|Y1,Y2)]: Other uncertainty, climate GCM (RCP) uncertainty;
    Y1: EMR, Y2: Realization"""
    df_copy = df.copy()
    df_copy["Model"] = [(hy,abm,bt,at) for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_copy.index]
    df_copy["Realization"] = [r for (rcp,prec_q,temp_q,r,hy,abm,bt,at) in df_copy.index]
    df_EV = df_copy.groupby(["Model", "Realization"]).var().reset_index()
    df_EV["AgType"] = [(bt,at) for (hy,abm,bt,at) in df_EV["Model"]]
    df_EV = df_EV.groupby(["AgType"]).mean() #+ 0.0001 # to plot in log scale
    df_EV = df_EV.drop("Realization", axis = 1)
    return df_EV
