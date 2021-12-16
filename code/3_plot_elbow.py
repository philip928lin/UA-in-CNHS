import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"\Explained_var_for_abmEMR.csv", index_col=[0])

# ['Learning_Linear_hydro1', 'Learning_Quadratic_hydro1',
# 'Adaptive_Linear_hydro1', 'Adaptive_Quadratic_hydro1',
# 'Static_Linear_hydro1', 'Learning_Linear_hydro2',
# 'Learning_Quadratic_hydro2', 'Adaptive_Linear_hydro2',
# 'Adaptive_Quadratic_hydro2', 'Static_Linear_hydro2']

name_dict = {("Learning", "Linear"): "$M_{L,L}$",
             ("Learning", "Quadratic"): "$M_{L,Q}$",
             ("Adaptive", "Linear"): "$M_{A,L}$",
             ("Adaptive", "Quadratic"): "$M_{A,Q}$",
             ("Static", "Linear"): "$M_{S}$"}
names = ["HydroEMR1-$M_{L,L}$","HydroEMR1-$M_{L,Q}$","HydroEMR1-$M_{A,L}$",
         "HydroEMR1-$M_{A,Q}$","HydroEMR1-$M_{S}$",
         "HydroEMR2-$M_{L,L}$","HydroEMR2-$M_{L,Q}$","HydroEMR2-$M_{A,L}$",
         "HydroEMR2-$M_{A,Q}$","HydroEMR2-$M_{S}$"]
df.columns = names

marker_list = [".", "o", "v", "^", "*", "s", "x", "d", "+", "1"]
fig, ax = plt.subplots()
for i, v in enumerate(names):
    ax.plot(df[v], label=v, marker=marker_list[i])
ax.axvline(4, color="black", ls="--", lw=1)
ax.legend(ncol=2, fontsize=8)
ax.set_ylabel("Explained variance")
ax.set_ylim([0,0.9])
ax.set_xlabel("Number of Kmeans clusters")