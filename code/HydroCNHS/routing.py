#%%
# Routing model for Land Surface model outputs based on Lohmann routing model
# by Chung-Yi Lin @ Lehigh University (philip928lin@gmail.com) 
# based on the code from Ethan Yang @ Lehigh University (yey217@lehigh.edu)
# 2021/02/05

import numpy as np
from scipy.stats import gamma


def form_UH_Lohmann(inputs, routing_pars, force_ingrid_off=False):
    """Derive HRU's UH at the (watershed) outlet.
    We seperately calculate in-grid UH_IG and river routing UH_RR and combine
    them into HRU's UH.

    Args:
        inputs (dict): Inputs dictionary containing 
            FlowLength [m] Travel distence of flow between two outlets [float]
            and InstreamControl [bool].
        routing_pars (dict): Four parameters for routing: GShape, GRate, Velo,
            Diff [float]
        force_ingrid_off (bool): If True, then in-grid routing will be turned
            off by force. Default False.
    """
    flow_len = inputs["FlowLength"]
    instream_control = inputs["InstreamControl"]
    
    #----- Base Time for in-grid (watershed subunit) UH and river/channel
    # routing UH --------------------------------------------------------------
    # In-grid routing
    T_IG = 12				# [day] Base time for in-grid UH 
    # River routing 
    T_RR = 96				# [day] Base time for river routing UH 
    dT_sec = 3600			# [sec] Time step in second for solving 
                            ##Saint-Venant equation. This will affect Tmax_hr.
    Tmax_hr = T_RR * 24		# [hr] Base time of river routing UH in hour
                            ##because dT_sec is for an hour
    Tgr_hr = 48 * 50		# [hr] Base time for Green function values
    #--------------------------------------------------------------------------
    
    #----- In-grid routing UH (daily) represented by Gamma distribution -------
    UH_IG = np.zeros(T_IG)
    if instream_control or force_ingrid_off:
        UH_IG[0] = 1
    elif (routing_pars.get("GShape") is None 
          and routing_pars.get("GRate") is None):
        UH_IG[0] = 1
    else:
        Shape = routing_pars["GShape"]
        Rate = routing_pars["GRate"]
        if Rate <= 0.0001: Rate = 0.0001    # Since we cannot divide zero.
        for i in range(T_IG):
            # x-axis is in hr unit. We calculate in daily time step.
            UH_IG[i] = gamma.cdf(24 * (i + 1), a=Shape, loc=0, scale=1/Rate) \
                        - gamma.cdf(24 * i, a=Shape, loc=0, scale=1/Rate)
    #--------------------------------------------------------------------------

    #----- Derive Daily River Impulse Response Function (Green's function) ----
    UH_RR = np.zeros(T_RR)
    if flow_len == 0:
        UH_RR[0] = 1
    else:
        Velo = routing_pars["Velo"] 
        Diff = routing_pars["Diff"]

        # Calculate h(x, t)
        t = 0
        UH_RRm = np.zeros(Tgr_hr) # h() in hour
        for k in range(Tgr_hr):
            t = t + dT_sec
            pot = ((Velo * t - flow_len) ** 2) / (4 * Diff * t)
            if pot <= 69:		    # e^(-69) = E-30 a cut-off threshold								
                H = flow_len /(2 * t * (np.pi * t * Diff)**0.5) * np.exp(-pot)  
            else:
                H = 0
            UH_RRm[k] = H

        if sum(UH_RRm) == 0:
            UH_RRm[0] = 1.0
        else:
            UH_RRm = UH_RRm / sum(UH_RRm)     # Force UH_RRm to sum to 1
        
        FR = np.zeros((Tmax_hr+23, Tmax_hr-1))   
        for i in range(Tmax_hr-1):
            FR[:,i] = np.pad(UH_RRm, (i+1, 23), 'constant',
                             constant_values=(0, 0))[:Tmax_hr+23]
        FR = np.sum(FR, axis = 1)/24
        # Lag 24 hrs
        FR = FR[:Tmax_hr] - np.pad(FR, (24, 0), 'constant',
                                   constant_values=(0, 0))[:Tmax_hr]
        
        # Aggregate to daily UH
        for t in range(T_RR):
            UH_RR[t] = sum(FR[(24*(t+1)-24):(24*(t+1)-1)])

	#----- Combined UH_IG and UH_RR for HRU's response at the (watershed)
    # outlet ------------------------------------------------------------------
    UH_direct = np.zeros(T_IG + T_RR - 1)   # Convolute total time step [day]
    for k in range (0, T_IG):
        for u in range (0, T_RR):
            UH_direct[k+u] = UH_direct[k+u] + UH_IG[k] * UH_RR[u]
    UH_direct = UH_direct/sum(UH_direct)
    UH_direct = np.trim_zeros(UH_direct, 'b')
    return UH_direct

def run_step_Lohmann(routing_outlet, routing, UH_Lohmann, Q, Q_LSM, t):
    """Calculate a single time step routing for the entire basin.
    Args:
        routing_outlet (str): routing node.
        routing (dict): Sub-model dictionary from your model.yaml file.
        UH_Lohmann (dict): Contain all pre-formed UH for all connections
            between gauged outlets and its upstream outlets.
            e.g. {(subbasin, gaugedoutlet): UH_direct}
        Q (dict): Contain all updated Q (array) for each outlet. 
        Q_LSM (dict): Contain all unupdated Q (array) for each outlet.
        t (int): Index of current time step (day).

    Returns:
        [dict]: Update Qt for routing.
    """   
    Qt = None
    ro = routing_outlet
    Qresult = 0
    Subbasin = list(routing[ro].keys())
    for sb in Subbasin:
        l = len(UH_Lohmann[(sb, ro)]) - 1
        # t+1 is length, t is index.
        UH = UH_Lohmann[(sb, ro)][0 : min(t + 1, l) ]
        if ro == sb:
            # Q[ro] is routed Q. We need to use unrouted Q (Q_LSM) to run the
            # routing.
            Q_reverse = np.flip(Q_LSM[sb][ max(t-(l-1), 0) : t+1])
        else:
            Q_reverse = np.flip(Q[sb][ max(t-(l-1), 0) : t+1])
        Qresult += np.sum(UH * Q_reverse)
    Qt = Qresult         # Store the result for time t
    return Qt

def run_step_Lohmann_convey(routing_outlet, routing, UH_Lohmann_convey,
                            Q_convey, t):
    """Calculate a single time step routing for the entire basin.
    Args:
        routing_outlet (str): routing node.
        routing (dict): Sub-model dictionary from your model.yaml file.
        UH_Lohmann_convey (dict): Contain pre-formed UH for all connections
            between gauged outlets and its upstream conveyed outlets (no 
            in-grid routing).
            e.g. {(subbasin, gaugedoutlet): UH_direct}
        Q_convey (dict): Contain conveyed water for its destinetion node. 
        t (int): Index of current time step (day).

    Returns:
        [dict]: Update Qt for routing.
    """   
    Qt = None
    ro = routing_outlet
    Qresult = 0
    Subbasin = list(routing[ro].keys())
    for sb in Subbasin:
        uh_convey = UH_Lohmann_convey.get((sb, ro))
        if uh_convey is not None:
            l = len(uh_convey) - 1
            # t+1 is length, t is index.
            UH = uh_convey[0 : min(t + 1, l)]
            Q_reverse = np.flip(Q_convey[sb][ max(t-(l-1), 0) : t+1])
            Qresult += np.sum(UH * Q_reverse)
    Qt = Qresult         # Store the result for time t
    return Qt