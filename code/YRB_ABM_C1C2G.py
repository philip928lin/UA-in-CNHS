import os
import pickle

pc = ""
path = r"".format(pc)
input_path = os.path.join(path, "\Input")


with open(os.path.join(input_path, "YRB_cali_abm_assigned.pickle"),
          "rb") as file:
    obv_assigned = pickle.load(file)


class base():
    def __init__(self, **kwargs):
        # Agent
        # name=agG, config=ag_config, start_date=start_date,
        # data_length=data_length, data_collector=dc, rn_gen=rn_gen
        # and assign dm or None

        # dm
        # start_date=start_date, data_length=data_length, abm=abm,
        # data_collector=dc, rn_gen=rn_gen
        for key in kwargs:  # Load back all the previous class attributions.
            setattr(self, key, kwargs[key])


class ResDam_AgType():
    def __init__(self, name, config, start_date, data_length,
                 data_collector, rn_gen):
        #super().__init__(**kwargs)
        self.inputs = config["Inputs"]
        self.assigned_actions = obv_assigned[name]
        print("Initialize reservoir agent: {}".format(name))

    def act(self, Q, outet, agent_dict, current_date, t):

        factor = self.inputs["Links"][outet]

        # Release (factor should be 1)
        if factor < 0:
            print("Something is not right in ResDam agent.")
        elif factor > 0:
            # Q["SCOO"][t] is the resevoir inflow
            res_t = self.assigned_actions[t]
            action = res_t
            return action

class IrrDiv_AgType():
    def __init__(self, name, config, start_date, data_length,
                 data_collector, rn_gen):
        #super().__init__(**kwargs)
        self.name = name
        self.inputs = config["Inputs"]
        self.pars = config["Pars"]
        self.dc = data_collector
        self.dc.add_field(self.name, {})
        records = self.dc.get_field(self.name)
        records["DivReq"] = []
        records["Div"] = []
        records["Shortage_D"] = []
        records["Qup"] = []
        records["Qdown"] = []
        records["ReFlow"] = []
        records["Shortage_M"] = []
        self.pre_date = start_date
        self.data_length = data_length
        self.assigned_actions = obv_assigned[name]
        print("Initialize irrigation diversion agent: {}".format(name))

    def act(self, Q, outet, agent_dict, current_date, t):
        records = self.dc.get_field(self.name)
        data_length = self.data_length

        # Get factor
        factor = self.inputs["Links"][outet]
        # For parameterized (for calibration) factor.
        if isinstance(factor, list):
            factor = self.pars[factor[0]][factor[1]]

        # Compute actual diversion or return flow
        if factor < 0:  # Diversion
            # We consider monthly shortage. Therefore the daily shortage will
            # be carried over to the next day until the end of the month.
            if current_date.month != self.pre_date.month or t+1 == data_length:
                shortage_M = records["Shortage_D"][-1] \
                             / self.pre_date.days_in_month
                records["Shortage_M"].append(shortage_M)
                remain_Req = 0
            else:
                try:
                    remain_Req = records["Shortage_D"][-1]
                except:     # init
                    remain_Req = 0

            div_req_t = self.assigned_actions[t] + remain_Req
            min_flow = 3.53     # [cms]
            available_water_t = max(0, Q[outet][t] - min_flow)
            if div_req_t > available_water_t:
                shortage_t = div_req_t - available_water_t
                div_t = available_water_t
            else:
                div_t = div_req_t
                shortage_t = 0

            records["Qup"].append(Q[outet][t])
            records["Qdown"].append(Q[outet][t] - div_t)
            records["DivReq"].append(div_req_t)
            records["Div"].append(div_t)
            records["Shortage_D"].append(shortage_t)
            action = factor * div_t
            self.pre_date = current_date
        else:           # Return flow
            div_t = records["Div"][t]
            action = factor * div_t
            records["ReFlow"] = action
        return action

class IrrDiv_RWS_AgType():
    def __init__(self, name, config, start_date, data_length,
                 data_collector, rn_gen):
        #super().__init__(**kwargs)
        self.dc = data_collector
        self.ag_list = ['Roza', 'Wapato', 'Sunnyside']
        # wr_ratio = proratable water right / total water right.
        self.wr_ratio = [1, 0.533851525, 0.352633532]

        # Create record for each agent in the group.
        self.assigned_actions = {}
        for ag in self.ag_list:
            self.dc.add_field(ag, {})
            records = self.dc.get_field(ag)
            records["DivReq"] = []
            records["Div"] = []
            records["Shortage_D"] = []
            records["Qup"] = []
            records["Qdown"] = []
            records["ReFlow"] = []
            records["Shortage_M"] = []
            records["AccMDivReq"] = 0   # for redistribution
            self.assigned_actions[ag] = obv_assigned[ag]

        self.pre_date = start_date
        self.data_length = data_length
        print("Initialize irrigation diversion agent group: {}".format(name))


    def act(self, Q, outet, agent_dict, current_date, t):
        data_length = self.data_length
        wr_ratio = self.wr_ratio

        # Get factor
        factor = -1

        # We consider monthly shortage. Therefore the daily shortage will
        # be carried over to the next day until the end of the month.
        ### Collect diversion requests of all agents in the group.
        div_reqs_t = []
        div_reqs_t_no_remain = []
        div_reqs_acc = []
        for ag in self.ag_list:
            records = self.dc.get_field(ag)
            if current_date.month != self.pre_date.month or t+1 == data_length:
                shortage_M = records["Shortage_D"][-1] \
                             / self.pre_date.days_in_month
                records["Shortage_M"].append(shortage_M)
                records["AccMDivReq"] = 0
                remain_Req = 0
            else:
                try:
                    remain_Req = records["Shortage_D"][-1]
                except:     # init
                    remain_Req = 0
            div_dm = self.assigned_actions[ag][t]
            records["AccMDivReq"] += div_dm
            div_reqs_acc.append(records["AccMDivReq"])
            div_req_t = div_dm + remain_Req
            div_reqs_t.append(div_req_t)
            div_reqs_t_no_remain.append(div_dm)
            records["DivReq"].append(div_req_t)
            records["Qup"].append(Q[outet][t])

        ### Calculate actual total diversion as a group.
        total_div_req_t = sum(div_reqs_t)
        min_flow = 3.53     # [cms]
        available_water_t = max(0, Q[outet][t] - min_flow)
        if total_div_req_t > available_water_t:
            total_shortage_t = total_div_req_t - available_water_t
            total_div_t = available_water_t
        else:
            total_div_t = total_div_req_t
            total_shortage_t = 0

        ### Disaggregate group value into each agents in the group.
        p_reqs = [wr_ratio[i] * div_reqs_acc[i] for i in range(3)]
        total_p_reqs = sum(p_reqs)
        if total_p_reqs == 0:   # Avoid dividing zero.
            r_p_reqs = [0, 0, 0]
        else:
            r_p_reqs = [p_req / total_p_reqs for p_req in p_reqs]
        shortages_t = [total_shortage_t * r_p_req for r_p_req in r_p_reqs]


        # Redistribute exceed shortages_t
        # Need to make sure shortage does not exceed monthly accumulated req.
        if shortages_t[0] > div_reqs_acc[0]:  # Roza
            redistritute = shortages_t[0] - div_reqs_acc[0]
            shortages_t[0] = div_reqs_acc[0]
            # To Wapato & Sunnyside
            total_rr_reqs = div_reqs_acc[1] + div_reqs_acc[2]
            shortages_t[1] += (div_reqs_acc[1]/total_rr_reqs * redistritute)
            shortages_t[2] += (div_reqs_acc[2]/total_rr_reqs * redistritute)
        if shortages_t[1] > div_reqs_acc[1]:  # Wapato
            redistritute = shortages_t[1] - div_reqs_acc[1]
            shortages_t[1] = div_reqs_acc[1]
            # To Sunnyside
            shortages_t[2] += redistritute
        if shortages_t[2] > div_reqs_acc[2]:  # Sunnyside
            if abs(shortages_t[2]-div_reqs_acc[2]) <= 10**(-5):
                shortages_t[2] = div_reqs_acc[2]
            else:
                print("Error! shortage distribution.")



        ### Record agents values
        q_down = Q[outet][t] - total_div_t
        for i, ag in enumerate(self.ag_list):
            records = self.dc.get_field(ag)
            div_t = div_reqs_t[i] - shortages_t[i]
            records["Shortage_D"].append(shortages_t[i])
            records["Div"].append(div_t)
            records["Qdown"].append(q_down)

        action = factor * total_div_t
        self.pre_date = current_date
        # No return flow for this group.
        return action




        # Redistribute exceed shortages_t
        # if shortages_t[0] > r_p_reqs[0]:  # Roza
        #     redistritute = shortages_t[0] - r_p_reqs[0]
        #     shortages_t[0] = r_p_reqs[0]
        #     # To Wapato & Sunnyside
        #     total_rr_reqs = r_p_reqs[1] + r_p_reqs[2]
        #     shortages_t[1] += (r_p_reqs[1]/total_rr_reqs * redistritute)
        #     shortages_t[2] += (r_p_reqs[2]/total_rr_reqs * redistritute)
        # if shortages_t[1] > r_p_reqs[1]:  # Wapato
        #     redistritute = shortages_t[1] - r_p_reqs[1]
        #     shortages_t[1] = r_p_reqs[1]
        #     # To Sunnyside
        #     shortages_t[2] += redistritute
        # if shortages_t[2] > r_p_reqs[2]:  # Sunnyside
        #     print("Error! shortage distribution.")