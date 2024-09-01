import numpy as np
from nsoran.base.ns_env import NsOranEnv 
from gymnasium import spaces

class TrafficSteeringEnv(NsOranEnv):
    def __init__(self, ns3_path:str, scenario_configuration:dict, output_folder:str, optimized:bool):
        super().__init__(ns3_path=ns3_path, scenario='scenario-one', scenario_configuration=scenario_configuration,
                         output_folder=output_folder, optimized=optimized,
                         control_header = ['timestamp','ueId','nrCellId'], log_file='TsActions.txt', control_file='ts_actions_for_ns3.csv')
        # These features can be hardcoded since they are specific for the use case
        self.columns_state = ['RRU.PrbUsedDl', 'L3 serving SINR', 'DRB.MeanActiveUeDl', 
                              'TB.TotNbrDlInitial.Qpsk', 'TB.TotNbrDlInitial.16Qam', 
                              'TB.TotNbrDlInitial.64Qam', 'TB.TotNbrDlInitial']

        self.columns_reward = ['DRB.UEThpDl.UEID']
        # TODO refine a little bit better the bounds
        self.observation_space = spaces.Box(shape=(len(self.columns_state),), low=-np.inf, high=np.inf, dtype=np.float64) 
        # In the traffic steering use case, the action is a combination between 
        n_gnbs = 7  # scenario one has always 7 gnbs 
        n_actions_ue = 7 # each UE can connect to a gNB identified by ID (from 2 to 8), 0 is No Action
        self.action_space = spaces.MultiDiscrete([n_actions_ue] * self.scenario_configuration['ues'] *  n_gnbs)

    def _compute_action(self, action) -> list[tuple]:    
        # action from multidiscrete shall become a list of ueId, targetCell.
        # If a targetCell is 0, it means No Handover, thus we don't send it
        action_list = []
        for ueId, targetCellId in enumerate(action):
            if targetCellId != 0: # and 
                # Once we are in this condition, we need to transform the action from the one of gym to the one of ns-O-RAN
                action_list.append((ueId + 1, targetCellId + 2))

        return action_list

    def _fill_datalake_usecase(self):
        # We don't need fill_datalake_usecase in TS use case
        pass

    def _get_obs(self) -> list:
        ue_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_state)                          
        # 'TB.TOTNBRDLINITIAL.QPSK_RATIO', 'TB.TOTNBRDLINITIAL.16QAM_RATIO', 'TB.TOTNBRDLINITIAL.64QAM_RATIO'
        # From per-UE values we need to extract per-Cell Values
        # obs_kpms = []
        # for ue_kpm in ue_kpms:
        #     imsi, kpms = ue_kpm
        #     obs_kpms.append(kpms)

        # _RATIO values are the per Cell value / Tot nbr dl initial

        self.observations = ue_kpms
        return self.observations
    
    def _compute_reward(self) -> float:
        reward_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_reward)
        # TODO compute and return the float value for the reward
        # Coordinate with Simone Palumbo
        reward = 2
        # for ue_kpm in reward_kpms:
        #     imsi, thp = ue_kpm
        #     reward += thp

        self.reward = reward
        return self.reward
