from nsoran.ns_env import NsOranEnv 
from gymnasium import spaces
import logging

import numpy as np

class TrafficSteeringEnv(NsOranEnv):
    def __init__(self, ns3_path:str, scenario_configuration:dict, output_folder:str, optimized:bool, verbose=False, time_factor=0.001, Cf=1.0, lambdaf=0.1):
        """Environment specific parameters:
            verbose (bool): enables logging
            time_factor (float): applies convertion from seconds to another multiple (eg. ms). See compute_reward
            Cf (float): Cost factor for handovers. See compute_reward
            lambdaf (float): Decay factor for handover cost. See compute_reward
        """
        super().__init__(ns3_path=ns3_path, scenario='scenario-one', scenario_configuration=scenario_configuration,
                         output_folder=output_folder, optimized=optimized,
                         control_header = ['timestamp','ueId','nrCellId'], log_file='TsActions.txt', control_file='ts_actions_for_ns3.csv')
        # These features can be hardcoded since they are specific for the use case
        self.columns_state = ['RRU.PrbUsedDl', 'L3 serving SINR', 'DRB.MeanActiveUeDl', 
                              'TB.TotNbrDlInitial.Qpsk', 'TB.TotNbrDlInitial.16Qam', 
                              'TB.TotNbrDlInitial.64Qam', 'TB.TotNbrDlInitial']

        # We need the throughput as well as the cell id to determine whether an handover occurred
        self.columns_reward = ['DRB.UEThpDl.UEID', 'nrCellId']
        # In the traffic steering use case, the action is a combination between 
        n_gnbs = 7  # scenario one has always 7 gnbs 
        n_actions_ue = 7 # each UE can connect to a gNB identified by ID (from 2 to 8), 0 is No Action
        # obs_space size: (# ues_per_gnb * # n_gnbs, # observation_columns + timestamp = 1)
        self.observation_space = spaces.Box(shape=(self.scenario_configuration['ues']*n_gnbs,len(self.columns_state)+1), low=-np.inf, high=np.inf, dtype=np.float64)
        self.action_space = spaces.MultiDiscrete([n_actions_ue] * self.scenario_configuration['ues'] *  n_gnbs)
        # Stores the kpms of the previous timestamp (see compute_reward)
        self.previous_df = None
        self.previous_kpms = None
        # Auxiliary functions to keep track of last handover time (see compute_reward)
        self.handovers_dict = dict()
        self.verbose = verbose
        if self.verbose:
            logging.basicConfig(filename='./reward_ts.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')
        self.time_factor = time_factor
        self.Cf = Cf
        self.lambdaf = lambdaf

    def _compute_action(self, action) -> list[tuple]:    
        # action from multidiscrete shall become a list of ueId, targetCell.
        # If a targetCell is 0, it means No Handover, thus we don't send it
        action_list = []
        for ueId, targetCellId in enumerate(action):
            if targetCellId != 0: # and 
                # Once we are in this condition, we need to transform the action from the one of gym to the one of ns-O-RAN
                action_list.append((ueId + 1, targetCellId + 2))
        if self.verbose:
            logging.debug(f'Action list {action_list}')
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

        self.observations = np.array(ue_kpms)
        return self.observations
    
    def _compute_reward(self) -> float:
        # Computes the reward for the traffic steering environment. Based off journal on TS
        # The total reward is the sum of per ue rewards, calculated as the difference in the
        # logarithmic throughput between indication periodicities. If an UE experienced an HO,
        # its reward takes into account a cost function related to said handover. The cost
        # function punishes frequent handovers.
        # See the docs for more info.

        total_reward = 0.0
        current_kpms = self.datalake.read_kpms(self.last_timestamp, self.columns_reward)

        # If this is the first iteration we do not have the previous kpms
        if self.previous_kpms is None:
            if self.verbose:
                logging.debug(f'Starting first reward computation at timestamp {self.last_timestamp}')
            self.previous_timestamp = self.last_timestamp - (self.scenario_configuration['indicationPeriodicity'] * 1000)
            self.previous_kpms = self.datalake.read_kpms(self.previous_timestamp, self.columns_reward)

        #Assuming they are of the same lenght
        for t_o, t_n in zip(self.previous_kpms, current_kpms):
            ueImsi_o, ueThpDl_o, sourceCell = t_o
            ueImsi_n, ueThpDl_n, currentCell = t_n
            if ueImsi_n == ueImsi_o:
                HoCost = 0
                if currentCell != sourceCell:
                    lastHo = self.handovers_dict.get(ueImsi_n, 0)  # Retrieve last handover time or default to 0
                    if lastHo != 0: # If this is the first HO the cost is 0
                        timeDiff = (self.last_timestamp - lastHo) * self.time_factor
                        HoCost = self.Cf * ((1 - self.lambdaf) ** timeDiff)
                    self.handovers_dict[ueImsi_n] = self.last_timestamp  # Update dictionary
 
                LogOld = 0
                LogNew = 0
                if ueThpDl_o != 0:
                    LogOld = np.log10(ueThpDl_o)
                if ueThpDl_n != 0:
                    LogNew = np.log10(ueThpDl_n)

                LogDiff = LogNew - LogOld
                reward_ue = LogDiff - HoCost
                if self.verbose:
                    logging.debug(f"Reward for UE {ueImsi_n}: {reward_ue} (LogDiff: {LogDiff}, HoCost: {HoCost})")
                total_reward += reward_ue
            else:
                if self.verbose:
                    logging.error(f"Unexpected UeImsi mismatch: {ueImsi_o} != {ueImsi_n} (current ts: {self.last_timestamp})")
        if(self.verbose):
            logging.debug(f"Total reward: {total_reward}")
        self.previous_kpms = current_kpms
        self.previous_timestamp = self.last_timestamp
        self.reward = total_reward
        return self.reward
