from typing_extensions import override
import numpy as np
import pandas as pd
from nsoran.ns_env import NsOranEnv 
import pandas as pd
import glob
import csv
import os

# Reward function components
# 'SUM_QOSFLOW_PDCPPDUVOLUMEDL_FILTER' represents the sum of individual QoS flow volume for downlink PDU per cell.
# 'SUM_TB_TOTNBRDL_1' is the total number of downlink transport blocks across cells.
# 'SUM_RLF_VALUE' indicates the total number of radio link failures (RLFs) counted where L3servingSINR < -5.
#     RLF_VALUE_{i} is computed as:
#     numValues = tempDf[tempDf['L3servingSINR'] < -5]['timestamp'].count()
#     df['RLF_VALUE'][df['timestamp'] == singleTimeStamp] = numValues
# 'SUM_ES_ON_COST' calculates the total cost associated with energy-saving states.
#     ES_ON_COST is computed using the es_on_cost_calculation() method.
# 'ZERO_COUNT' indicates how many zero states are present, using the zero_count() method.

# Reward function formula:
# 0.51 * SUM_QOSFLOW_PDCPPDUVOLUMEDL_FILTER_QUANTILE
# - 0.19 * (SUM_TB_TOTNBRDL_1_QUANTILE + ZERO_COUNT_QUANTILE)
# - 0.2 * SUM_RLF_VALUE_QUANTILE
# - 0.1 * SUM_ES_ON_COST_QUANTILE

# Base station state attributes used in the model:
# EEKPI_RL_{i} represents the ratio of QoS flow volume to transport blocks for downlink.
# ES_ON_COST_{i} is included as part of the reward function (see es_on_cost_calculation()).
# QOSFLOW_PDCPPDUVOLUMEDL_FILTER_{i} measures the QoS PDU volume for downlink flows.
# RLF_COUNTER_{i} counts radio link failures where L3servingSINR < -5:
#     numValues = tempDf[tempDf['L3servingSINR'] < -5]['timestamp'].count()
#     rlfValue = (numValues / totalCount) * 100
# RRU_PRBTOTDL_{i} represents the percentage of physical resource blocks used for downlink:
#     df.apply(lambda x: (x['RRU_PRBUSEDDL'] / 139) * 100, axis=1)
# TB_TOTNBRDLINITIAL_64QAM_RATIO_{i} computes the ratio of 64QAM transport blocks:
#     TB_TOTNBRDLINITIAL_SUM = sum of QPSK, 16QAM, and 64QAM initial transport blocks.
#     Ratio is calculated as TB_TOTNBRDLINITIAL_64QAM / TB_TOTNBRDLINITIAL_SUM, handling division by zero.

# Attributes extracted from ns-3 logs per cell:
# - QOSFLOW_PDCPPDUVOLUMEDL_FILTER: QoS flow downlink volume.
# - TB_TOTNBRDL_1: Total number of downlink transport blocks.
# - L3servingSINR: Signal-to-Interference-plus-Noise Ratio (SINR) at Layer 3.
# - RRU_PRBUSEDDL: Physical resource block usage for downlink.
# - TB_TOTNBRDLINITIAL_64QAM, TB_TOTNBRDLINITIAL_QPSK, TB_TOTNBRDLINITIAL_16QAM: Transport block metrics by modulation scheme.
# - ES_STATE: Energy-saving state (1 = OFF, 0 = ON).


class EnergySavingEnv(NsOranEnv):
    
    gnb_state_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "cellId": "INTEGER",
            "state": "INTEGER"
        } 
        
    def __init__(self, ns3_path:str, scenario_configuration:dict, output_folder:str, optimized:bool, do_heuristic:bool = True):
        super().__init__(ns3_path=ns3_path, scenario='scenario-three', scenario_configuration=scenario_configuration,
                output_folder=output_folder, optimized=optimized,
                control_header = ['timestamp','cellId','hoAllowed'], log_file='EsActions.txt', control_file='es_actions_for_ns3.csv')
        
        self.folder_name = "Simulation"
        self.ns3_simulation_time = scenario_configuration['simTime']*1000
        self.columns_state = ['EEKPI_RL_2', 'EEKPI_RL_3', 'EEKPI_RL_4', 'EEKPI_RL_5', 'EEKPI_RL_6', 'EEKPI_RL_7', 'EEKPI_RL_8',
            'ES_ON_COST_2', 'ES_ON_COST_3', 'ES_ON_COST_4', 'ES_ON_COST_5', 'ES_ON_COST_6', 'ES_ON_COST_7', 'ES_ON_COST_8',
            'QosFlow.PdcpPduVolumeDL_Filter_2', 'QosFlow.PdcpPduVolumeDL_Filter_3', 'QosFlow.PdcpPduVolumeDL_Filter_4', 'QosFlow.PdcpPduVolumeDL_Filter_5', 'QosFlow.PdcpPduVolumeDL_Filter_6', 'QosFlow.PdcpPduVolumeDL_Filter_7', 'QosFlow.PdcpPduVolumeDL_Filter_8',
            'RLF_Counter_2', 'RLF_Counter_3', 'RLF_Counter_4', 'RLF_Counter_5', 'RLF_Counter_6', 'RLF_Counter_7', 'RLF_Counter_8',
            'RLF_VALUE_2', 'RLF_VALUE_3', 'RLF_VALUE_4', 'RLF_VALUE_5', 'RLF_VALUE_6', 'RLF_VALUE_7', 'RLF_VALUE_8',
            'RRU_PRBTOTDL_2', 'RRU_PRBTOTDL_3', 'RRU_PRBTOTDL_4', 'RRU_PRBTOTDL_5', 'RRU_PRBTOTDL_6', 'RRU_PRBTOTDL_7', 'RRU_PRBTOTDL_8',
            'RRU.PrbUsedDl_2', 'RRU.PrbUsedDl_3', 'RRU.PrbUsedDl_4', 'RRU.PrbUsedDl_5', 'RRU.PrbUsedDl_6', 'RRU.PrbUsedDl_7', 'RRU.PrbUsedDl_8',
            'TB_TOTNBRDLINITIAL_64QAM_RATIO_2', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_3', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_4', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_5', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_6', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_7', 'TB_TOTNBRDLINITIAL_64QAM_RATIO_8',
            'SUM_QosFlow.PdcpPduVolumeDL_Filter',
            'SUM_RLF_VALUE',
            'SUM_TB.TotNbrDl.1',
            'SUM_ES_ON_COST',
            'ZERO_COUNT']
        # ["cellId", "QOSFLOW_PDCPPDUVOLUMEDL_FILTER", "TB_TOTNBRDL_1", "L3servingSINR", "RRU_PRBUSEDDL", "TB_TOTNBRDLINITIAL_64QAM", "TB_TOTNBRDLINITIAL_QPSK", "TB_TOTNBRDLINITIAL_16QAM", "ES_STATE"]
        self.columns_reward = ['SUM_QosFlow.PdcpPduVolumeDL_Filter',
            'SUM_TB.TotNbrDl.1',
            'SUM_RLF_VALUE',
            'SUM_ES_ON_COST',
            'ZERO_COUNT'
            ]
        # Action space
        self.action_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 28, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 48, 49, 50, 52, 56, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 76, 80, 81, 82, 84, 88, 96, 97, 98, 100, 104, 112]
        self.cellList = [2, 3, 4, 5, 6, 7, 8]
        self.observations = []
        self.cells_states = {}
        # Used for ES_ON_COST and used to save the timestamp when a cell change state
        self.cell_timestamp_state_dict = {cell: float('inf') for cell in self.cellList} 
        self.Cf = 1
        self.lambdaf = 0.1
        self.time_factor = 0.01
        self.heur = do_heuristic
        self.num_steps = 0
        self.previous_inverted_action = "0000000"

    @override
    def _compute_action(self, action):
        """ Function that converts the agents action defined in Gym into the ns-O-RAN required format according to the use case
        """
        cell_act_comb_lst = []
        if self.heur == True:
            cell_id=[2,3,4,5,6,7,8]
            cell_act_comb_lst=[[cell,act] for cell,act in zip(cell_id,action)]
        else:
            # Action is an index for our array of actions
            dec_action = self.action_list[action]
            bin_actions = [int(i) for i in list(f'{dec_action:07b}')]
            cell_id=[2,3,4,5,6,7,8]
            cell_act_comb_lst=[[cell,bin_action] for cell,bin_action in zip(cell_id,bin_actions)]
            # Correcting the transformation logic using list comprehension
            cell_act_comb_lst = [
                [cell, 1 if bin_action == 0 else 0 if bin_action == 1 else bin_action] 
                for cell, bin_action in cell_act_comb_lst
            ]
        return cell_act_comb_lst
    
    def _update_cell_states(self):
        """Function that updates the states of the cells saved in a class variable
        """
        cell_states_table = self.datalake.read_table('bsState')
        # Primary key of the results is "Timestamp", "ueIMSI" so the cellID and the state are duplicates
        # each element of the list is a row of the file with the following columns
        # [("Timestamp", "ueIMSI", "Id", "State"),(...)]
        states_of_interest = []
        # Filter rows only from last timestamp
        for cell_state in cell_states_table:
            if cell_state[0] == (self.last_timestamp-100):
                states_of_interest.append(cell_state)
        # For model purposes
        if len(states_of_interest)!=7:
            for cellId in self.cellList:
                self.cells_states[cellId] = 1
        else:
            # Populate/update cell state dictionary
            for state in states_of_interest:
                # Extract the cellId
                cellId = state[2]
                # Save in a dictionary, cellId as key and the State column as state
                # Even if cellId is redundant, in the dictionary there will be just one occurrence 
                self.cells_states[cellId] = state[3]

    @override
    def _get_obs(self):
        # ["cellId", "QOSFLOW_PDCPPDUVOLUMEDL_FILTER", "TB_TOTNBRDL_1", "L3servingSINR", "RRU_PRBUSEDDL", "TB_TOTNBRDLINITIAL_64QAM", "TB_TOTNBRDLINITIAL_QPSK", "TB_TOTNBRDLINITIAL_16QAM", "ES_STATE"] #Database (1=ON, 0=OFF), Mavnenir(1=OFF, 0=ON)
        kpms_raw = ["nrCellId", "QosFlow.PdcpPduVolumeDL_Filter", "TB.TotNbrDl.1", "L3 serving SINR", "RRU.PrbUsedDl", "TB.TotNbrDlInitial.64Qam", "TB.TotNbrDlInitial.Qpsk", "TB.TotNbrDlInitial.16Qam"]       
        ue_kpms = self.datalake.read_kpms(self.last_timestamp, kpms_raw) 
        self._update_cell_states()  
        # Now cells_states is updated with state of latest cells           
        # iterate over ue_kpms to add state value
        ue_complete_kpms = []
        # For each row in ue_kpms look for its state into cells_states and save it
        for ue_kpm in ue_kpms:
            # Create a new tuple with the same elements of ue_kpm + state in the latest position
            # State is calculated using self.cell_state[ue_kpm[1]] because index 0 if for ueIMSI
            state = self.cells_states.get(ue_kpm[1], ())  # Get state using cell_id (second element in ue_kpm)
            # Create a new tuple by concatenating ue_kpm with the state tuple
            new_ue_kpm = ue_kpm + (state,)
            ue_complete_kpms.append(new_ue_kpm)      
        # At this point observations_raw is UEIMSI + kpms_raw + state
        # Define the column names based on kpms_raw and the single state column
        columns = ['ueImsiComplete'] + kpms_raw + ['state']  # Add 'state' as the last column
        # Create the DataFrame
        df = pd.DataFrame(ue_complete_kpms, columns=columns)
        df["timestamp"] = self.last_timestamp
        # Count the RLF at UEs level
        df, columns= self.getRLFCounter(df, columns)
        # Now we need to convert the dataframe from UEs centric to Cell centric
        df = self.ue_centric_tocell_centric(df)
        self.observations = self.offline_training_preprocessing(df)
        states = self.observations[self.columns_state]
        states_tuple = [tuple(states.iloc[0].values)]
        return states_tuple
        
    @override
    def _compute_reward(self):
        # Since reward kpms are the same as state kpms
        cell_df = self.observations[self.columns_reward].copy()
        # Grafana db 
        db_row = {}
        db_row['timestamp'] = self.last_timestamp
        db_row['ueImsiComplete'] = None
        db_row['time_grafana'] = self.last_timestamp
        db_row['step'] = self.num_steps
        db_row['throughput'] =float(cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter'][0])*10/10**6
        db_row['en_cons'] = float(cell_df['SUM_TB.TotNbrDl.1'][0])
        db_row['rlf'] = float(cell_df['SUM_RLF_VALUE'][0])
        db_row['on_cost'] = float(cell_df['SUM_ES_ON_COST'][0])
        # Step 2: Calculate reward values
        cell_df['reward'] = (
            0.51 * cell_df['SUM_QosFlow.PdcpPduVolumeDL_Filter']
            - 0.19 * (cell_df['SUM_TB.TotNbrDl.1'] + cell_df['ZERO_COUNT'])
            - 0.2 * cell_df['SUM_RLF_VALUE']
            - 0.1 * cell_df['SUM_ES_ON_COST']
        )
        reward = cell_df['reward'][0]
        # Grafana db 
        db_row['reward'] = reward
        # Insert the data into the datalake
        self.datalake.insert_data("grafana", db_row)
        return reward

    @override
    def _init_datalake_usecase(self):
        # Grafana table
        grafana_keys = {
            "timestamp": "INTEGER",
            "ueImsiComplete": "INTEGER",
            "time_grafana": "INTEGER",
            "step": "INTEGER",
            "throughput": "REAL",
            "en_cons": "REAL",
            "rlf": "REAL",
            "on_cost": "REAL",
            "reward": "REAL"
        } 
        self.datalake._create_table("bsState",self.gnb_state_keys)  
        self.datalake._create_table("grafana",grafana_keys)  
        return super()._init_datalake_usecase()

    @override
    def _fill_datalake_usecase(self):
        for file_path in glob.glob(os.path.join(self.sim_path, 'bsState.txt')):
            with open(file_path, 'r') as csvfile:
                for row in csv.DictReader(csvfile, delimiter=' '):
                    timestamp = int(row['UNIX'])
                    if timestamp >= self.last_timestamp:
                        # Prepare the database row
                        # Insert ueImsiComplete=NULL because we don't need to read from the database, we just dave a local variable
                        db_row = {}
                        db_row['timestamp'] = timestamp
                        db_row['ueImsiComplete'] = None  # Set to null
                        db_row['cellId'] = int(row['Id'])
                        db_row['state'] = int(row['State'])
                        # Insert the data into the datalake
                        self.datalake.insert_data("bsState", db_row)
                        # Update the last timestamp
                        self.last_timestamp = timestamp

    def ue_centric_tocell_centric(self, df):
        """Function used to clean the dataframe with ns-3 row data
        """
        # Delete columns that are ue centric
        df.drop(columns=['ueImsiComplete', 'L3 serving SINR'], inplace=True)
        # Remove completely identical rows
        df = df.drop_duplicates()
        # Reset index
        df.reset_index(drop=True, inplace=True)
        return df
        
    def rename_columns(self, columns, cell_no):
        cols = []
        for i in columns:
            cols.append(i+'_'+str(cell_no))
        return cols

    def offline_training_preprocessing(self, df):
        """
        Preprocess the DataFrame by calculating KPIs and KPMs for reward for each cell.
        """
        df = self.add_eekpi_qpsk_16_64qam_sum_and_ratio(df)
        # Sort the final DataFrame by the TIMESTAMP column in ascending order
        df.sort_values(by=["timestamp"], ascending=True, inplace=True)
        # Invert State values
        df["state"] = df["state"].apply(lambda x: 1 if x == 0 else (0 if x == 1 else x))
        # Initialize an empty DataFrame to store the merged results
        cell_df = pd.DataFrame()
        is_initial_cell = True  # Flag to track the first cell's DataFrame
        # Iterate over the list of cells
        for cell in self.cellList:
            # Filter the data for the current cell and create a copy to avoid modifying the original DataFrame
            temp_cell_df = df.loc[df["nrCellId"] == cell].copy()
            # Calculate the percentage of PRB utilization
            temp_cell_df['RRU_PRBTOTDL'] = (temp_cell_df['RRU.PrbUsedDl'] / 139) * 100
            # Calculate the general EEKPI (Energy Efficiency KPI) for the downlink
            temp_cell_df['EEKPI_RL'] = (
                temp_cell_df['QosFlow.PdcpPduVolumeDL_Filter'] / temp_cell_df['TB.TotNbrDl.1']
            )
            # Rename the columns for the current cell to ensure uniqueness
            temp_cell_df.columns = self.rename_columns(temp_cell_df.columns, cell)
            # Rename the TIMESTAMP column to align across all cells for merging
            temp_cell_df.rename(columns={f"timestamp_{cell}": "timestamp"}, inplace=True)
            # Merge the data of the current cell with the overall DataFrame
            if is_initial_cell:
                cell_df = temp_cell_df
                is_initial_cell = False  # Mark the first cell as processed
            else:
                cell_df = pd.merge(cell_df, temp_cell_df, how="outer", on=["timestamp"])
            # Free up memory by deleting the temporary DataFrame
            del temp_cell_df
        # Replace NaN values in 'ES_STATE' columns with 1 and convert to integers
        es_state_columns = cell_df.columns[cell_df.columns.str.startswith("state_")]
        cell_df[es_state_columns] = cell_df[es_state_columns].fillna(1).astype(np.int64)
        # Replace all remaining NaN values in the DataFrame with 0
        cell_df = cell_df.fillna(0)
        # Step 1: Calculate ES on-costs for all cells
        cell_df = self.es_on_cost_calculation(cell_df)
        # Step 2: Save BS states
        # Step 3: Manage missing values for specific columns
        columns_to_clean = {
            'QosFlow.PdcpPduVolumeDL_Filter_': 'float32',
            'TB.TotNbrDl.1_': 'float32',
            'EEKPI_RL_': 'float32',
            'RLF_VALUE': 'float32'
        }
        for prefix, dtype in columns_to_clean.items():
            for col in cell_df.columns[cell_df.columns.str.startswith(prefix)]:
                cell_df.loc[cell_df[col] == '', col] = 0.0
                cell_df[col] = cell_df[col].astype(dtype)
        
        # Step 4: Round numeric columns to 2 decimal places
        cell_df = cell_df.round(2)
        # Step 5: Calculate reward components
        # Convert decimal to binary string (padded to 7 bits)
        cell_df['ACTION_BINARY'] = self.previous_inverted_action
        cell_df['ACTION_BINARY'] = cell_df['ACTION_BINARY'].astype(str)
        # Count the number of zeros in the binary representation
        cell_df['ZERO_COUNT'] = cell_df['ACTION_BINARY'].apply(
            lambda x: x.count('0')
        )
        # Sum columns for KPIs and costs
        kpi_sums = {
            'SUM_QosFlow.PdcpPduVolumeDL_Filter': 'QosFlow.PdcpPduVolumeDL_Filter_',
            'SUM_TB.TotNbrDl.1': 'TB.TotNbrDl.1_',
            'SUM_ES_ON_COST': 'ES_ON_COST_',
            'SUM_RLF_VALUE': 'RLF_VALUE_'
        }
        for sum_col, prefix in kpi_sums.items():
            cell_df[sum_col] = cell_df.filter(like=prefix).sum(axis=1)
        # Ensure numeric type for summed columns
        for sum_col in kpi_sums.keys():
            cell_df[sum_col] = pd.to_numeric(cell_df[sum_col])
        # Step 6: Calculate EEKPI_RL by cell
        for cell in self.cellList:
            tb_col = f'TB.TotNbrDl.1_{cell}'
            qos_col = f'QosFlow.PdcpPduVolumeDL_Filter_{cell}'
            eekpi_col = f'EEKPI_RL_{cell}'
            # Avoid division by zero
            cell_df[tb_col] = cell_df[tb_col].apply(
                lambda x: x if x != 0 else 0.00001
            )
            # Calculate EEKPI_RL
            cell_df[eekpi_col] = cell_df.apply(
                lambda x: x[qos_col] / x[tb_col], axis=1
            )
        return cell_df
    
    def add_eekpi_qpsk_16_64qam_sum_and_ratio(self, df):
        """
        Adds multiple EEKPI-related columns to the DataFrame by performing operations on existing columns.
        """
        # Calculate the total sum of TB_TOTNBRDLINITIAL_* columns
        df['TB.TOTNBRDLINITIAL.SUM'] = (
            df['TB.TotNbrDlInitial.Qpsk'] +
            df['TB.TotNbrDlInitial.16Qam'] +
            df['TB.TotNbrDlInitial.64Qam']
        )
        df['TB_TOTNBRDLINITIAL_64QAM_RATIO'] = (
            df['TB.TotNbrDlInitial.64Qam'] / df['TB.TOTNBRDLINITIAL.SUM']
        ).fillna(0.00001)
        # Handle RRU_PRBUSEDDL (avoid division by zero by using a small default value)
        df['RRU.PrbUsedDl'] = df['RRU.PrbUsedDl'].replace(0, 0.00001)
        # TB.TotNbrDl.1 handling to avoid division by zero
        df['TB.TotNbrDl.1'] = df['TB.TotNbrDl.1'].replace(0, 0.00001)
        return df

    def getRLFCounter(self, df, columns):
        """Function that adds the KPI related to the RLFs into the dataframe
        """
        # Ensure the timestamp column is of integer type
        df['timestamp'] = df['timestamp'].astype(int)
        # Initialize RLF_Counter and RLF_VALUE columns
        df['RLF_Counter'] = 0.0
        df['RLF_VALUE'] = 0
        columns += ['RLF_Counter', 'RLF_VALUE']
        # Replace -inf values in 'L3 serving SINR' with 0
        df['L3 serving SINR'] = df['L3 serving SINR'].replace(-np.inf, 0)
        # Group by timestamp and nrCellId for efficient processing
        grouped = df.groupby(['timestamp', 'nrCellId'])
        # Iterate through each group
        for (timestamp, cell), group in grouped:
            total_count = group.shape[0]
            if total_count > 0:  # Avoid division by zero
                num_values = group[group['L3 serving SINR'] < -5].shape[0]
                rlf_value = (num_values / total_count) * 100
                # Update RLF_Counter and RLF_VALUE for the current group
                mask = (df['timestamp'] == timestamp) & (df['nrCellId'] == cell)
                df.loc[mask, 'RLF_Counter'] = rlf_value
                df.loc[mask, 'RLF_VALUE'] = num_values

        return df, columns

    def es_on_cost_calculation(self, cell_df):
        """
        Calculate the energy-saving (ES) on-cost for each cell in the cell list.
        Cost increases if cell turning OFF too fast (cell stays ON (state=0) for too short)
        Intent is to AVOID turning OFF the same cell too frequently (1/ES MODE , ON->OFF->ON) 
        """
        for cell in self.cellList:
            current_timestamp = self.last_timestamp
            # Initialize a list for TIME_DIFF_OBS
            time_diff_obs = []
            current_state = self.cells_states.get(cell, ())
            # Convert states coming from cells_states
            if current_state==0:
                current_state=1
            # If it is 1
            else:
                current_state = 0
            if current_state == 1:
                # If state == 1, calculate time difference from saved timestamp
                if self.cell_timestamp_state_dict[cell] == float('inf'):
                    # First time state becomes 1, set initial timestamp
                    time_diff_obs.append(100)  # No time has elapsed yet
                    # Update the saved timestamp for this cell
                    self.cell_timestamp_state_dict[cell] = current_timestamp
                else:
                    # Calculate time difference
                    time_diff_obs.append(current_timestamp - self.cell_timestamp_state_dict[cell]+100)
            else:
                # If state == 0, reset the saved timestamp and set time difference to inf
                time_diff_obs.append(float('inf'))
                self.cell_timestamp_state_dict[cell] = float('inf')

            # Add TIME_DIFF_OBS column to the DataFrame
            time_diff_obs_col = f'TIME_DIFF_OBS_{cell}'
            cell_df[time_diff_obs_col] = time_diff_obs
            
            # Calculate the ES on-cost using the formula
            es_on_cost_col = f'ES_ON_COST_{cell}'
            cell_df[es_on_cost_col] = cell_df[time_diff_obs_col].apply(
                lambda diff: self.Cf * ((1 - self.lambdaf) ** (diff * self.time_factor)) if diff != float('inf') else 0
            )
        return cell_df
    
    def bs_states_list(self):
        """The function retrieves the current state of BSs from a datalake table for the latest timestamp 
        and returns a list of corresponding KPMs in an inverted binary format.  
        """
        # Get actual bs state with present timestamp
        cell_states_table = self.datalake.read_table('bsState')
        states_of_interest = []
        # Filter rows only from last timestamp
        for cell_state in cell_states_table:
            if cell_state[0] == self.last_timestamp:
                states_of_interest.append(cell_state)
        current_kpms = []
        for state in states_of_interest:
            current_kpms.append(state[3])
        # "cell_state  cellId: State {2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}" 
        # Invert 0 and 1 because of logic (0 = ES OFF/Cell ON, 1 = ES ON/Cell OFF)
        inverted_action_ar = [1 if element == 0 else 0 for element in current_kpms]
        return inverted_action_ar
        