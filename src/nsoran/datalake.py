from collections import defaultdict
import os
import sqlite3
import re

class SQLiteDatabaseAPI:
    lte_cu_cp_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "numActiveUes": "INTEGER",
        "cellId": "INTEGER",
        "DRB.EstabSucc.5QI.UEID (numDrb)": "INTEGER",
        "sameCellSinr": "REAL",
        "sameCellSinr 3gpp encoded": "REAL"
    }
    gnb_cu_cp_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "numActiveUes": "INTEGER",
        "DRB.EstabSucc.5QI.UEID (numDrb)": "INTEGER",
        "L3 serving Id(m_cellId)": "INTEGER",
        "UE (imsi)": "INTEGER",
        "L3 serving SINR": "REAL",
        "L3 serving SINR 3gpp": "REAL",
        "L3 neigh Id 1 (cellId)": "INTEGER",
        "L3 neigh SINR 1": "REAL",
        "L3 neigh SINR 3gpp 1 (convertedSinr)": "REAL",
        "L3 neigh Id 2 (cellId)": "INTEGER",
        "L3 neigh SINR 2": "REAL",
        "L3 neigh SINR 3gpp 2 (convertedSinr)": "REAL",
        "L3 neigh Id 3 (cellId)": "INTEGER",
        "L3 neigh SINR 3": "REAL",
        "L3 neigh SINR 3gpp 3 (convertedSinr)": "REAL",
        "L3 neigh Id 4 (cellId)": "INTEGER",
        "L3 neigh SINR 4": "REAL",
        "L3 neigh SINR 3gpp 4 (convertedSinr)": "REAL",
        "L3 neigh Id 5 (cellId)": "INTEGER",
        "L3 neigh SINR 5": "REAL",
        "L3 neigh SINR 3gpp 5 (convertedSinr)": "REAL",
        "L3 neigh Id 6 (cellId)": "INTEGER",
        "L3 neigh SINR 6": "REAL",
        "L3 neigh SINR 3gpp 6 (convertedSinr)": "REAL"
    }
    lte_cu_up_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "DRB.PdcpSduDelayDl(cellAverageLatency)": "REAL",
        "m_pDCPBytesDL(cellDlTxVolume)": "REAL",
        "DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)": "REAL",
        "Tot.PdcpSduNbrDl.UEID (txDlPackets)": "REAL",
        "DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)": "REAL",
        "DRB.PdcpSduDelayDl.UEID (pdcpLatency)": "REAL",
    }
    gnb_cu_up_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "cellId": "INTEGER",
        "QosFlow.PdcpPduVolumeDL_Filter.UEID(txPdcpPduBytesNrRlc)": "REAL",
        "DRB.PdcpPduNbrDl.Qos.UEID (txPdcpPduNrRlc)": "REAL"
    }
    du_keys = {
        "timestamp": "INTEGER",
        "ueImsiComplete": "INTEGER",
        "nrCellId": "INTEGER",
        "dlAvailablePrbs": "REAL",
        "ulAvailablePrbs": "REAL",
        "qci": "INTEGER",
        "dlPrbUsage": "REAL",
        "ulPrbUsage": "REAL",
        "TB.TotNbrDl.1": "REAL",
        "TB.TotNbrDlInitial": "REAL",
        "TB.TotNbrDlInitial.Qpsk": "REAL",
        "TB.TotNbrDlInitial.16Qam": "REAL",
        "TB.TotNbrDlInitial.64Qam": "REAL",
        "RRU.PrbUsedDl": "REAL",
        "TB.ErrTotalNbrDl.1": "REAL",
        "QosFlow.PdcpPduVolumeDL_Filter": "REAL",
        "CARR.PDSCHMCSDist.Bin1": "REAL",
        "CARR.PDSCHMCSDist.Bin2": "REAL",
        "CARR.PDSCHMCSDist.Bin3": "REAL",
        "CARR.PDSCHMCSDist.Bin4": "REAL",
        "CARR.PDSCHMCSDist.Bin5": "REAL",
        "CARR.PDSCHMCSDist.Bin6": "REAL",
        "L1M.RS-SINR.Bin34": "REAL",
        "L1M.RS-SINR.Bin46": "REAL",
        "L1M.RS-SINR.Bin58": "REAL",
        "L1M.RS-SINR.Bin70": "REAL",
        "L1M.RS-SINR.Bin82": "REAL",
        "L1M.RS-SINR.Bin94": "REAL",
        "L1M.RS-SINR.Bin127": "REAL",
        "DRB.BufferSize.Qos": "REAL",
        "DRB.MeanActiveUeDl": "REAL",
        "TB.TotNbrDl.1.UEID": "REAL",
        "TB.TotNbrDlInitial.UEID": "REAL",
        "TB.TotNbrDlInitial.Qpsk.UEID": "REAL",
        "TB.TotNbrDlInitial.16Qam.UEID": "REAL",
        "TB.TotNbrDlInitial.64Qam.UEID": "REAL",
        "TB.ErrTotalNbrDl.1.UEID": "REAL",
        "QosFlow.PdcpPduVolumeDL_Filter.UEID": "REAL",
        "RRU.PrbUsedDl.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin1.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin2.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin3.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin4.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin5.UEID": "REAL",
        "CARR.PDSCHMCSDist.Bin6.UEID": "REAL",
        "L1M.RS-SINR.Bin34.UEID": "REAL",
        "L1M.RS-SINR.Bin46.UEID": "REAL",
        "L1M.RS-SINR.Bin58.UEID": "REAL",
        "L1M.RS-SINR.Bin70.UEID": "REAL",
        "L1M.RS-SINR.Bin82.UEID": "REAL",
        "L1M.RS-SINR.Bin94.UEID": "REAL",
        "L1M.RS-SINR.Bin127.UEID": "REAL",
        "DRB.BufferSize.Qos.UEID": "REAL",
        "DRB.UEThpDl.UEID": "REAL",
        "DRB.UEThpDlPdcpBased.UEID": "REAL"
    }

    debug: bool = False

    def __init__(self, simulation_dir, num_ues_gnb, debug=False):
        """Create an SQLite Database inside the simulation folder and use it as data source

        Args:
            simulation_dir (str): path of the folder of the simulation
            num_ues_gnb (int): number of UEs for each gNB in the simulation
            debug (bool): if True, do not erase the db at the end of the simulation
        """        
        self.simulation_dir = simulation_dir
        self.num_ues = num_ues_gnb * 7 # number of gNBs in the scenario
        self.database_path = os.path.join(simulation_dir, 'database.db')
        self.tables = {} # we keep a reference of all the active tables
        # key is the table name, value is the dictionary {kpm name: type}

        self.acquire_connection()
        # print("Connected to the database.")
        self._create_table("lte_cu_cp", self.lte_cu_cp_keys)
        self._create_table("gnb_cu_cp", self.gnb_cu_cp_keys)
        self._create_table("lte_cu_up", self.lte_cu_up_keys)
        self._create_table("gnb_cu_up", self.gnb_cu_up_keys)
        self._create_table("du", self.du_keys)
        self.release_connection()

        self.debug = debug

    @staticmethod
    def sanitize_column_name(column_name):
            # Convert to lowercase
            column_name = column_name.lower()
            
            # Replace spaces with underscores
            column_name = column_name.replace(' ', '_')
            
            # Remove special characters using regular expression
            column_name = re.sub(r'[^\w\s]', '', column_name)
            
            return column_name

    def acquire_connection(self):
        self.connection = sqlite3.connect(self.database_path)
        if self.debug:
            self.connection.set_trace_callback(print)
        self.cursor = self.connection.cursor()
        return True
    
    def release_connection(self):
        if self.connection is None:
            print("Error: Not connected to the database, no need to release.")
            return True
        self.connection.commit()
        self.connection.close()
        self.connection = None
        return True

    def lock_connection(func):
        def wrapper(self, *args, **kwargs):
            need_connection =  self.connection is None
            if need_connection:
                self.acquire_connection()
            if self.debug:
                self.connection.set_trace_callback(print)
            result = func(self, *args, **kwargs)
            if need_connection:
                self.release_connection()
            return result
        return wrapper
    
    @lock_connection
    def _create_table(self, table_name: str, columns: dict[str,str]):
        """
        table_name (str): name of the table as it is gonna appear in the dataset
        columns (dict[str,str]): dictionary having the keys as the names of the kpms and the types
        """
        if self.connection is None:
            print(f"Error in creating table {table_name}: Not connected to the database.")
            return

        column_definitions = ', '.join([f"{SQLiteDatabaseAPI.sanitize_column_name(name)} {type}" for name, type in columns.items()])
        # Add UNIQUE constraint for timestamp and ueImsiComplete columns
        column_definitions += f", UNIQUE (timestamp, {SQLiteDatabaseAPI.sanitize_column_name('ueImsiComplete')})"

        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_definitions})"
        self.cursor.execute(query)
        self.tables[table_name] = columns

        if self.debug:
            print(f"Table '{table_name}' created.")

    @lock_connection
    def entry_exists(self, table_name, timestamp, ue_imsi_complete) -> bool:
        query = f"SELECT COUNT(*) FROM {table_name} WHERE {SQLiteDatabaseAPI.sanitize_column_name('timestamp')} = ? AND {SQLiteDatabaseAPI.sanitize_column_name('ueImsiComplete')} = ?"
        values = (timestamp, ue_imsi_complete)

        result = self.cursor.execute(query, values).fetchone()
        return result[0] > 0  # If the count is greater than 0, the row exists

    def insert_lte_cu_cp(self, data: dict):
        self.insert_data('lte_cu_cp', data)
    
    def insert_gnb_cu_cp(self, data: dict):
        self.insert_data('gnb_cu_cp', data)

    def insert_lte_cu_up(self, data: dict):
        self.insert_data('lte_cu_up', data)

    def insert_gnb_cu_up(self, data: dict):
        self.insert_data('gnb_cu_up', data)

    def insert_du(self, data: dict):
        self.insert_data('du', data)

    @lock_connection
    def insert_data(self, table_name, data: dict):
        if table_name in self.tables:
            admitted_keys = self.tables[table_name]
        else:
            raise ValueError(f'Input table name not found in the tables: {table_name} not in {self.tables.keys()}')
        
        # Filter kpms dictionary to only include acceptable columns
        filtered_kpms = {key: value for key, value in data.items() if key in admitted_keys}
        
        if not filtered_kpms:
            raise ValueError("No acceptable columns found in the input dictionary.")
        
        if self.entry_exists(table_name,timestamp=filtered_kpms['timestamp'], ue_imsi_complete=filtered_kpms['ueImsiComplete']):
            # we already inserted this in the DB, no need to do it again
            return

        placeholders = ', '.join(['?' for _ in filtered_kpms.values()])
        columns = ', '.join([SQLiteDatabaseAPI.sanitize_column_name(col_name) for col_name in filtered_kpms.keys()])
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        values = tuple(filtered_kpms.values())
        self.cursor.execute(query, values)
        if self.debug:
            print("Data inserted into the table.")

    @lock_connection
    def read_table(self, table_name):
        query = f"SELECT * FROM {table_name}"
        result = self.cursor.execute(query)
        return result.fetchall()

    @lock_connection
    def read_kpms(self, timestamp : int, required_kpms: list) -> list[tuple]:
        """Query the datalake to retrieve the observation vector. 
            The return value is the list of tuples of the size of the number of UEs in the scenario. 
            Each tuple is built by having as the first element the ueImsiComplete following the required_kpms.
            Order is ensured, i.e., the KPMs will be returned as the listed in the KPM.
            KPM with the same name are both returned expliciting the source table
           Args:
              timestamp (int): timestamp of the observation vector to retrieve
              required_kpms (list): list of KPMs to be retrieved
        """
        tables_involved: dict[list] = {} # key: table_name, value: list of the names of the kpms 

        found_kpms = [False] * len(required_kpms)

        for table_name, keys in self.tables.items():
            for index, required_kpm in enumerate(required_kpms):
                if required_kpm in keys:
                    if table_name not in tables_involved:
                        tables_involved[table_name] = []
                    tables_involved[table_name].append(required_kpm)
                    found_kpms[index] = True

        not_found_kpms = [kpm for found, kpm in zip(found_kpms, required_kpms) if not found]
        if not_found_kpms:
            raise ValueError(f"Column(s) {not_found_kpms} not found in any table.")

        # We have knowledge of what we want, let's create the query

        # Reverse mapping of kpm to the tables they appear in
        kpm_to_tables = defaultdict(list)
        for table, kpms in tables_involved.items():
            for kpm in kpms:
                kpm_to_tables[kpm].append(table)
        
        # Construct the SQL query
        from_clause = next(iter(tables_involved))  # Get the first table for the FROM clause
        select_clause = [f"{from_clause}.ueImsiComplete"] # Add ueImsiComplete to the select clause once
        join_clause = []
        joined_tables = set([from_clause])

        # Ensure the SELECT clause includes all requested KPMs in the order of required_kpms
        for required_kpm in required_kpms:
            if len(kpm_to_tables[required_kpm]) > 1:  # If the KPM appears in more than one table
                for table in kpm_to_tables[required_kpm]:
                    select_clause.append(f"{table}.{self.sanitize_column_name(required_kpm)} AS required_kpm_{table}")
            else:  # If the KPM appears in only one table
                table = kpm_to_tables[required_kpm][0]
                select_clause.append(f"{table}.{self.sanitize_column_name(required_kpm)}")

        # Create joins on timestamp and ueImsiComplete
        tables = list(tables_involved.keys())
        base_table = tables[0]
        for table in tables[1:]:
            join_clause.append(f"INNER JOIN {table} ON {base_table}.timestamp = {table}.timestamp AND {base_table}.ueImsiComplete = {table}.ueImsiComplete")
            joined_tables.add(table)

        # Combine clauses into the final SQL query
        query = f"SELECT {', '.join(select_clause)} FROM {from_clause}"
        if join_clause:
            query += " " + " ".join(join_clause)

        # Add the WHERE clause using the from_clause table's timestamp
        query += f" WHERE {from_clause}.timestamp = ?"

        result = self.cursor.execute(query, (timestamp,)).fetchall()
        return result if result else None # [(observation_tuple)]

    @staticmethod
    def extract_cellId(filepath) -> int:
        # Define a regular expression pattern to match the number at the end of the path
        pattern = r'(\d+).txt$'

        # Use re.search to find the match in the path
        match = re.search(pattern, filepath)

        # Check if a match is found
        if match:
            # Extract and return the matched number
            return int(match.group(1))
        
        raise ValueError("Unable to extract cellId")
    
    def __del__(self):
        if self.connection is not None:
            self.release_connection()
            if self.debug:
                print("Connection to the database closed.")

if __name__ == "__main__":
    simulation_dir = "./"
    db_api = SQLiteDatabaseAPI(simulation_dir, 7, False)

    db_api.insert_data("lte_cu_cp", {
                             "timestamp": 1000,
                             "ueImsiComplete": "12T",
                             "numActiveUes": 12,
                             "DRB.EstabSucc.5QI.UEID (numDrb)": 1,
                             "sameCellSinr": 90.3,
                             "sameCellSinr 3gpp encoded": 93.2,
                             "id": 1
                             })

    data = db_api.read_table("lte_cu_cp")
    print("Read data:", data)

    db_api.insert_lte_cu_up({
                            "timestamp": 1000,
                            "ueImsiComplete": "1",
                            "cellId": 1,
                            "DRB.PdcpSduDelayDl(cellAverageLatency)": 10,
                            "m_pDCPBytesDL(cellDlTxVolume)": 10,
                            "DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)": 100,
                            "Tot.PdcpSduNbrDl.UEID (txDlPackets)": 100,
                            "DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)": 100,
                            "DRB.PdcpSduDelayDl.UEID (pdcpLatency)": 100,
                            "QosFlow.PdcpPduVolumeDL_Filter.UEID(txPdcpPduBytesNrRlc)": 100,
                            "DRB.PdcpPduNbrDl.Qos.UEID (txPdcpPduNrRlc)": 100
                            })

    db_api.insert_lte_cu_up({   "timestamp": 1000,
                            "ueImsiComplete": "2",
                            "cellId": 1,
                            "DRB.PdcpSduDelayDl(cellAverageLatency)": 10,
                            "m_pDCPBytesDL(cellDlTxVolume)": 10,
                            "DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)": 100,
                            "Tot.PdcpSduNbrDl.UEID (txDlPackets)": 100,
                            "DRB.PdcpSduBitRateDl.UEID (pdcpThroughput)": 200,
                            "DRB.PdcpSduDelayDl.UEID (pdcpLatency)": 100,
                            "QosFlow.PdcpPduVolumeDL_Filter.UEID(txPdcpPduBytesNrRlc)": 100,
                            "DRB.PdcpPduNbrDl.Qos.UEID (txPdcpPduNrRlc)": 100
                            })

    res = db_api.read_kpms(1000, ["DRB.PdcpSduVolumeDl_Filter.UEID (txBytes)", "Tot.PdcpSduNbrDl.UEID (txDlPackets)"])
    print("lte_cu_up contents: ", db_api.read_table("lte_cu_up"))
    print("Read data:", res)
    #data = db_api.read_avg_thp_cell(1000)
    #print("Read data:", data)
