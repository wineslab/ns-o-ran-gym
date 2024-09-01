from os import path

class ActionController():
    """
    The ActionController class is responsible for delivering the action to ns-O-RAN. 
    In the stand-alone mode, the action is delivered by writing on the appropriate file.
    """
    directory : str
    log_filename : str
    control_filename : str

    def __init__(self, sim_path, log_filename, control_filename, header):
        """Initialize Controller and its files
        Args:
            sim_path (str): the simulation path
            log_filename (str): the name of the file where the; This file is purely for logging purposed and it is not read by ns-3
            control_filename (str): the name of the control file that delivers the action to ns-3; This file is read by ns-3
            header (dict): dictionary whose keys represent the fields of the action that the agent is going to write 
        """
        self.directory = sim_path
        self.log_filename = log_filename
        self.control_filename = control_filename
        with open(path.join(self.directory, self.log_filename), 'w') as file:
                file.write(f"{','.join(header)}\n")
                file.flush()        

        open(path.join(self.directory, self.control_filename), 'a').close()

    def create_control_action(self, timestamp: int, actions):    
        """Applies the control action by writing it in the appropriate file
            timestamp (int) : action's timestamp
            actions [(tuple)]: list of tuples representing the actions to be sent
        """
        with open(path.join(self.directory, self.log_filename), 'a') as logFile:
            with open(path.join(self.directory, self.control_filename), 'a') as file:
                for action in actions:
                    control_action = f"{timestamp},{','.join(map(str, action))}\n"
                    file.write(control_action)
                    logFile.write(control_action)
                    file.flush()
                    logFile.flush()
                
