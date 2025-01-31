import argparse
import json
from nsoran.environments.es_env import EnergySavingEnv


if __name__ == '__main__':
    #######################
    # Parse data #
    #######################
    parser = argparse.ArgumentParser(description="Select the environment")
    args = parser.parse_args()

    # Configuration scenario values
    configuration_path = '../scenario_configurations/es_use_case.json'

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)

    output_folder = '/workspace/ns-o-ran-gymnasium/output'
    
    print('Creating ES Environment')
    env = EnergySavingEnv(ns3_path='/workspace/ns3-mmwave-oran', scenario_configuration=scenario_configuration, output_folder=output_folder, optimized=True)
    num_steps = 1000

    obs, info = env.reset()
    
    print(f'First set of observations {obs}')
    print(f'Info {info}')
    # Action logic
    for step in range(num_steps):
        model_action = []
        cell_states_table = env.datalake.read_table('bsState')
        states_of_interest = []
        # Filter rows only from last timestamp
        for cell_state in cell_states_table:
            if cell_state[0] == env.last_timestamp:
                states_of_interest.append(cell_state)
        model_action = [state[3] for state in states_of_interest]
         
        print(f'Step {step} ', end='', flush=True)
        obs, reward, terminated, truncated, info = env.step(model_action)

        print('done', flush=True)

        print(f'Status t = {step}')
        print(f'Actions {env._compute_action(model_action)}') 
        print(f'Observations {obs}')
        print(f'Reward {reward}')
        print(f'Terminated {terminated}')
        print(f'Truncated {truncated}')
        print(f'Info {info}')

        # If the environment is over, exit
        if terminated:
            break

        # If the episode is up (environment still running), then start another one
        if truncated:
            break # We don't want this outside the training
            obs, info = env.reset()

