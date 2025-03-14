import argparse
import json
from environments.es_env import EnergySavingEnv

if __name__ == '__main__':
    #######################
    # Parse arguments #
    #######################
    parser = argparse.ArgumentParser(description="Run the energy saving environment")
    parser.add_argument("--config", type=str, default="src/environments/scenario_configurations/es_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/workspace/ns3-mmwave-oran",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")

    args = parser.parse_args()

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps
    optimized = args.optimized

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)

    print('Creating ES Environment')
    env = EnergySavingEnv(ns3_path=ns3_path, scenario_configuration=scenario_configuration, 
                          output_folder=output_folder, optimized=optimized)

    print('Environment Created!')

    print('Launch reset ', end='', flush=True)
    obs, info = env.reset()
    print('done')
    
    print(f'First set of observations {obs}')
    print(f'Info {info}')

    # Action logic
    for step in range(2, num_steps):
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

