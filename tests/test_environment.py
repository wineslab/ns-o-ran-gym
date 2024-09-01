import argparse
import json
from gymnasium.utils.env_checker import check_env
from nsoran.environments.ts_env import TrafficSteeringEnv

if __name__ == '__main__':
    #######################
    # Parse data #
    #######################
    parser = argparse.ArgumentParser(description="Select the environment")
    args = parser.parse_args()

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # pd.set_option('expand_frame_repr', False)

    configuration_path = './scenario_configurations/ts_use_case.json'

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)

    output_folder = '/workspace/ns-o-ran-network-gym/output'
    
    print('Creating TS Environment')
    env = TrafficSteeringEnv(ns3_path='/workspace/ns3-mmwave-oran', scenario_configuration=scenario_configuration, output_folder=output_folder, optimized=False)

    num_steps = 1000

    print('Environment Created!')

    print('Launch reset ', end='', flush=True)
    obs, info = env.reset()
    print('done')
    
    print(f'First set of observations {obs}')
    print(f'Info {info}')

    for step in range(num_steps):

        action = env.action_space.sample()  # agent policy that uses the observation and info
        print(f'Step {step} ', end='', flush=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print('done', flush=True)

        print(f'Status t = {step}')
        print(f'Actions {env._compute_action(action)}') # used here only for visualization purposes
        print(f'Observations {obs}')
        print(f'Reward {reward}')
        print(f'Terminated {terminated}')
        print(f'Truncated {truncated}')
        print(f'Info {info}')

        # If the environment is end, exit
        if terminated:
            break

        # If the episode is up (environment still running), then start another one
        if truncated:
            break # We don't want this outside the training
            obs, info = env.reset()

    # check_env(env)
