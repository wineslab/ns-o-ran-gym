import argparse
import json
from gymnasium.utils.env_checker import check_env
from environments.ts_env import TrafficSteeringEnv

if __name__ == '__main__':
    #######################
    # Parse arguments #
    #######################
    parser = argparse.ArgumentParser(description="Run the traffic steering environment")
    parser.add_argument("--config", type=str, default="src/environments/scenario_configurations/ts_use_case.json",
                        help="Path to the configuration file")
    parser.add_argument("--output_folder", type=str, default="output",
                        help="Path to the output folder")
    parser.add_argument("--ns3_path", type=str, default="/workspace/ns3-mmwave-oran",
                        help="Path to the ns-3 mmWave O-RAN environment")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of steps to run in the environment")
    parser.add_argument("--optimized", action="store_true",
                        help="Enable optimization mode")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    configuration_path = args.config
    output_folder = args.output_folder
    ns3_path = args.ns3_path
    num_steps = args.num_steps # maximum number of steps for each environment is calculated using the indication periodicity, we refer here to training steps
    optimized = args.optimized
    verbose = args.verbose

    try:
        with open(configuration_path) as params_file:
            params = params_file.read()
    except FileNotFoundError:
        print(f"Cannot open '{configuration_path}' file, exiting")
        exit(-1)

    scenario_configuration = json.loads(params)

    print('Creating TS Environment')
    env = TrafficSteeringEnv(ns3_path=ns3_path, scenario_configuration=scenario_configuration,
                             output_folder=output_folder, optimized=optimized, verbose=verbose)

    print('Environment Created!')

    print('Launch reset ', end='', flush=True)
    obs, info = env.reset()
    print('done')

    print(f'First set of observations {obs}')
    print(f'Info {info}')

    for step in range(2, num_steps):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        print(f'Step {step} ', end='', flush=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print('done', flush=True)

        print(f'Status t = {step}')
        print(f'Actions {env._compute_action(action)}')  # used here only for visualization purposes
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
