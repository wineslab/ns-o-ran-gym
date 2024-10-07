import pytest
import json
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from nsoran.environments.ts_env import TrafficSteeringEnv
import warnings

"""
Ignore Warnings format:

CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
    ]
]
"""
CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "This version of the mujoco environments depends on the mujoco-py bindings, which are no longer maintained and may stop working. Please upgrade to the v4 versions of the environments (which depend on the mujoco python bindings instead), unless you are trying to precisely replicate previous works).",
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is infinity. This is probably too high.",
        # NOTE: Why is this -infinity?
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "Casting input x to numpy array.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
    ]
]

def test_all_env_api():

    configuration_path = './scenario_configurations/ts_use_case.json'
    with open(configuration_path) as params_file:
        params = params_file.read()
    scenario_configuration = json.loads(params)
    assert scenario_configuration

    output_folder = '/workspace/ns-o-ran-gymnasium/output'

    env = TrafficSteeringEnv(ns3_path='/workspace/ns3-mmwave-oran', scenario_configuration=scenario_configuration, output_folder=output_folder, optimized=False)
    assert env

    with warnings.catch_warnings(record=True) as caught_warnings:
        check_env(env, skip_render_check=True)
        env.close()

    for warning in caught_warnings:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")
