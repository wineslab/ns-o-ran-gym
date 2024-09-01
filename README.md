# Gymnasium environment for ns-o-ran

This repository contains a demo for a [gymnasium](https://gymnasium.farama.org/) based reinforcement learning environment for the 5G O-RAN architecture through the ns-O-RAN simulator.

## An overview of the code

At a high level: the system can be viewed through its different parts as divided in the `src` folder: 

+ The `base` folder contains `ActionController` an helper class that feeds the agent's control action into the ns3 based simulation environment, as well as the SQL and Base environment logic. More specifically, `datalake.py` contains an SQLite API and `ns_env.py` contains **`NsOranEnv`**, the base environment used to derive every other use case specific environment.
+ The `environments` folder contains `TrafficSteeringEnv`, an environments derived from `NsOranEnv`, implementing the Traffic Steering use case 

To extend the code to allow for gymnasium compliant environments, `NsOranEnv` is created. To support the widest variety of use cases we've created this abstract environment which constitutes the main building block for every new environments. NsOranEnv coordinates the environment and the ns-3 simulation and offers some utilities as well. Briefly, every new environment that is built on NsOranEnv should provide a json file to the constructor (`scenario_configuration`) that will be used to fetch the necessary information for the environment to work. Of course, this json can be fine tuned to provide the necessary parameters for any specific new environment. An example of this can be found on `ts_env.py`, which extends BaseEnv for the O-RAN Traffic Steering use case. For specific details on how each abstract method should be extended by new environments refer to `NsOranEnv` directly.

The base environment also implements the `step()` method. The environment firstly checks if the simulation has not ended. In that case, the new action is computed and the environment updates itself based on the action. NsOranEnv then writes the action to the files shared with ns-3 (this routine is implemented by `ActionController`) and fills the database. As soon as the simulation step is finished, step returns the new observation on which to operate control.  
If the simulation ended, the base environment will indicate that properly signaling termination or truncation. 

This framework allows new environments to be easy to create and test, since the inner details reguarding ns-3 and the its simulation can be handled by the available BaseEnv. 

![](./Docs/environments.svg)

## References

If you use the Gymnasium Base Environment, please reference the following paper:

```
@INPROCEEDINGS{10619796,
  author={Lacava, Andrea and Pietrosanti, Tommaso and Polese, Michele and Cuomo, Francesca and Melodia, Tommaso},
  booktitle={2024 IFIP Networking Conference (IFIP Networking)}, 
  title={Enabling Online Reinforcement Learning Training for Open RAN}, 
  year={2024},
  volume={},
  number={},
  pages={577-582},
  keywords={Training;Cellular networks;Open RAN;Computer architecture;Software;Quality of experience;Telemetry;Open RAN;ns-3;deep reinforcement learning;artificial intelligence;gymnasium},
  doi={10.23919/IFIPNetworking62109.2024.10619796}}
```

If you use ns-O-RAN without the Gym Environment, please reference the following paper:

```

```

If you use the TrafficSteering Environment with no changes, please reference the following paper:

```

```

- Framework presentation: https://openrangym.com/ran-frameworks/ns-o-ran
- Paper of the Framework: https://dl.acm.org/doi/abs/10.1145/3592149.3592161
- Journal about Traffic Steering with ns-O-RAN: https://ieeexplore.ieee.org/document/10102369
- tutorial OSC RIC version E ns-O-RAN connection: https://www.nsnam.org/tutorials/consortium23/oran-tutorial-slides-wns3-2023.pdf
- tutorial Colosseum RIC (i.e., OSC RIC bronze reduced) ns-O-RAN: https://openrangym.com/tutorials/ns-o-ran
- recording of the tutorial OSC RIC version E done at the WNS3 2023: https://vimeo.com/867704832


