# Related Work

This project sits between classical ball-and-plate control and modern reinforcement-learning-based dynamic manipulation.

The exact combination used here:

- a 7-DoF manipulator,
- a free ball on an end-effector-mounted plate,
- MuJoCo contact dynamics,
- RL with asymmetric actor-critic observations,
- end-effector force/torque sensing,

is fairly specific. The papers below are therefore best read as the closest structural and conceptual neighbors rather than exact replicas of the repository.

## Closest Structural Match

### Spacek et al. (2022)

V. Spacek, J. Vojtesek, A. Gazdos, "Control of Unstable Systems Using a 7 DoF Robotic Manipulator," *Machines*, 2022.

Link:

- https://www.mdpi.com/2075-1702/10/12/1164

Why it matters:

- it uses a 7-DoF manipulator to control a ball-on-plate system,
- it focuses on stabilization and disturbance rejection,
- it explicitly treats the robot dynamics as part of the control problem.

Relation to this repo:

- this repository solves a very similar plant-level problem,
- the main difference is the control method,
- that paper uses a model-based controller derived from a simplified system model,
- this repo uses learned control through MuJoCo simulation and domain randomization.

What this repo does differently:

- it learns through contact-rich simulation instead of deriving a compact analytical controller,
- it uses reward shaping to encode both stabilization and smooth robot behavior,
- it includes explicit penalties on joint motion, actuator effort proxies, racquet velocity, and racquet displacement from the nominal balancing region.

## Ball-and-Plate Control References

### Zarzycki and Lawrynczuk (2021)

K. Zarzycki, M. Lawrynczuk, "Fast Real-Time Model Predictive Control for a Ball-on-Plate Process," *Sensors*, 2021.

Link:

- https://www.mdpi.com/1424-8220/21/12/3959

Why it matters:

- it is a strong reference for the ball-on-plate problem itself,
- it highlights the importance of short control periods, state estimation, and constrained support motion,
- it reinforces that velocity is not secondary to position in this task class.

Relation to this repo:

- the repo also treats the task as dynamic stabilization rather than static centering,
- this is visible in the reward design where contact-gated `ball_centering` is paired with `ball_speed`,
- the simulation timing in `task_parameters.py` uses `timestep = 0.002` and `decimation = 5`, giving a `0.01 s` policy period, which is appropriate for contact-sensitive balancing.

### Awtar et al. (2002)

S. Awtar et al., "Mechatronic Design of a Ball-on-Plate Balancing System," *Mechatronics*, 2002.

Link:

- https://doi.org/10.1016/S0957-4158(01)00062-9

Why it matters:

- it is a classic ball-on-plate reference,
- it frames the benchmark in terms of sensing, mechanics, and feedback control,
- it is useful background for the physics of balancing a free ball on a moving support surface.

Relation to this repo:

- this repo inherits the same physical intuition,
- but it replaces a small dedicated platform with a full manipulator,
- that makes robot kinematics, joint effort, null-space configuration, and end-effector motion regularization part of the control design.

## RL References

### Grimshaw and Oyekan (2021)

J. Grimshaw, J. Oyekan, "Applying Deep Reinforcement Learning to Cable Driven Parallel Robots for Balancing Unstable Loads: A Ball Case Study," *Frontiers in Robotics and AI*, 2021.

Links:

- https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.611203/full
- https://pubmed.ncbi.nlm.nih.gov/33693031/

Why it matters:

- it is one of the closest RL references for balancing a ball on a moving support,
- it also uses RL at the reference-command level rather than raw low-level actuation,
- it is concerned with stabilizing an unstable object under disturbance and model complexity.

Relation to this repo:

- this repo follows a similar architectural idea,
- the joint-space policy outputs joint-space references to position-controlled actuators,
- the Cartesian policy outputs local end-effector position references that are turned into joint commands through differential IK,
- in both cases, learning happens above the low-level actuator realization.

### Zhou et al. (2022)

W. Zhou, X. Lin, H. Wang, G. Zhang, "Learning Ball-balancing Robot Through Deep Reinforcement Learning," arXiv, 2022.

Link:

- https://arxiv.org/abs/2208.10142

Why it matters:

- it supports the use of RL for balancing problems where contacts and recovery dynamics are difficult to model accurately,
- it is a useful reference for learned stabilization in nonlinear, contact-rich systems.

Relation to this repo:

- the mechanical system is different because that paper studies a ballbot rather than a manipulator-mounted plate,
- the conceptual overlap is still useful: RL is attractive when accurate hand-designed controllers become cumbersome under complex contact behavior.

## Sensor and State Considerations

This repository uses asymmetric observations and end-effector force/torque sensing.

That places it in a middle position between:

- classical ball-and-plate systems that often assume direct ball-state measurement,
- and more realistic robotic manipulation settings where object state is partial, delayed, or noisy.

The repo reflects that tradeoff by:

- giving the actor only robot and wrench observations,
- giving the critic privileged ball state in the plate frame,
- injecting actor-side observation noise,
- using contact-aware reward terms instead of purely geometric support heuristics,
- only rewarding plate-frame centering while MuJoCo contact confirms the ball is actually supported.

## What This Repo Is Actually Optimizing

Compared with the papers above, this repository is not just minimizing ball position error.

From the configured rewards and terminations, it is optimizing for:

- keeping the ball near the support center,
- damping ball translation and rotation,
- maintaining plate contact,
- avoiding high robot aggressiveness,
- staying near a local balancing posture rather than sweeping the arm through large excursions.

This is visible in the parameterization:

- `ball_centering = 40.0`
- `ball_speed = -8.0`
- `ball_no_contact = -18.0`
- `ball_height_above_plate = -50.0`
- `plate_drop_under_ball = -2.0`
- `racquet_lin_vel_l2 = -5.0`
- `racquet_dist_from_initial_l2 = -30.0`

along with additional penalties on action smoothness, joint velocity, acceleration, and torque proxies.

## Interpretation of the Current Parameters

The current parameter choices suggest that this repo is trying to learn a local, robust stabilizer rather than a highly aggressive recovery controller.

Evidence:

- the reset distribution starts the ball close to the plate center,
- the termination boundary is substantially larger than the main centering scale,
- disturbances are present, but moderate,
- domain randomization ranges are broad enough for robustness but not so broad that the nominal structure disappears,
- the PPO configuration is compact and conservative rather than over-parameterized.

In other words, the repository is closest in spirit to:

- manipulator-based ball-and-plate stabilization from the classical literature,
- implemented as an RL policy learning support-surface control under realistic contact dynamics.
