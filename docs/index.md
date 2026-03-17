# MjLab Kinova Ball Balancing

This project provides reinforcement-learning environments for balancing a free ball on a plate mounted to a Kinova Gen3 end-effector in MuJoCo through MjLab.

It is best understood as a coupled robotics and mechanics problem:

- the robot controls plate motion,
- the plate motion changes contact forces on the ball,
- the policy must both recenter and damp the ball,
- the implementation is designed for robust RL rather than idealized scripted control.

## What This Documentation Covers

This site is organized around the actual problem the code is solving:

- how to install and run training,
- what the environment exposes as observations and actions,
- how rewards, resets, and terminations are defined,
- how contact-aware reward shaping changes the learned behavior,
- why the plate frame and contact terms matter physically,
- where the relevant code lives.

## Main Task Variants

- `Mjlab-BallBalancing-Kinova`: default baseline joint-space policy
- `Mjlab-BallBalancing-Kinova-Baseline`: explicit baseline alias
- `Mjlab-BallBalancing-Kinova-Cartesian`: Cartesian end-effector action variant
- `Mjlab-BallBalancing-Kinova-BaselineNoRobotModelRand`: baseline without robot model randomization
- `Mjlab-BallBalancing-Kinova-BaselineNoRand`: baseline with deterministic resets and no randomization
- `Mjlab-BallBalancing-Kinova-Play`: play-mode alias

## Reading Order

If you are new to the project, start here:

1. [Getting Started](getting-started.md)
2. [Task Overview](environment/overview.md)
3. [MDP Design](environment/mdp.md)
4. [Physics and Control](environment/physics.md)
5. [Related Work](environment/related-work.md)
6. [Project Structure](code/structure.md)
