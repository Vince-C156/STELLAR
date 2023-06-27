# STELLAR : Single-Stage-Trajectory-Execution-Looking-to-Learning-in-Autonomous-Rendezvous
Reinforcement Learning for Spacecraft Docking Trajectories with Proximal Policy Optimization. Learning a model-free approach to unify model switching standard.

## What is space docking?
Space docking is the process of connecting two spacecraft together in space. It involves bringing the spacecraft into close proximity, aligning their docking ports, and using a docking mechanism to secure the connection. This maneuver enables crew transfers, resupply missions, module attachment, and the formation of larger structures in space. Rendezvous, alignment, capture, sealing, and pressurization are key steps in the docking process. Space docking plays a crucial role in space exploration, satellite servicing, and the assembly of space stations, requiring precise coordination and specialized docking systems to achieve a successful connection in the challenging conditions of space.

![Alt Text](https://raw.githubusercontent.com/Vince-C156/PLANNING-AUTONOMOUS-SPACECRAFT-RENDEZVOUS-AND-DOCKING-TRAJECTORIES-VIA-REINFORCEMENT-LEARNING/main/misc/arpod_diagram.png)

## Who am I
Hi!, my name is Vince. I am an undergraduate researcher specializing in control with reinforcement learning (RL). Currently pursuing a Bachelor of Science in Computer Science at the University of California, Irvine, I am passionate about exploring the intersection of machine learning and control systems. My research focuses on leveraging RL techniques to develop intelligent control algorithms for various applications. I am fascinated by the potential of RL to enhance the capabilities of autonomous systems. Additionally I also hold strong experience in computer vision from object detection to computational geometry (3D reconstruction) through my senior design team UAV Forge, where I have held co-lead and lead responsibilities for the vision subteam.

## PRESENTATION AT THE UCI UROP SYMPOSIUM
![UROP Symposium Poster Presentation](https://raw.githubusercontent.com/Vince-C156/PLANNING-AUTONOMOUS-SPACECRAFT-RENDEZVOUS-AND-DOCKING-TRAJECTORIES-VIA-REINFORCEMENT-LEARNING/418cbda5307087b857e35ee0f2a6eb84fa8ee8d6/misc/ARPOD.pptx%20(2).png)

## Abstract
We present a Proximal Policy Optimization (PPO) Reinforcement Learning algorithm for three-dimensional autonomous spacecraft trajectory planning. Specifically,
we consider a chaser spacecraft performing a rendezvous and docking mission with
a target spacecraft on a circular orbit. This reinforcement learning approach utilizes
an actor and critic method to plan safe trajectories for the chaser spacecraft given
constraints on its motion, including maximum thrust and line-of-sight constraints.
We consider a fully actuated chaser spacecraft capable of applying continuous thrust
in all three dimensions. Given this action space, we train a PPO model to perform
rendezvous and docking maneuvers using spacecraft relative motion dynamics. We
describe the training procedure and environment in detail and present results of nu-
merous simulations showing that the trained model produces successful rendezvous
and docking trajectories that satisfy line-of-sight constraints, even with significant
random variations in initial conditions


## Resources to research lab and publication
Link to presentation at AAS Guidance, Navigation and Control (GN&C) Conference, 2023 [[slides](https://docs.google.com/presentation/d/1PnKZOP27mqJtLG8vqgT4AOFqMHhQYM8q/edit?usp=sharing&ouid=109971343941983675406&rtpof=true&sd=true)]


Link to publication [here]

Advisor Professor Copp [[website](https://dcopp.eng.uci.edu/index.html)]

First Author Vincent Beau Chen

Chen, V., Phillips, S. A., Copp, D. A., Planning Autonomous Spacecraft Rendezvous and Docking Trajectories via Reinforcement Learning Proceedings of the AAS Guidance, Navigation and Control (GN&C) Conference, 2023.

This work was supported by UCI’s Undergraduate Research Opportunities Program (UROP) and
the Air Force Office of Scientific Research (AFOSR) through the Air Force Research Laboratory
16

## Final results published
<div style="display: flex; justify-content: space-between;">
  <figure style="text-align: center; width: 20%;">
    <img src="https://raw.githubusercontent.com/Vince-C156/PLANNING-AUTONOMOUS-SPACECRAFT-RENDEZVOUS-AND-DOCKING-TRAJECTORIES-VIA-REINFORCEMENT-LEARNING/1e744758088842f3eeb2445dbdfc8c2a8fb2c7a3/misc/view1newlos.svg" alt="Image 1" style="width: 20%;">
    <figcaption style="font-size: small;">100 Simulated Missions</figcaption>
  </figure>
  <figure style="text-align: center; width: 20%;">
    <img src="https://raw.githubusercontent.com/Vince-C156/PLANNING-AUTONOMOUS-SPACECRAFT-RENDEZVOUS-AND-DOCKING-TRAJECTORIES-VIA-REINFORCEMENT-LEARNING/323d2620649c80c0c6f77b5b48dc91ae0750c7d6/misc/run15shortzoom%20(1).svg" alt="Image 2" style="width: 20%;">
    <figcaption style="font-size: small;">1 Simulated Mission</figcaption>
  </figure>
  <figure style="text-align: center; width: 20%;">
    <img src="https://raw.githubusercontent.com/Vince-C156/PLANNING-AUTONOMOUS-SPACECRAFT-RENDEZVOUS-AND-DOCKING-TRAJECTORIES-VIA-REINFORCEMENT-LEARNING/4dcce89938472fa7a986b863b52275e5e6001590/misc/view1oldlos.svg" alt="Image 3" style="width: 20%;">
    <figcaption style="font-size: small;">100 Simulated Missions Conventional</figcaption>
  </figure>
</div>

## CORE ENVIRONMENT DURING TIME OF RELEASE (6/10/23)
Python3.10 Ubuntu20.04 Cuda-12.0 Vulkan-1.2 at the time of development.

Archetecture : x86_64

## INSTALLATION INSTRUCTIONS

Once fully setup the github repo should just be a simple pip install for any python above Python3.8

[python3 -m pip install git+git@github.com:Vince-C156/PLANNING-AUTONOMOUS-SPACECRAFT-RENDEZVOUS-AND-DOCKING-TRAJECTORIES-VIA-REINFORCEMENT-LEARNING.git]

## Usage guide




