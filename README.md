# DynamoRobo
Dynamic Path Planning for Mobile Robots with Deep Reinforcement Learning

Traditional path planning algorithms for mobile robots are not effective to solve high-
dimensional problems, and suffer from slow convergence and complex modelling. Therefore, it
is highly essential to design a more efficient algorithm to realize intelligent path planning of
mobile robots. 

This work proposes an improved path planning algorithm, which is based on the
algorithm of Soft Actor-Critic (SAC). It attempts to solve a problem of poor robot performance
in complicated environments with static and dynamic obstacles. 

This work designs an improved reward function to enable mobile robots to quickly avoid obstacles and reach targets by using
state dynamic normalization and priority replay buffer techniques. To evaluate its performance,
a Pygame-based simulation environment is constructed. 

The proposed method is compared with a Proximal Policy Optimization (PPO) algorithm in the simulation environment. Experimental
results demonstrate that the cumulative reward of the proposed method is much higher than
that of PPO, and it is also more robust than PPO.
