# panda_robot_task
Some custom-designed environments for reinforcement learning based on panda-gym.

Pre-trained agents are provided, some of which are from original environments in panda-gym.

May serve as a tutorial for customizing your own environments.

# requirements
panda-gym == 2.0.1

stable-baselines3 == 1.5.1a5

sb3-contrib == 1.5.1a5

# trained agents
Files can be found under the "trained_agents" folder.

To enjoy the trained agents, run xxx_show.py.

To re-train an agent, run xxx_train.py.

## My_PandaReachJointsDense

control-type: joints

### gif

![image](/trained_agents/show/My_PandaReachJointsDense.gif)

### log

![image](/trained_agents/show/My_PandaReachJointsDense_TQC.jpg)

## Two_PandaReachDense

The robot is supposed to pass two points successively.

### gif

![image](/trained_agents/show/Two_PandaReachDense.gif)

### log

![image](/trained_agents/show/Two_PandaReachDense_PPO.jpg)

## Three_PandaReachDense

The robot is supposed to pass three points successively.

### gif

![image](/trained_agents/show/Three_PandaReachDense.gif)

### log

![image](/trained_agents/show/Three_PandaReachDense_PPO.jpg)

## PandaPush

### gif

![image](/trained_agents/show/PandaPush.gif)

### log

![image](/trained_agents/show/PandaPush_SAC.jpg)

## Two_PandaPushDense

The robot is supposed to push the object to pass two points successively.

### gif

![image](/trained_agents/show/Two_PandaPushDense.gif)

### log

![image](/trained_agents/show/Two_PandaPushDense_TQC.jpg)

## Two_Obj_PandaPushDense

The robot is supposed to push two object to their targets respectively.

### gif

![image](/trained_agents/show/Two_Obj_PandaPushDense.gif)

### log

![image](/trained_agents/show/Two_Obj_PandaPushDense_SAC.jpg)

## My_PandaSlideDense

### gif

![image](/trained_agents/show/My_PandaSlideDense.gif)

### log

![image](/trained_agents/show/My_PandaSlideDense_TQC.jpg)

## PandaPickAndPlace

The robot learns to push the object when it is on the table and pick the object when it is in the air.

This is different from the robot trained by dense reward.

### gif

![image](/trained_agents/show/PandaPickAndPlace.gif)

### log

![image](/trained_agents/show/PandaPickAndPlace_SAC.jpg)

## PandaPickAndPlaceDense

### gif

![image](/trained_agents/show/PandaPickAndPlaceDense.gif)

### log

![image](/trained_agents/show/PandaPickAndPlaceDense_SAC.jpg)

## My_TwoPickAndPlaceDense

The robot is supposed to pick the object and carry it to pass two points successively.

### gif

![image](/trained_agents/show/My_TwoPickAndPlaceDense.gif)

### log

![image](/trained_agents/show/My_TwoPickAndPlaceDense_TQC.jpg)

## My_PandaReachPlateJointsDense

The robot is supposed to reach a point while balancing a ball on its plate.

control-type: joints

### gif

![image](/trained_agents/show/My_PandaReachPlateJointsDense.gif)

### log

![image](/trained_agents/show/My_PandaReachPlateJointsDense_PPO.jpg)

## My_PandaReachPlateJointsDense

The robot is supposed to reach two points successively while balancing a ball on its plate.

control-type: joints

### gif

![image](/trained_agents/show/My_TwoPandaReachPlateJointsDense.gif)

### log

![image](/trained_agents/show/My_TwoPandaReachPlateJointsDense_PPO.jpg)
