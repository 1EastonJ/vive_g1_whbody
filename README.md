# Mink_RL: Whole-Body Teleoperation Framework

This repository provides a full-body teleoperation system that combines reinforcement learning (RL) for leg control with inverse kinematics (IK) for upper-body motion.  
The implementation is inspired by the paper *Homie: Whole-body Teleoperation for Humanoids*.

## 🧩 Demo

![Demo](https://github.com/user-attachments/assets/2c2f80c0-7ed7-4cab-882a-3f150e35b139)


---

## Overview

- **Lower Body (RL-based)**  
  Controlled by a pre-trained RL policy from Legged Gym, enabling walking, turning, and balancing.

- **Upper Body (IK-based)**  
  Controlled by Mink IK, allowing torso twist and smooth arm movements such as reaching up, down, and forward.

- **Integration**  
  Both subsystems run within MuJoCo for unified physics and synchronized whole-body control.

---

## Capabilities

| Component | Control Method | Features |
|------------|----------------|-----------|
| Upper body | Mink IK | Torso twist, arm reach up/down, hand movement |
| Lower body | RL policy | Walking, turning, balancing |
| Teleoperation | Pose input | Real-time control and coordination |

---

## Setup

All setup steps are automated in `instuction.sh`.

```bash
# Create and activate environment, install dependencies
chmod +x instuction.sh
./instuction.sh
````

This will:

* Create a `.venv` virtual environment (using uv)
* Activate it automatically
* Install dependencies from `MujocoDeploy/requirements.txt`

---

## Run the System

After setup, run the deployment script:

```bash
cd MujocoDeploy
python mujoco_deploy_g1.py
```

This will launch the MuJoCo simulation and initialize:

* The RL leg policy
* The IK-based upper body controller
* The teleoperation interface

---

## Requirements

* Python 3.8+
* MuJoCo 3.x
* uv package manager

---

## Reference

* Homie: Whole-Body Teleoperation for Humanoids
* Mink: Differentiable Inverse Kinematics Library
* Legged Gym / IsaacGym: Reinforcement Learning locomotion framework

---
