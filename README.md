# controller_push_IKQP

## Project Description
**Controlled Push of an Object with Franka Emika Panda**

This project implements a controlled push task using a 6-DOF robotic arm (Franka Emika Panda) with a parallel gripper. 
The control architecture utilizes **Instantaneous Differential Inverse Kinematics** posed as a Quadratic Program (QP) for position control, 
combined with a **Hybrid Force/Position** strategy.
A proportional **Admittance Controller** is implemented to regulate contact force along the vertical (Z) axis while executing planar pushing movements.

System 1 (Inner Control Loop): Position Control via Differential IK (QP)

System 2 (Outer Control Loop): Force Regulation via Admittance Control

### Watch the demo videos in the following folder 🎬🎥▶️: [test_videos](./test_videos) 

## 📂 Repository Structure

```text
controller_push_IKQP
├── commons/
│   ├── logger.py             # Thread-safe logging utility
│   └── plotter.py            # Visualization tools
├── controllers/
│   └── ik_qp.py              # Differential IK Solver (QP) & Admittance Controller
├── model/                    # MuJoCo XML models (scene, robot, objects) recovered from repos(see below). 
├── simulation/
│   ├── mujoco_interface.py   # Bridge between Python and MuJoCo engine
│   ├── mujoco_sim.py         # Main simulation loop and multiprocessing handling
│   └── robot_interface.py    # Standardized interface for robot actuation
├── test_videos/              # Demo recordings of displacements
├── example1.py               # Main entry point / Example execution script
└── requirements.txt          # Python dependencies

```

## 🚀 Getting Started

### Prerequisites

* Python 3.11+
* [MuJoCo](https://mujoco.org/) Mujoco py (pip install mujoco)
* `scipy`, `numpy`, `osqp`

### Installation

1. Clone the repository.
2. Install dependencies:
```bash
pip install -r requirements.txt

```

### Usage

1. Enter to the main example script [example1](./example1.py) to start the simulation and control loop. 
2. Write your directory path if cloned repo is inside a project.  
3. Run the script 

This script initializes the environment, connects the `MujocoInterface`, and executes the Finite State Machine (FSM) to push the box.

---

## 🤖 System Architecture

### 1. The Challenge

* **Task:** Controlled push of a non-deformable box (`box_geom`) on a table.
* **Agent:** 6-DOF Robotic Arm + Parallel End-Effector.
* **Sensing:** Joint positions + Contact/Force sensors (5 fingertip pads per finger R and L).
* **Environment:**
  * Table with a box placed at a random position within reachable workspace.
* **Interaction Logic:**
  * Finite State Machine (FSM) to handle approach, contact, and regulation safely.
  * Hybrid Force/Position Control:
    * Position Control in XY (planar push).
    * Force Regulation in Z (vertical contact force).
* **Failure/Mitigation:**
  * Bad pose with retry policy (see FSM).
  * Loss of contact handled by admittance controller.
* **Success Criteria:**
  * Object translates the requested .
  * Stable contact established (confirmed via force pads).
  * Controller handles different initial object poses robustly and different robot initial configurations.
* **Safety Considerations**
  * Robot joint position limits.
  * Robot joint velocity limits.

### 2. Finite State Machine (FSM)

The high-level logic is governed by a State Machine to handle approach, contact, and regulation safely.

| State                   | Description                                                                                                                                         | Transition Condition |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------| --- |
| **1. APPROACH**         | Move EE to a pre-contact position (SO3 Pose Control).                                                                                               | Distance to target < Threshold |
| **2. SEEK CONTACT**     | Move vertically slowly to find the object.                                                                                                          | Force sensors > Threshold (Contact detected) |
| **2.5 POSE RECOVERY**   | (NOT IMPLEMENTED) Failure: Bad object pose. Mitigation: After a time passed w/o contact. This state will retry around a radii (x,y) to seek contact | Force sensors > Threshold (Contact detected) |
| **3. CLOSE GRIPPER**    | Close fingers until firm grip on object.                                                                                                            | Both fingers in contact |
| **4. FORCE REGULATION** | **Hybrid Control**: Push in  while regulating  via admittance.                                                                                      | Target  reached |
| **5. OPEN/RETRACT**     | Release object and move to safe home position.                                                                                                      | Task complete |

*Mitigation Strategy:* If "Seek Contact" fails (bad pose or timeout), the system enters a **Retry Policy**, returning to a safe height and retrying with a radius search.

---

## 🧠 Theoretical Background

### Differential Inverse Kinematics (QP)

We solve the instantaneous kinematics problem by formulating it as a **Quadratic Program (QP)**. This allows us to handle joint limits and velocity bounds strictly.

**The Objective:**
Minimize the error between the desired task-space velocity () and the current robot velocity (), while regularizing joint velocities:
$$\min_{\dot{q}} \quad \frac{1}{2} || (\dot p_d - J(q)\dot{q}) +  (p_d-p_{current}) ||^2_W + \frac{1}{2} \lambda ||\dot{q}||^2$$
Subject to:

Where:

* $p_d$  Is the desired position (velocity here \dot p_d assumed as zero in steady-state).
* $p_current$  Is current robot position (forward kinematics).
* $J(q)$  is the robot Jacobian.
* $Kp$  is the CLIK gain.
* $\lambda$ is a regulating factor (Levy-Levenberg) to regulate the constrained solution.

### Admittance Control (Force Regulation)

To prevent crushing the object or losing contact during the push, we use an **Implicit Force Control** loop around the position controller.
$$\Delta x_z = -k_p (F_{measured} - F_{target})$$
For the Z-axis, the control law acts as a proportional admittance filter:

This displacement $\Delta x_z$ is added to the IK target, making the robot "compliant" vertically:

* If $F_{measured} > F_{target}$, the robot senses a force **greater than reference** (yields).
* If $F_{measured} < F_{target}$, the robot senses a force **lower than reference** (presses).

---

## 📦 Core Classes

* **`MujocoInterface`**: Handles the communication with the simulation, retrieving telemetry (poses, contacts) and sending joint commands.
* **`InverseKinematicsController`**: The solver class. It accepts a target pose and outputs joint velocities using `osqp` to solve the optimization problem described above.
* **`AdmittanceController`**: A dumb proportional controller that modifies the Z-reference based on force feedback.

## 📖 References 

* **System Inspiration**:  Admittance Force Tracking Control for Position-Controlled Robot Manipulators Under
Unknown Environment by SeulJung (2020). 
* **Quaternion Error Calculation CLIK with Lyapunov Theoretical Guarantee**: "The Unit Quaternion: A Useful Tool for Inverse Kinematics of Robot Manipulators" by S. Chiaverini and B. Siciliano

## 🤖 AI Usage

This project utilized AI assistance for the following tasks:
* **Code Debugging:** Debugging certain parts specially with multiprocessing queuing.
* **Code Refactoring:** Cleaning and organizing the codebase to improve structure and readability.
* **Polishing:** Enhancing code quality with type hints, better variable naming, and standardized formatting.
* **Documentation:** Generating this `README.md` by analyzing and transcribing handwritten notes, diagrams, and repository screenshots.

## 🛢️ Useful Repos
* [mujoco_menagerie](https://github.com/vikashplus/furniture_sim) : MuJoCo models of franka robot.
* [furniture_sim](https://github.com/vikashplus/furniture_sim) : Table and objects for simulation env. 
```

```