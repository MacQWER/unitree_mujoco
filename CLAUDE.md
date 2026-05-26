# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

### C++ Simulator (simulate/)

**Dependencies:**
```bash
sudo apt install libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev
```

**Build:**
```bash
cd simulate
ln -s ~/.mujoco/mujoco-3.3.6 mujoco  # Link mujoco
mkdir build && cd build
cmake ..
make -j4
```

**Run simulator:**
```bash
./unitree_mujoco -r go2 -s scene_terrain.xml
```

**Run test:**
```bash
./test  # Sends unitree_go message, each motor outputs 1Nm torque
```

### Python Simulator (simulate_python/)

**Dependencies:**
```bash
pip3 install mujoco pygame mediapy  # mediapy for recording
```

**Run simulator:**
```bash
cd simulate_python
python3 ./unitree_mujoco.py
```

**Run test:**
```bash
python3 ./test/test_unitree_sdk2.py
```

### Examples

**Python stand example:**
```bash
python3 example/python/stand_go2.py      # Simulation (domain_id=1, lo)
python3 example/python/stand_go2.py enp3s0  # Real robot
```

**C++ stand example:**
```bash
cd example/cpp && mkdir build && cd build && cmake .. && make -j4
./stand_go2      # Simulation
./stand_go2 enp3s0  # Real robot
```

**ROS2 example:**
```bash
source ~/unitree_ros2/setup.sh
cd example/ros2
colcon build
./install/stand_go2/bin/stand_go2
```

## Architecture Overview

This is a robotics simulator for Unitree robots (Go2, B2, H1, H1-2, G1, Go2w, B2w) that bridges **Unitree SDK2** with **MuJoCo**. It enables sim-to-real development by exposing the same DDS interfaces used on physical robots.

### Core Components

**`simulate/` (C++)** and **`simulate_python/`** - Two parallel implementations:
- **`unitree_sdk2_bridge.h` / `unitree_sdk2py_bridge.py`**: DDS bridge that subscribes to `rt/lowcmd` and publishes `rt/lowstate`, `rt/sportmodestate`, `rt/wirelesscontroller`. Maps motor commands to MuJoCo actuators.
- **`main.cc` / `unitree_mujoco.py`**: Main simulation loop with MuJoCo physics and GLFW rendering.

**DDS Message Types:**
- `LowCmd` / `LowState`: Motor control (12 DOF for quadrupeds, 20+ for humanoids)
- `SportModeState`: Robot position and velocity
- `WirelessController`: Gamepad input
- `IMUState`: Torso IMU at `rt/secondary_imu` (G1 only)

**Robot Detection:** The bridge auto-detects robot type by motor count (>20 motors = G1 humanoid, uses `unitree_hg` IDL; otherwise `unitree_go` IDL).

### Control Flow

```
User Controller → rt/lowcmd (DDS) → UnitreeBridge → MuJoCo actuators
                                          ↓
MuJoCo sensors → UnitreeBridge → rt/lowstate (DDS) → User Subscriber
```

### Configuration

- **C++:** `simulate/config.yaml` - robot type, domain_id, interface, joystick settings
- **Python:** `simulate_python/config.py` - same settings plus recording options (EGL headless, video capture)

**Wireless Command Modes (Python):**
- `USE_JOYSTICK=1`: Physical gamepad via pygame
- `USE_KEYBOARD=1`: External keyboard publisher (run `scripts/go2_joystick2/keyboard_wireless_controller.py`)
- Both 0: Fixed command values (`FIXED_WIRELESS_CMD_VX/VY/YAW`)

**Sim-to-Real Pattern:**
```python
if len(sys.argv) < 2:
    ChannelFactoryInitialize(1, "lo")  # Simulation
else:
    ChannelFactoryInitialize(0, sys.argv[1])  # Real robot interface
```

### Scripts Directory

- **`go2_trot/`**: ONNX-based trotting controller (`trot_go2_apg.py` + `Go2OnnxController` class). Policy file: `onnx/go2_apg_policy.onnx`
- **`go2_joystick2/`**: Keyboard wireless controller (`keyboard_wireless_controller.py`) + run script (`run_go2_joystick2.py`)

### Terrain Tool

`terrain_tool/terrain_generator.py` generates scene XML with stairs, rough ground, and Perlin noise heightfields. Output goes to `unitree_robots/<robot>/scene_terrain.xml`.