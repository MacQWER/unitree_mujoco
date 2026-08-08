ROBOT = "go2" # Robot name, "go2", "b2", "b2w", "h1", "go2w", "g1" 
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml" # Robot scene
DOMAIN_ID = 1 # Domain id
INTERFACE = "lo" # Interface 

USE_JOYSTICK = 0 # Simulate Unitree WirelessController using a gamepad
JOYSTICK_TYPE = "xbox" # support "xbox" and "switch" gamepad layout
JOYSTICK_DEVICE = 0 # Joystick number

# Keyboard publisher settings (used by scripts/go2_joystick2/keyboard_wireless_controller.py)
USE_KEYBOARD = 0 # Use external keyboard publisher scripts/go2_joystick2/keyboard_wireless_controller.py
KEYBOARD_CMD_MAX_VX = 1.5
KEYBOARD_CMD_MAX_VY = 0.2
KEYBOARD_CMD_MAX_YAW = 3.141592653589793
KEYBOARD_CMD_STEP_VX = 0.25
KEYBOARD_CMD_STEP_VY = 0.1
KEYBOARD_CMD_STEP_YAW = 0.6

# Fixed wireless command used when USE_JOYSTICK == 0 and USE_KEYBOARD == 0.
FIXED_WIRELESS_CMD_VX = 0.5  # m/s
FIXED_WIRELESS_CMD_VY = 0.0  # m/s
FIXED_WIRELESS_CMD_YAW = 0.0  # rad (target yaw)

PRINT_SCENE_INFORMATION = True # Print link, joint and sensors information of robot
ENABLE_ELASTIC_BAND = False # Virtual spring band, used for lifting h1

SIMULATE_DT = 0.002  # Need to be larger than the runtime of viewer.sync()
VIEWER_DT = 0.02  # 50 fps for viewer

# Headless / recording options
HEADLESS_EGL = True  # True: set MUJOCO_GL=egl and run without interactive viewer
MAX_SIM_TIME = 20.0  # seconds, <=0 means run until viewer closes (or forever in headless)

ENABLE_RECORDING = True
RECORD_PATH = "./record.mp4"
RECORD_FPS = 50
RECORD_WIDTH = 640
RECORD_HEIGHT = 480
RECORD_AUTO_START = True
RECORD_START_TIME_SEC = 0.0  # used when RECORD_AUTO_START is False
RECORD_END_TIME_SEC = -1.0  # <=0 means no end-time limit
RECORD_DURATION_SEC = -1.0  # >0 overrides RECORD_END_TIME_SEC

# Camera tracking for recording
TRACKING_CAMERA = True
TRACKING_BODY_NAME = "base_link"  # fallback to "torso_link" if not found
TRACKING_DISTANCE = 2.5
TRACKING_AZIMUTH = 120.0
TRACKING_ELEVATION = -20.0
