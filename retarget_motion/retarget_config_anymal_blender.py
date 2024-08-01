import numpy as np

URDF_FILENAME = "retarget_motion/anymal_d/urdf/anymal_d.urdf"

REF_POS_SCALE = 1.0 # 1.2 is very good!, 1.4 is also good
INIT_POS = np.array([0, 0, 0.66])
INIT_ROT = np.array([0, 0, 0, 1.0])

SIM_TOE_JOINT_IDS = [
    49,  # right hand, RF
    71,  # right foot, RH
    38,  # left hand, LF
    60,  # left foot, LH
]
SIM_HIP_JOINT_IDS = [40, 62, 29, 51]
SIM_ROOT_OFFSET = np.array([0, 0, 0.00])
SIM_TOE_OFFSET_LOCAL = [
    np.array([0, -0.05, 0.0]), # default y: -0.05
    np.array([0, -0.05, 0.0]),
    np.array([0, 0.05, 0.0]), # default y: 0.05
    np.array([0, 0.05, 0.0])
]
JOINT_RANGE = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])

DEFAULT_JOINT_POSE = np.array([-0.13859, 0.480936, -0.761428,    # FL
                               0.13859, 0.480936, -0.761428,    # FR
                               -0.13859, -0.480936, 0.761428,   # HL
                               0.13859, -0.480936, 0.761428,]) # HR
JOINT_DAMPING = [0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01,
                 0.1, 0.05, 0.01]

FORWARD_DIR_OFFSET = np.array([0, 0, 0.03]) # np.array([0, 0, 0.05])
