"""Run from motion_imitation/retarget_motion to find data correctly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import time

import tensorflow as tf
import numpy as np
import torch

from motion_imitation.utilities import pose3d
from pybullet_utils import transformations
import pybullet
import pybullet_data as pd
from motion_imitation.utilities import motion_util


# import retarget_config_a1 as config
# import retarget_config_laikago as config
# import retarget_config_vision60 as config
import retarget_config_anymal_blender as config

import matplotlib.pyplot as plt

POS_SIZE = 3
ROT_SIZE = 4
DEFAULT_ROT = np.array([0, 0, 0, 1])
FORWARD_DIR = np.array([1, 0, 0])
PLOT = False

GROUND_URDF_FILENAME = "plane_implicit.urdf"

# reference motion
FRAME_DURATION = 0.02 # 50Hz, default, 0.01667
# REF_COORD_ROT = transformations.quaternion_from_euler(0.5 * np.pi, 0, 0)
REF_COORD_ROT = transformations.quaternion_from_euler(0, 0, 0)
REF_POS_OFFSET = np.array([0, 0, 0])
# REF_ROOT_ROT = transformations.quaternion_from_euler(0, 0, 0.47 * np.pi) # jump01, low shorter jump
angle = [0, 0, np.pi] - np.array([-0.030348604592354243, 0.0004194554676767022, -0.1503376313516547])
REF_ROOT_ROT = transformations.quaternion_from_euler(*angle) # highjump01_new

REF_PELVIS_JOINT_ID = 8
REF_NECK_JOINT_ID = 9
REF_HIP_JOINT_IDS = [4, 5, 6, 7] # LF, LH, RF, RH
REF_TOE_JOINT_IDS = [0, 1, 2, 3] # LF, LH, RF, RH
# switch the order of the joints to match the order in the isaacgym: LF, LH, RF, RH
ISAACINDEX = np.concatenate([np.arange(0, 3), np.arange(6, 9), np.arange(3, 6), np.arange(9, 12)])

OUTPUT = True
RECORD = False

RELA_PATH = "/home/zewzhang/codespace/motion_retarget/motion_imitation/retarget_motion/"
LOG_DIR = "retarget_motion/ret_data/tencent_motions/"
mocap_motions = [
  # ["jump01", "blender_data/data_joint_pos_dog_jump_high_001_420.txt",None,None],
  # ["jump01_new", "blender_data/data_joint_pos_dog_jump_high_001_515_low_height_shorter.txt",1,49],
  # ["jump01", "blender_data/data_joint_pos_dog_jump_002_90.txt",20,None],
  ["highjump01_new", "blender_data/data_joint_pos_dog_jump_006_730.txt",5, None], # SIM_TOE_OFFSET_LOCAL: x: -0.05 for hind legs, y: -0.03, REF_POS_SCALE:0.97
  # ["walk01", "blender_data/data_joint_pos_dog_quad_walk_001_3400.txt",None,None],
  # ["trot", "blender_data/data_joint_pos_dog_fast_run_02_004_1500_trot.txt",None,None], # NOTE: not available
  # ["slow_run", "blender_data/data_joint_pos_dog_fast_run_02_004_600_slowrun.txt",None,None], # forward_off: 0.03
  # ["walk02", "blender_data/data_joint_pos_dog_fast_run_02_004_2750_walk.txt",None,None], # forward_off: 0.03
  # ["trot01", "blender_data/data_joint_pos_dog_quad_walk_001_3900_trot.txt",None,None], # forward_off: 0.05
  # ["run01", "blender_data/data_joint_pos_dog_fast_run_02_004_1000.txt",None,None], # forward_off: 0.03, scale: 1.1, tough
  # ["walk03", "blender_data/data_joint_pos_dog_quad_walk_001_3600.txt",None,None], # forward_off: 0.03  
  # ["leftturn", "blender_data/data_joint_pos_dog_quad_walk_001_4250_leftturn.txt",None,None], # shitty
  # ["rightturn01", "blender_data/data_joint_pos_dog_quad_walk_001_10350_rightturn.txt",None,None], # not bad, forward_off: 0.03, scale: 1.1
  # ["rightturn02", "blender_data/data_joint_pos_dog_quad_walk_001_10650_rightturn.txt",None,None], # good, forward_off: 0.03, scale: 1.1
  # ["run01", "blender_data/data_joint_pos_dog_fast_run_02_004_1350.txt",None,None], # forward_off: 0.03, scale: 1.0, tough
]

def build_markers(num_markers):
  marker_radius = 0.02

  markers = []
  for i in range(num_markers):
    if (i == REF_NECK_JOINT_ID) or (i == REF_PELVIS_JOINT_ID)\
        or (i in REF_HIP_JOINT_IDS):
      col = [0, 0, 1, 1]
    elif (i in REF_TOE_JOINT_IDS):
      col = [1, 0, 0, 1]
    else:
      col = [0, 1, 0, 1]

    virtual_shape_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_SPHERE,
                                                  radius=marker_radius,
                                                  rgbaColor=col)
    body_id =  pybullet.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=-1,
                                  baseVisualShapeIndex=virtual_shape_id,
                                  basePosition=[0,0,0],
                                  useMaximalCoordinates=True)
    markers.append(body_id)

  return markers

def get_joint_limits(robot):
  num_joints = pybullet.getNumJoints(robot)
  joint_limit_low = []
  joint_limit_high = []

  if 'anymal_d' in config.URDF_FILENAME:
    joint_range = config.JOINT_RANGE
    joint_limit_low = config.DEFAULT_JOINT_POSE - joint_range / 2
    joint_limit_high = config.DEFAULT_JOINT_POSE + joint_range / 2
  else:
    for i in range(num_joints):
      joint_info = pybullet.getJointInfo(robot, i)
      joint_type = joint_info[2]

      if (joint_type == pybullet.JOINT_PRISMATIC or joint_type == pybullet.JOINT_REVOLUTE):
        joint_limit_low.append(joint_info[8])
        joint_limit_high.append(joint_info[9])


  return joint_limit_low, joint_limit_high

def get_root_pos(pose):
  return pose[0:POS_SIZE]

def get_root_rot(pose):
  return pose[POS_SIZE:(POS_SIZE + ROT_SIZE)]

def get_joint_pose(pose):
  return pose[(POS_SIZE + ROT_SIZE):]

def set_root_pos(root_pos, pose):
  pose[0:POS_SIZE] = root_pos
  return

def set_root_rot(root_rot, pose):
  pose[POS_SIZE:(POS_SIZE + ROT_SIZE)] = root_rot
  return

def set_joint_pose(joint_pose, pose):
  pose[(POS_SIZE + ROT_SIZE):] = joint_pose
  return

def set_pose(robot, pose):
  num_joints = pybullet.getNumJoints(robot)
  root_pos = get_root_pos(pose)
  root_rot = get_root_rot(pose)
  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  for j in range(num_joints):
    j_info = pybullet.getJointInfo(robot, j)
    j_state = pybullet.getJointStateMultiDof(robot, j)

    j_pose_idx = j_info[3]
    j_pose_size = len(j_state[0])
    j_vel_size = len(j_state[1])

    if (j_pose_size > 0):
      j_pose = pose[j_pose_idx:(j_pose_idx + j_pose_size)]
      j_vel = np.zeros(j_vel_size)
      pybullet.resetJointStateMultiDof(robot, j, j_pose, j_vel)

  return

def set_maker_pos(marker_pos, marker_ids):
  num_markers = len(marker_ids)
  assert(num_markers == marker_pos.shape[0])

  for i in range(num_markers):
    curr_id = marker_ids[i]
    curr_pos = marker_pos[i]

    pybullet.resetBasePositionAndOrientation(curr_id, curr_pos, DEFAULT_ROT)

  return

def process_ref_joint_pos_data(joint_pos):
  proc_pos = joint_pos.copy()
  num_pos = joint_pos.shape[0]

  for i in range(num_pos):
    curr_pos = proc_pos[i]
    curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_COORD_ROT)
    curr_pos = pose3d.QuaternionRotatePoint(curr_pos, REF_ROOT_ROT)
    curr_pos = curr_pos * config.REF_POS_SCALE + REF_POS_OFFSET
    proc_pos[i] = curr_pos

  return proc_pos

def retarget_root_pose(ref_joint_pos):
  pelvis_pos = ref_joint_pos[REF_PELVIS_JOINT_ID]
  neck_pos = ref_joint_pos[REF_NECK_JOINT_ID]

  left_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[0]]
  right_shoulder_pos = ref_joint_pos[REF_HIP_JOINT_IDS[2]]
  left_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[1]]
  right_hip_pos = ref_joint_pos[REF_HIP_JOINT_IDS[3]]

  forward_dir = neck_pos - pelvis_pos
  forward_dir[2] *= 0.5  # make the base more stable
  forward_dir += config.FORWARD_DIR_OFFSET
  forward_dir = forward_dir / np.linalg.norm(forward_dir)

  delta_shoulder = left_shoulder_pos - right_shoulder_pos
  delta_hip = left_hip_pos - right_hip_pos
  dir_shoulder = delta_shoulder / np.linalg.norm(delta_shoulder)
  dir_hip = delta_hip / np.linalg.norm(delta_hip)

  left_dir = 0.5 * (dir_shoulder + dir_hip)

  up_dir = np.cross(forward_dir, left_dir)
  up_dir = up_dir / np.linalg.norm(up_dir)

  left_dir = np.cross(up_dir, forward_dir)
  left_dir[2] = 0.0 # make the base more stable
  left_dir = left_dir / np.linalg.norm(left_dir)

  rot_mat = np.array([[forward_dir[0], left_dir[0], up_dir[0], 0],
                      [forward_dir[1], left_dir[1], up_dir[1], 0],
                      [forward_dir[2], left_dir[2], up_dir[2], 0],
                      [0, 0, 0, 1]])

  root_pos = 0.5 * (pelvis_pos + neck_pos)
  #root_pos = 0.25 * (left_shoulder_pos + right_shoulder_pos + left_hip_pos + right_hip_pos)
  root_rot = transformations.quaternion_from_matrix(rot_mat)
  root_rot = transformations.quaternion_multiply(root_rot, config.INIT_ROT)
  root_rot = root_rot / np.linalg.norm(root_rot)

  return root_pos, root_rot

def retarget_pose(robot, default_pose, ref_joint_pos):
  joint_lim_low, joint_lim_high = get_joint_limits(robot)

  root_pos, root_rot = retarget_root_pose(ref_joint_pos)
  root_pos += config.SIM_ROOT_OFFSET
  # root_pos[2] = (root_pos[2] - 0.4) * 0.8 + 0.4

  pybullet.resetBasePositionAndOrientation(robot, root_pos, root_rot)

  inv_init_rot = transformations.quaternion_inverse(config.INIT_ROT)
  heading_rot = motion_util.calc_heading_rot(transformations.quaternion_multiply(root_rot, inv_init_rot))

  tar_toe_pos = []
  for i in range(len(REF_TOE_JOINT_IDS)):
    ref_toe_id = REF_TOE_JOINT_IDS[i]
    ref_hip_id = REF_HIP_JOINT_IDS[i]
    sim_hip_id = config.SIM_HIP_JOINT_IDS[i]
    toe_offset_local = config.SIM_TOE_OFFSET_LOCAL[i]

    ref_toe_pos = ref_joint_pos[ref_toe_id]
    ref_hip_pos = ref_joint_pos[ref_hip_id]

    hip_link_state = pybullet.getLinkState(robot, sim_hip_id, computeForwardKinematics=True)
    sim_hip_pos = np.array(hip_link_state[4])

    toe_offset_world = pose3d.QuaternionRotatePoint(toe_offset_local, heading_rot)

    ref_hip_toe_delta = ref_toe_pos - ref_hip_pos
    sim_tar_toe_pos = sim_hip_pos + ref_hip_toe_delta
    sim_tar_toe_pos[2] = ref_toe_pos[2] * 0.8
    sim_tar_toe_pos += toe_offset_world

    tar_toe_pos.append(sim_tar_toe_pos)

  if 'anymal_d' in config.URDF_FILENAME:
    joint_pose = pybullet.calculateInverseKinematics2(robot, config.SIM_TOE_JOINT_IDS,
                                                    tar_toe_pos,
                                                    # solver=pybullet.IK_DLS,
                                                    # jointDamping=config.JOINT_DAMPING,
                                                    maxNumIterations=2000,
                                                    residualThreshold=1e-16,
                                                    # lowerLimits=list(joint_lim_low),
                                                    # upperLimits=list(joint_lim_high),
                                                    # jointRanges=list(joint_lim_high-joint_lim_low),
                                                    restPoses=list(default_pose))
  else:
    joint_pose = pybullet.calculateInverseKinematics2(robot, config.SIM_TOE_JOINT_IDS,
                                                    tar_toe_pos,
                                                    jointDamping=config.JOINT_DAMPING,
                                                    lowerLimits=joint_lim_low,
                                                    upperLimits=joint_lim_high,
                                                    restPoses=default_pose)
  joint_pose = np.array(joint_pose)

  pose = np.concatenate([root_pos, root_rot, joint_pose])

  return pose, tar_toe_pos

def update_camera(robot):
  base_pos = np.array(pybullet.getBasePositionAndOrientation(robot)[0])
  [yaw, pitch, dist] = pybullet.getDebugVisualizerCamera()[8:11]
  pybullet.resetDebugVisualizerCamera(3, 90, -5, base_pos)
  # pybullet.resetDebugVisualizerCamera(3, 45, -15, base_pos)
  return

def load_ref_data(JOINT_POS_FILENAME, FRAME_START, FRAME_END):
  joint_pos_data = np.loadtxt("retarget_motion/" + JOINT_POS_FILENAME, delimiter=",")

  start_frame = 0 if (FRAME_START is None) else FRAME_START
  end_frame = joint_pos_data.shape[0] if (FRAME_END is None) else FRAME_END
  joint_pos_data = joint_pos_data[start_frame:end_frame]

  return joint_pos_data

def retarget_motion(robot, joint_pos_data, name=None):
  num_frames = joint_pos_data.shape[0]
  last_ref_joint_pos = None
  last_last_ref_joint_pos = None
  tar_toe_pos_list = []
  for f in range(num_frames):
    ref_joint_pos = joint_pos_data[f]
    ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
    ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)
    # if f >= 2:
    #   ref_joint_pos = (ref_joint_pos + last_ref_joint_pos + last_last_ref_joint_pos) / 3
    #   last_last_ref_joint_pos = last_ref_joint_pos.copy()
    #   last_ref_joint_pos = ref_joint_pos.copy()
    # else:
    #   last_last_ref_joint_pos = last_ref_joint_pos.copy() if last_ref_joint_pos is not None else ref_joint_pos.copy()
    #   last_ref_joint_pos = ref_joint_pos.copy()
    curr_pose, tar_toe_pos = retarget_pose(robot, config.DEFAULT_JOINT_POSE, ref_joint_pos)
    tar_toe_pos_list.append(tar_toe_pos)
    set_pose(robot, curr_pose)
    # lin_vel = (last_curr_pose[:3] - curr_pose[:3]) / FRAME_DURATION
    # dof_vel = (last_curr_pose[-12:] - curr_pose[-12:]) / FRAME_DURATION
    if f == 0:
      pose_size = curr_pose.shape[-1]
      new_frames = np.zeros([num_frames, pose_size])
      new_frames_vel = np.zeros([num_frames, pose_size-1])
    new_frames[f] = curr_pose

  plot_foot_base_pos(tar_toe_pos_list, new_frames[:, :7], name)
  new_frames = get_new_frames(robot, tar_toe_pos_list, new_frames[:, :7]) # update new frames
  new_frames[:, 0:2] -= new_frames[0, 0:2]
  base_lin_vel = (new_frames[1:, :3] - new_frames[:-1, :3]) / FRAME_DURATION
  dof_vel = (new_frames[1:, -12:] - new_frames[:-1, -12:]) / FRAME_DURATION
  new_frames_vel[:, :3] = np.concatenate((base_lin_vel[[0], :], base_lin_vel), axis=0)
  new_frames_vel[:, -12:] = np.concatenate((dof_vel[[0], :], dof_vel), axis=0)
  new_frames_vel[:, 3:6] = get_base_ang_vel_from_base_quat(new_frames[:, 3:7], dt=FRAME_DURATION, target_frame="global")
  projected_gravity = get_projected_gravity(new_frames[:, 3:7])
  dof_pos = (new_frames[:, -12:] - config.DEFAULT_JOINT_POSE)[:, ISAACINDEX]
  dof_vel = new_frames_vel[:, -12:][:, ISAACINDEX]
  for i in range(6):
    new_frames_vel[1:, i] = moving_average(new_frames_vel[:, i], 2)
  saved_frames = np.concatenate([new_frames[:, :7], 
                                  new_frames_vel[:, :6], 
                                  projected_gravity, 
                                  dof_pos, 
                                  dof_vel], axis=1)
  return new_frames, saved_frames

def output_motion(frames, out_filename, num_steps=150):
  # show the traj of joint
  plt.figure()
  plt.title("LF")
  plt.plot(frames[:, 16:19])
  plt.figure()
  plt.title("LH")
  plt.plot(frames[:, 19:22])
  plt.figure()
  plt.plot(frames[:, 22:25])
  plt.title("RF")
  plt.figure()
  plt.plot(frames[:, 25:28])
  plt.title("RH")
  plt.figure()
  plt.plot(frames[:, 7:10])
  plt.title("Linear Velocity")
  plt.figure()
  plt.plot(frames[:, 7:10], label="Angular Velocity")
  plt.title("Angular Velocity")
  plt.show()
  with open(LOG_DIR + out_filename, "w") as f:
    f.write("{\n")
    f.write("\"LoopMode\": \"Wrap\",\n")
    f.write("\"FrameDuration\": " + str(FRAME_DURATION) + ",\n")
    f.write("\"EnableCycleOffsetPosition\": true,\n")
    f.write("\"EnableCycleOffsetRotation\": true,\n")
    f.write("\n")

    f.write("\"Frames\":\n")

    f.write("[")
    num_step = 0
    data = torch.zeros(
            1, num_steps, frames.shape[1], dtype=torch.float, requires_grad=False
        )
    while True:  # repete num_trajs times
      for i in range(frames.shape[0]-1):
        curr_frame = frames[i+1]
        data[0, num_step, :] = torch.tensor(curr_frame, dtype=torch.float)
        if i != 0:
          f.write(",")
        f.write("\n  [")

        for j in range(frames.shape[1]):
          curr_val = curr_frame[j]
          if j != 0:
            f.write(", ")
          f.write("%.5f" % curr_val)

        f.write("]")

        num_step += 1
        if num_step >= num_steps:
          f.write("\n]")
          f.write("\n}")    
          # save as pt file
          torch.save(data, LOG_DIR + "motion_data_" + out_filename.replace(".txt", ".pt"))
          return 

def get_base_ang_vel_from_base_quat(base_quat, dt, target_frame="local"):
    """
    Get the base angular velocity from the base quaternion.
    args:
        base_quat:      torch.Tensor (num_trajs, num_steps, 4)
        dt:             float
    returns:
        base_ang_vel:   torch.Tensor (num_trajs, num_steps, 3) expressed in the target frame
    """
    if len(base_quat.shape) < 3:
      base_quat = np.expand_dims(base_quat, axis=0)
    num_trajs, num_steps, _ = base_quat.shape
    mapping = np.zeros((num_trajs, num_steps, 3, 4), dtype=np.float)
    mapping[:, :, :, -1] = -base_quat[:, :, :-1]
    if target_frame == "local":
        mapping[:, :, :, :-1] = get_skew_matrix(-base_quat[:, :, :-1].reshape((-1, 3))).reshape((num_trajs, num_steps, 3, 3))
    elif target_frame == "global":
        mapping[:, :, :, :-1] = get_skew_matrix(base_quat[:, :, :-1].reshape((-1, 3))).reshape((num_trajs, num_steps, 3, 3))
    else:
        raise ValueError(f"Unknown target frame {target_frame}")
    mapping[:, :, :, :-1] += np.tile(np.eye(3, dtype=np.float), (num_trajs, num_steps, 1, 1)) * base_quat[:, :, -1].reshape((num_trajs, num_steps, 1, 1))
    base_ang_vel = 2 * mapping[:, :-1, :, :] @ (base_quat[:, 1:, :] - (base_quat[:, :-1, :]) / dt).reshape((num_trajs, num_steps-1, 4, 1))
    base_ang_vel = np.concatenate((base_ang_vel[:, [0]], base_ang_vel), axis=1).squeeze(-1)
    return base_ang_vel

def get_skew_matrix(vec):
    """Get the skew matrix of a vector."""
    matrix = np.zeros((vec.shape[0], 3, 3), dtype=np.float)
    matrix[:, 0, 1] = -vec[:, 2]
    matrix[:, 0, 2] = vec[:, 1]
    matrix[:, 1, 0] = vec[:, 2]
    matrix[:, 1, 2] = -vec[:, 0]
    matrix[:, 2, 0] = -vec[:, 1]
    matrix[:, 2, 1] = vec[:, 0]
    return matrix

def get_projected_gravity(base_quat):
  num_steps = base_quat.shape[0]
  gravity_vec = to_torch(get_axis_params(-1.0, 2)).repeat((num_steps, 1))
  proj_vec = quat_rotate_inverse(torch.tensor(base_quat, dtype=torch.float), gravity_vec)
  return proj_vec.detach().cpu().numpy()

def to_torch(x, dtype=torch.float, requires_grad=False):
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)

def get_axis_params(value, axis_idx, x_value=0., dtype=np.float, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))

def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c
  
def moving_average(data, window_size):
    """Calculate the moving average using NumPy."""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, 'valid')

def plot_foot_base_pos(tar_toe_pos_list, base_pos_list, name):
  tar_toe_pos_list = np.array(tar_toe_pos_list)
  new_tar_toe_pos_list, new_base_pos_list = preprocess_foot_base_pos(tar_toe_pos_list, base_pos_list)
  plt.figure()
  plt.plot(tar_toe_pos_list[:, 0, 2], label="LF_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 1, 2], label="LH_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 2, 2], label="RF_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 3, 2], label="RH_orig", linestyle="--")
  plt.plot(base_pos_list[:, 2], label="Base_orig", linestyle="--")
  plt.plot(new_tar_toe_pos_list[:, 0, 2], label="LF", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 1, 2], label="LH", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 2, 2], label="RF", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 3, 2], label="RH", linestyle="-")
  plt.plot(new_base_pos_list[:, 2], label="Base", linestyle="-")
  plt.legend()
  plt.savefig(f"retarget_motion/{name}_foot_base_pos_z_new.png")
  
  plt.figure()
  plt.plot(tar_toe_pos_list[:, 0, 1], label="LF_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 1, 1], label="LH_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 2, 1], label="RF_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 3, 1], label="RH_orig", linestyle="--")
  plt.plot(base_pos_list[:, 1], label="Base_orig", linestyle="--")
  plt.plot(new_tar_toe_pos_list[:, 0, 1], label="LF", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 1, 1], label="LH", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 2, 1], label="RF", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 3, 1], label="RH", linestyle="-")
  plt.plot(new_base_pos_list[:, 1], label="Base", linestyle="-")
  plt.legend()
  plt.savefig(f"retarget_motion/{name}_foot_base_pos_y_new.png")
  
  plt.figure()
  plt.plot(tar_toe_pos_list[:, 0, 0], label="LF_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 1, 0], label="LH_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 2, 0], label="RF_orig", linestyle="--")
  plt.plot(tar_toe_pos_list[:, 3, 0], label="RH_orig", linestyle="--")
  plt.plot(base_pos_list[:, 0], label="Base_orig", linestyle="--")
  plt.plot(new_tar_toe_pos_list[:, 0, 0], label="LF", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 1, 0], label="LH", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 2, 0], label="RF", linestyle="-")
  plt.plot(new_tar_toe_pos_list[:, 3, 0], label="RH", linestyle="-")
  plt.plot(new_base_pos_list[:, 0], label="Base", linestyle="-")
  plt.legend()
  plt.savefig(f"retarget_motion/{name}_foot_base_pos_x_new.png")

def preprocess_foot_base_pos(orig_tar_toe_pos_list, orig_base_pos_list):
  tar_toe_pos, base_pos = preprocess_foot_base_pos_z(orig_tar_toe_pos_list, orig_base_pos_list)
  final_tar_toe_pos , final_base_pos = preprocess_foot_base_pos_xy(tar_toe_pos, base_pos)
  return final_tar_toe_pos, final_base_pos

def preprocess_foot_base_pos_z(orig_tar_toe_pos_list, orig_base_pos_list):
  tar_toe_pos_list = orig_tar_toe_pos_list.copy()
  base_pos_list = orig_base_pos_list.copy()
  # highjump01_new
  base_orig_height = 0.45
  base_height_scale = 0.7
  foot_height_scale = np.array([0.5, 0.5, 0.5, 0.5])
  jumping_frames = np.array([16, 28, 20, 28]) # LF, LH, RF, RH # highjump01_new
  # jump01_new
  # base_orig_height = 0.6
  # base_height_scale = 0.5
  # foot_height_scale = np.array([0.7, 0.5, 0.7, 0.5])
  # jumping_frames = np.array([0, 11, 0, 11]) # LF, LH, RF, RH # jump01_new
  base_pos_list[:, 2] = (base_pos_list[:, 2] - base_orig_height) * base_height_scale + base_orig_height # 0.4 is the original height of base
  tar_toe_pos_min = np.min(tar_toe_pos_list[:, :, 2], axis=0)
  # tar_toe_pos_max = np.max(tar_toe_pos_list[:, :, 2], axis=0)
  tar_toe_pos_list[:, :, 2] = tar_toe_pos_list[:, :, 2] - tar_toe_pos_min.clip(max=0)
  for i, jumping_frame in enumerate(jumping_frames):
    tar_toe_pos_list[jumping_frame:, i, 2] = tar_toe_pos_list[jumping_frame:, i, 2] * foot_height_scale[i]
    # if i == 1 or i == 3:
    #   num_frames = tar_toe_pos_list.shape[0]
    #   tar_toe_pos_list[jumping_frame:, i, 2] = tar_toe_pos_list[:num_frames-jumping_frame, i-1, 2]

  return tar_toe_pos_list, base_pos_list

def preprocess_foot_base_pos_xy(orig_tar_toe_pos_list, orig_base_pos_list):
  # slow down the forward speed
  tar_toe_pos_list = orig_tar_toe_pos_list.copy()
  base_pos_list = orig_base_pos_list.copy()
  # highjump01_new
  xy_scale = 0.7
  # jump01_new
  # xy_scale = 0.6
  local_tar_toe_pos_list = tar_toe_pos_list - base_pos_list[:, :3].reshape(-1, 1, 3)
  base_pos_list[:, :2] = (base_pos_list[:, :2] - base_pos_list[0, :2]) * xy_scale + base_pos_list[0, :2]
  global_tar_toe_pos_list = local_tar_toe_pos_list + base_pos_list[:, :3].reshape(-1, 1, 3)
  
  return global_tar_toe_pos_list, base_pos_list

def get_new_frames(robot, tar_toe_pos_list, base_pos_list):
  tar_toe_pos_list = np.array(tar_toe_pos_list)
  num_frames = tar_toe_pos_list.shape[0]
  new_tar_toe_pos_list, new_base_pos_list = preprocess_foot_base_pos(tar_toe_pos_list, base_pos_list)
  new_frames = np.zeros([num_frames, 7 + 12])
  
  for k in range(num_frames):
    pybullet.resetBasePositionAndOrientation(robot, new_base_pos_list[k, :3], base_pos_list[k, 3:7])
    joint_pose = pybullet.calculateInverseKinematics2(robot, config.SIM_TOE_JOINT_IDS,
                                                    new_tar_toe_pos_list[k, :, :],
                                                    maxNumIterations=2000,
                                                    residualThreshold=1e-16)
    joint_pose = np.array(joint_pose)
    pose = np.concatenate([new_base_pos_list[k, :7], joint_pose])
    set_pose(robot, pose)
    new_frames[k] = pose
    
  return new_frames
  

def main(argv):
  
  p = pybullet
  if RECORD:
    p.connect(p.GUI, options="--width=2560 --height=1440")
  else:
    p.connect(p.GUI, options="--width=1920 --height=1080 --mp4=\"retarget.mp4\" --mp4fps=60")
  p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)

  pybullet.setAdditionalSearchPath(pd.getDataPath())
  

  while True:
    
    for mocap_motion in mocap_motions:
      pybullet.resetSimulation()
      pybullet.setGravity(0, 0, 0)
    
      ground = pybullet.loadURDF(GROUND_URDF_FILENAME)
      robot = pybullet.loadURDF(config.URDF_FILENAME, config.INIT_POS, config.INIT_ROT)
      # Set robot to default pose to bias knees in the right direction.
      set_pose(robot, np.concatenate([config.INIT_POS, config.INIT_ROT, config.DEFAULT_JOINT_POSE]))

      p.removeAllUserDebugItems()
      print("mocap_name=", mocap_motion[0])
      joint_pos_data = load_ref_data(mocap_motion[1],mocap_motion[2],mocap_motion[3])
    
      num_markers = joint_pos_data.shape[-1] // POS_SIZE
      marker_ids = build_markers(num_markers)
    
      retarget_frames, saved_frames = retarget_motion(robot, joint_pos_data, mocap_motion[0])
      f = 0
      num_frames = joint_pos_data.shape[0]
      max_frames = num_frames * 5 # 10000 # 200 # max(150, num_frames)
      # max_frames = 10000
      # for _ in range (min(5*num_frames, max_frames)):
      if OUTPUT:
        output_motion(saved_frames, f"{mocap_motion[0]}.txt", num_steps=max_frames)
      for _ in range (max_frames):
        time_start = time.time()
     
        f_idx = f % num_frames
        print("Frame {:d}".format(f_idx))
    
        ref_joint_pos = joint_pos_data[f_idx]
        ref_joint_pos = np.reshape(ref_joint_pos, [-1, POS_SIZE])
        ref_joint_pos = process_ref_joint_pos_data(ref_joint_pos)
    
        pose = retarget_frames[f_idx]
    
        set_pose(robot, pose)
        set_maker_pos(ref_joint_pos, marker_ids)
    
        # update_camera(robot)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,1)
        f += 1
    
        time_end = time.time()
        sleep_dur = FRAME_DURATION - (time_end - time_start)
        sleep_dur = max(0, sleep_dur)
    
        time.sleep(sleep_dur)
        #time.sleep(0.5) # jp hack
      for m in marker_ids:
        p.removeBody(m)
      marker_ids = []

    break

  pybullet.disconnect()

  return


if __name__ == "__main__":
  tf.app.run(main)

