import sys
import time
import collections
import yaml
import torch
import numpy as np
import mujoco
import mujoco.viewer
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "RL" / "legged_gym"))


from legged_gym import LEGGED_GYM_ROOT_DIR
import mink
from pathlib import Path
import socket
import orjson




# Joints used in old training (legs + torso + arms, no fingers)
OLD_POLICY_JOINTS = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

def compute_tracker_velocity(tracker_data, last_pos, last_time, last_cmd=None):
    """
    Compute discrete tracker velocity command with sticky bidirectional hysteresis:
    - vx >  1.0 → start moving forward (cmd = [1,0,0])
    - vx < -1.0 → start moving backward (cmd = [-1,0,0])
    - stays at previous state until the *opposite* threshold is crossed.
    """
    current_time = time.time()
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    height_cmd = 0.9
    base_height_offset = 0.85
    vx = 0.0

    # get VR tracker data
    if tracker_data.get("tracker") is not None and "pos" in tracker_data["tracker"]:
        raw_pos = np.array(tracker_data["tracker"]["pos"], dtype=np.float32)
        pos = raw_pos.copy()
        pos[2] += base_height_offset

        if last_pos is not None and last_time is not None:
            dt = current_time - last_time
            if dt > 1e-4:
                vel = (pos - last_pos) / dt
                vx = vel[0]

                if last_cmd is None:
                    last_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                # --- sticky hysteresis state logic ---
                cmd = last_cmd.copy()

                if last_cmd[0] >= 1.0:  # currently moving forward
                    if vx < -1.0:       # strong reverse trigger
                        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                elif last_cmd[0] <= -1.0:  # currently moving backward
                    if vx > 1.0:           # strong forward trigger
                        cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                elif np.allclose(last_cmd, [0.0, 0.0, 0.0]):
                    if vx > 1.0:
                        cmd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    elif vx < -1.0:
                        cmd = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

        # --- compute height (continuous) ---
        height_cmd = float(np.clip(pos[2], 0.45, 1.05))

        # print debug info every frame
        print(f"[VR CMD] vx={vx:.3f} m/s → cmd={cmd} | height={height_cmd:.3f}")

        return cmd, height_cmd, pos, current_time

    print(f"[VR CMD] (no data) → cmd={cmd} | height={height_cmd:.3f}")
    return cmd, height_cmd, last_pos, current_time







def extract_yaw_pitch_quat(q):
    """
    Decompose a world quaternion into yaw (around Z) and pitch (around X).
    Returns a recombined quaternion with only those two components.
    """
    # Convert quaternion to rotation matrix
    rot_flat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_flat, q)
    rot = rot_flat.reshape(3, 3)

    # Extract yaw and pitch (Z-up coordinate system)
    pitch = np.arctan2(-rot[2, 1], rot[2, 2])   # X-axis
    yaw   = -np.arctan2(rot[1, 0], rot[0, 0])   # Z-axis (flipped to match VR frame)

    # Build quaternions
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    yaw_quat   = np.array([cy, 0.0, 0.0, sy])
    pitch_quat = np.array([cp, sp, 0.0, 0.0])

    # Combine pitch first, then yaw
    combined = np.zeros(4)
    mujoco.mju_mulQuat(combined, pitch_quat, yaw_quat)
    return combined / np.linalg.norm(combined)



def world_to_local_quat(q_world, parent_quat):
    """Convert world-space quaternion into local frame under parent_quat."""
    inv_parent = np.zeros(4)
    mujoco.mju_negQuat(inv_parent, parent_quat)
    q_local = np.zeros(4)
    mujoco.mju_mulQuat(q_local, inv_parent, q_world)
    return q_local


def recv_latest_vr(sock):
    """Read latest UDP packets for left, right, and torso trackers."""
    latest = {"left": None, "right": None, "tracker": None}
    while True:
        try:
            msg, _ = sock.recvfrom(512)
            vr_data = orjson.loads(msg)
            role = vr_data.get("role")
            if role in latest:
                latest[role] = vr_data
        except (BlockingIOError, socket.timeout, ValueError):
            break
    return latest


def load_config(config_path):
    """Load and process the YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Process paths with LEGGED_GYM_ROOT_DIR
    for path_key in ['policy_path', 'xml_path']:
        config[path_key] = config[path_key].format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    
    # Convert lists to numpy arrays where needed
    array_keys = ['kps', 'kds', 'default_angles', 'cmd_scale', 'cmd_init']
    for key in array_keys:
        config[key] = np.array(config[key], dtype=np.float32)
    
    return config

def local_to_world(pos_local, pelvis_pos, pelvis_quat):
    """Transform a local offset (in pelvis frame) to world coordinates."""
    # Convert quaternion to rotation matrix
    rot = np.zeros((3, 3))
    mujoco.mju_quat2Mat(rot.flatten(), pelvis_quat)
    return pelvis_pos + rot.dot(pos_local)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_rotate_inverse(q, v):
    """Rotate vector v by the inverse of quaternion q"""
    w = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]
    
    q_conj = np.array([w, -x, -y, -z])
    
    return np.array([
        v[0] * (q_conj[0]**2 + q_conj[1]**2 - q_conj[2]**2 - q_conj[3]**2) +
        v[1] * 2 * (q_conj[1] * q_conj[2] - q_conj[0] * q_conj[3]) +
        v[2] * 2 * (q_conj[1] * q_conj[3] + q_conj[0] * q_conj[2]),
        
        v[0] * 2 * (q_conj[1] * q_conj[2] + q_conj[0] * q_conj[3]) +
        v[1] * (q_conj[0]**2 - q_conj[1]**2 + q_conj[2]**2 - q_conj[3]**2) +
        v[2] * 2 * (q_conj[2] * q_conj[3] - q_conj[0] * q_conj[1]),
        
        v[0] * 2 * (q_conj[1] * q_conj[3] - q_conj[0] * q_conj[2]) +
        v[1] * 2 * (q_conj[2] * q_conj[3] + q_conj[0] * q_conj[1]) +
        v[2] * (q_conj[0]**2 - q_conj[1]**2 - q_conj[2]**2 + q_conj[3]**2)
    ])

def get_gravity_orientation(quat):
    """Get gravity vector in body frame"""
    gravity_vec = np.array([0.0, 0.0, -1.0])
    return quat_rotate_inverse(quat, gravity_vec)

def compute_observation(d, config, action, cmd, height_cmd, model, joint_names):
    """Compute observation using exactly the same joint set as old policy"""
    qj_list, dqj_list = [], []

    # Collect joint states by name (ignore extra new joints)
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = model.jnt_qposadr[jid]
        dadr = model.jnt_dofadr[jid]
        qj_list.append(d.qpos[qadr])
        dqj_list.append(d.qvel[dadr])

    qj = np.array(qj_list, dtype=np.float32)
    dqj = np.array(dqj_list, dtype=np.float32)
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()
    n_joints = len(qj)

    # Pad or trim defaults
    if len(config['default_angles']) < n_joints:
        padded_defaults = np.zeros(n_joints, dtype=np.float32)
        padded_defaults[:len(config['default_angles'])] = config['default_angles']
    else:
        padded_defaults = config['default_angles'][:n_joints]

    qj_scaled = (qj - padded_defaults) * config['dof_pos_scale']
    dqj_scaled = dqj * config['dof_vel_scale']
    gravity_orientation = get_gravity_orientation(quat)
    omega_scaled = omega * config['ang_vel_scale']

    # keep old observation layout
    single_obs_dim = 3 + 1 + 3 + 3 + n_joints + n_joints + 12
    single_obs = np.zeros(single_obs_dim, dtype=np.float32)
    single_obs[0:3] = cmd[:3] * config['cmd_scale']
    single_obs[3:4] = np.array([height_cmd])
    single_obs[4:7] = omega_scaled
    single_obs[7:10] = gravity_orientation
    single_obs[10:10+n_joints] = qj_scaled
    single_obs[10+n_joints:10+2*n_joints] = dqj_scaled
    single_obs[10+2*n_joints:10+2*n_joints+12] = action

    return single_obs, single_obs_dim



def main():

    # Load configuration
    config = load_config("g1.yaml")

    # --- UDP setup for VR data ---
    UDP_IP = "0.0.0.0"
    PORT = 5005
    print(f"📡 Listening for VR data on UDP {UDP_IP}:{PORT}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, PORT))
    sock.setblocking(False)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
    tracker_data = {"left": None, "right": None, "tracker": None}


    
    # Load robot model
    m = mujoco.MjModel.from_xml_path(config['xml_path'])
    d = mujoco.MjData(m)
    m.opt.timestep = config['simulation_dt']

    # Hide all mocap body geometries (red spheres)
    for i in range(m.nbody):
        if m.body_mocapid[i] != -1:  # mocap body
            geom_start = m.body_geomadr[i]
            geom_count = m.body_geomnum[i]
            for g in range(geom_start, geom_start + geom_count):
                m.geom_rgba[g, 3] = 0.0  # set alpha = 0 (invisible)


    configuration = mink.Configuration(m)
    configuration.update(d.qpos)

    hands = ["right_palm", "left_palm"]
    feet = ["right_foot", "left_foot"]
    hands_mid = [m.body(f"{h}_target").mocapid[0] for h in hands]
    feet_mid = [m.body(f"{f}_target").mocapid[0] for f in feet]

    hand_tasks = [mink.FrameTask(h, "site", position_cost=35.0, orientation_cost=8.0, lm_damping=0.6) for h in hands]
    feet_tasks = [mink.FrameTask(f, "site", position_cost=80.0, orientation_cost=3.0, lm_damping=1.0) for f in feet]

    for hand_task in hand_tasks:
        hand_task.position_cost = 20.0   # 从 200 提高到 800，确保目标优先
        hand_task.orientation_cost = 10.0
        hand_task.lm_damping = 0.3
        hand_task.weight_distribution = "jacobian"

    tasks = hand_tasks + feet_tasks
    solver = "daqp"


    

    # # === JOINT ORDER CHECK ===
    # print("\n=== JOINT ORDER CHECK ===")
    # for jid in range(m.njnt):
    #     jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
    #     jadr = m.jnt_qposadr[jid]
    #     print(f"{jid:2d}: {jname:30s} | qpos_adr = {jadr}")
    # print("===========================\n")

    # for j in ["left_shoulder_roll_joint", "right_shoulder_roll_joint"]:
    #     jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    #     print(j, m.jnt_axis[jid])



    # ----------------------------------------------------------
    # Set initial joint positions (stand pose from default_angles)
    # ----------------------------------------------------------
    print("\nSetting initial joint positions based on config['default_angles']...")
    for i, jname in enumerate([
        'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
    ]):
        if i < len(config['default_angles']):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qpos_adr = m.jnt_qposadr[jid]
            d.qpos[7 + qpos_adr] = config['default_angles'][i]
            print(f"{jname:30s} qpos_init -> {config['default_angles'][i]:.3f}")
    print("Initial posture applied.\n")

    print("\n=== Left shoulder initial angles ===")
    for j in ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint"]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        qadr = m.jnt_qposadr[jid]
        print(f"{j:35s} qpos_init = {d.qpos[qadr + 7]:.3f}")

    # Optional: manually adjust if any angle starts near its lower limit
    # For example, slightly open the shoulder roll:
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")]] += 0.2
    print("Adjusted left_shoulder_roll_joint +0.2 rad for better range\n")


    # --- Improve left shoulder initial pose (more open and raised) ---
    # Pitch: forward/backward swing; Roll: lift up; Yaw: twist
    shoulder_pitch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_pitch_joint")
    shoulder_roll_id  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_roll_joint")
    shoulder_yaw_id   = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_shoulder_yaw_joint")

    d.qpos[m.jnt_qposadr[shoulder_pitch_id]] += 0.3   # move arm slightly forward
    d.qpos[m.jnt_qposadr[shoulder_roll_id]]  += 0.5   # lift arm outward
    d.qpos[m.jnt_qposadr[shoulder_yaw_id]]   += 0.2   # open arm outward
    print("[INIT] Adjusted left shoulder pose for better range (pitch+0.3, roll+0.5, yaw+0.2)")




    # --- Define lower-body joints to match the trained policy ---
    leg_joint_names = [
        'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint'
    ]



    for name in leg_joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos_adr = m.jnt_qposadr[jid]
        print(f"{name:30s} qpos_init = {d.qpos[7 + qpos_adr]:.3f}")


    print("\n=== Joint Axis and Range Debug ===")
    for jname in leg_joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        axis = m.jnt_axis[jid]
        jrange = m.jnt_range[jid]
        print(f"{jname:30s} axis={axis} range={np.round(jrange,3)}")
    print("===================================")



    # Get qpos and qvel indices for these joints
    leg_qpos_indices = []
    leg_qvel_indices = []

    for jname in leg_joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        leg_qpos_indices.append(m.jnt_qposadr[jid])
        leg_qvel_indices.append(m.jnt_dofadr[jid])

    print("\n=== Arm Joint Order (for debugging symmetry) ===")
    for jname in [
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
        'left_elbow_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_joint'
    ]:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        print(f"{jname:35s} qpos_adr={m.jnt_qposadr[jid]:2d} | qvel_adr={m.jnt_dofadr[jid]:2d}")


    # Check number of joints
    n_joints = d.qpos.shape[0] - 7
    # print(f"Robot has {n_joints} joints in MuJoCo model")
    
    # Initialize variables
    action = np.zeros(config['num_actions'], dtype=np.float32)
    target_dof_pos = config['default_angles'].copy()
    cmd = config['cmd_init'].copy()
    height_cmd = config['height_cmd']
    
    # Initialize observation history
    # Only compute leg state (subset of qpos/qvel)
    qpos_leg = d.qpos[7 + np.array(leg_qpos_indices)]
    qvel_leg = d.qvel[6 + np.array(leg_qvel_indices)]
    n_leg_joints = len(leg_joint_names)

    # single_obs, single_obs_dim = compute_observation(
    #     d, config, action, cmd, height_cmd, n_leg_joints,
    # )

    # ✅ 强制使用旧策略的 27 个 joint
    n_joints = len(OLD_POLICY_JOINTS)
    print(f"[INFO] Forcing n_joints = {n_joints} (old policy joint set)")

    single_obs, single_obs_dim = compute_observation(
        d, config, action, cmd, height_cmd, m, OLD_POLICY_JOINTS
    )




    # single_obs, single_obs_dim = compute_observation(d, config, action, cmd, height_cmd, n_joints)
    
    obs_history = collections.deque(maxlen=config['obs_history_len'])
    for _ in range(config['obs_history_len']):
        obs_history.append(np.zeros(single_obs_dim, dtype=np.float32))
    
    # Prepare full observation vector
    obs = np.zeros(config['num_obs'], dtype=np.float32)
    
    # Load policy
    policy = torch.jit.load(config['policy_path'])
    
    counter = 0

        # ✅ Automatic joint movement test to find mapping
    # print("\n=== 🔍 TESTING WHICH JOINT EACH POLICY OUTPUT CONTROLS ===")
    # test_action = np.zeros(config['num_actions'], dtype=np.float32)
    # with mujoco.viewer.launch_passive(m, d) as viewer:
    #     for i in range(config['num_actions']):
    #         test_action[:] = 0.0
    #         test_action[i] = 0.5  # move a little
    #         target_dof_pos = test_action * config['action_scale'] + config['default_angles']

    #         # Apply target pose to all leg joints
    #         for j, qpos_idx in enumerate(leg_qpos_indices):
    #             d.qpos[qpos_idx + 7*0] = target_dof_pos[j]
            
    #         mujoco.mj_forward(m, d)
    #         viewer.sync()

    #         print(f"Now moving policy output index {i}: {leg_joint_names[i] if i < len(leg_joint_names) else 'unknown'}")
    #         input("Press Enter to move to the next joint...")
    # print("=== Joint test finished. Close viewer to continue simulation. ===\n")


    render_every = 5  # render once every 5 simulation steps

    # ✅ Add posture regularization and joint limits
    posture_task = mink.PostureTask(m, cost=0.3)
    posture_task.set_target_from_configuration(configuration)
    limits = [mink.ConfigurationLimit(m)]


    # ✅ Initialize targets before first solve
    # ✅ 躯干 / 骨盆 / COM 稳定任务（弱化以释放肩部自由度）
    # ✅ 躯干 / 骨盆 / COM 稳定任务（弱化以释放肩部自由度）
    pelvis_task = mink.FrameTask(
        "pelvis", "body",
        position_cost=3.0,
        orientation_cost=10.0,   # ❗ zero out to free yaw twist
        lm_damping=1.2
    )

    torso_task = mink.FrameTask(
        "torso_link", "body",
        position_cost=0.0,
        orientation_cost=50.0,
        lm_damping=1.0
    )


    com_task = mink.ComTask(cost=2.0)   # 从 6 降低到 2.0

    tasks.extend([posture_task, pelvis_task, torso_task, com_task])

    # --- Hand tasks ---
    for hand_task in hand_tasks:
        hand_task.position_cost = 15.0
        hand_task.orientation_cost = 3.0
        hand_task.lm_damping = 0.5

    # --- Torso task ---
    torso_task.position_cost = 0.0
    torso_task.orientation_cost = 25.0
    torso_task.lm_damping = 1.2

    # --- Pelvis task ---
    pelvis_task.position_cost = 1.0
    pelvis_task.orientation_cost = 5.0

    # --- COM and posture ---
    com_task.cost = np.ones(3) * 1.0                      # vector (x,y,z)
    posture_task.cost = np.ones(configuration.nv) * 0.2  # all DOFs uniform


    # ✅ 初始化目标
    pelvis_task.set_target_from_configuration(configuration)
    torso_task.set_target_from_configuration(configuration)
    com_task.set_target(d.subtree_com[1].copy())
    # ⚠️ 从 6 降低到 2，保持整体平衡但不过度约束

    tasks.extend([posture_task, pelvis_task, torso_task, com_task])

    # Control gains
    Kp_up, Kd_up = 120.0, 1.0
    integration_gain = 8
    num_leg_dof = 12

    # --- Hand height offset setup (so base_z = actual punch height) ---
    left_palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
    right_palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

    # Measure site positions
    mujoco.mj_forward(m, d)
    left_palm_z = d.site_xpos[left_palm_id][2]
    right_palm_z = d.site_xpos[right_palm_id][2]

    # Compute offset so base_z means "hand punch center" not wrist site
    # (you can tweak 0.05–0.1 depending on how far the site is from hand center)
    hand_z_offset = 0.1

    print(f"[INFO] Hand z offset applied = {hand_z_offset:.3f} m (site below fingertip)")



    # Store the first valid VR hand positions as baseline (outside main loop)
    initial_hand_pos = {"left": None, "right": None}

    tracker_last = None
    tracker_last_time = None
    cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    move_state = "idle"  # forward / backward / idle

    with mujoco.viewer.launch_passive(m, d) as viewer:

        start = time.time()
        while viewer.is_running() and time.time() - start < config['simulation_duration']:
            for _ in range(render_every):
                step_start = time.time()

                # --- RL lower-body control ---
                q_leg = d.qpos[7:7+config['num_actions']]
                dq_leg = d.qvel[6:6+config['num_actions']]
                d.ctrl[:12] = pd_control(
                    target_dof_pos,
                    q_leg,
                    config['kps'],
                    np.zeros_like(config['kps']),
                    dq_leg,
                    config['kds']
                )

                # --- 🧠 Receive latest VR data ---
                new_data = recv_latest_vr(sock)
                for role in ["left", "right", "tracker"]:
                    if new_data[role] is not None:
                        tracker_data[role] = new_data[role]

                for role in ["left", "right"]:
                    if tracker_data[role] is not None and initial_hand_pos[role] is None:
                        initial_hand_pos[role] = np.array(tracker_data[role]["pos"], dtype=float)


                # --- 🧭 Tracker-based velocity sensing ---

                # Get instantaneous measurement
                inst_cmd, height_cmd, tracker_last, tracker_last_time = compute_tracker_velocity(
                    tracker_data, tracker_last, tracker_last_time
                )

                # --- persistent state variables ---
                if "vx_buffer" not in locals():
                    vx_buffer = collections.deque(maxlen=10)
                    same_dir_count = 0
                    target_dir = 0
                    stand_cooldown = 0  # countdown timer for stand state

                vx = inst_cmd[0]
                vx_buffer.append(vx)
                vx_avg = np.mean(vx_buffer)

                # --- direction detection ---
                if vx_avg > 0.3:
                    dir_sign = 1
                elif vx_avg < -0.3:
                    dir_sign = -1
                else:
                    dir_sign = 0

                # --- maintain direction counter ---
                if dir_sign == target_dir:
                    same_dir_count += 1
                else:
                    same_dir_count = 0
                    target_dir = dir_sign

                # --- cooldown countdown ---
                if stand_cooldown > 0:
                    stand_cooldown -= 1

                # --- decision logic ---
                if same_dir_count >= 5 and stand_cooldown == 0:
                    if target_dir == 1:
                        if move_state == "backward":
                            # stop first, then wait a bit before moving again
                            move_state = "stand"
                            cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            stand_cooldown = 10  # stay standing for 10 timesteps
                        elif move_state in ["idle", "stand"]:
                            move_state = "forward"
                            cmd = np.array([1.0, 0.0, 0.0], dtype=np.float32)

                    elif target_dir == -1:
                        if move_state == "forward":
                            move_state = "stand"
                            cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                            stand_cooldown = 10  # stay standing for 10 timesteps
                        elif move_state in ["idle", "stand"]:
                            move_state = "backward"
                            cmd = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

                print(f"[STATE] move_state={move_state:8s} | vx_avg={vx_avg:+.3f} | "
                    f"target_dir={target_dir:+d} | same_dir_count={same_dir_count:2d} | "
                    f"cooldown={stand_cooldown:2d} | cmd={cmd}")





                

                # Outside the simulation loop (after mujoco.mj_forward(m, d))
                # --- Record initial COM position ---
                com_id = 1  # usually body 1 = base / pelvis subtree COM
                com_init = d.subtree_com[com_id].copy()

                # --- 获取机器人质心（COM）和骨盆姿态 ---
                pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                com_pos = d.subtree_com[pelvis_body_id].copy()
                pelvis_quat = d.xquat[pelvis_body_id].copy()

                

                # --- 从四元数中提取机器人朝向的偏航角（绕 Z 轴的旋转） ---
                # 这里只取偏航角，使得手只随着机器人绕 Z 轴旋转，不受俯仰/翻滚影响
                yaw = np.arctan2(
                    2.0 * (pelvis_quat[0]*pelvis_quat[3] + pelvis_quat[1]*pelvis_quat[2]),
                    1.0 - 2.0 * (pelvis_quat[2]**2 + pelvis_quat[3]**2)
                )
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                R_yaw = np.array([
                    [cos_y, -sin_y, 0.0],
                    [sin_y,  cos_y, 0.0],
                    [0.0,    0.0,   1.0]
                ])

                # --- 定义相对于质心的左右手局部偏移量（单位：米） ---
                # 你可以调整这些偏移量来控制手相对于身体的位置
                # --- 定义相对于质心的左右手局部偏移量（单位：米） ---
                # 调整这些偏移量来让手在身体前方一点，并稍微抬高
                left_offset_local  = np.array([0.15,  0.22, 0.05])   # 左手：前+外+上
                right_offset_local = np.array([0.15, -0.22, 0.05])   # 右手：前+外+上



                # --- ✋ Update hand mocap targets from VR ---
                # --- ✋ Update hand mocap targets (COM-follow + VR offset) ---
                # --- ✋ Update hand mocap targets (COM-follow + full 3D VR offset) ---
                # --- ✋ Update hand mocap targets (COM-follow + full 3D VR offset) ---
                for i, role in enumerate(["right", "left"]):
                    if tracker_data[role] is None:
                        continue

                    # === VR raw data ===
                    vr_pos = np.array(tracker_data[role]["pos"], dtype=float)
                    vr_quat = np.array(tracker_data[role]["quat"], dtype=float)

                    # === Torso reference (for yaw rotation only) ===
                    torso_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
                    torso_pos = d.xpos[torso_body_id].copy()
                    torso_quat = d.xquat[torso_body_id].copy()

                    # Extract torso yaw (rotation about z)
                    torso_yaw = np.arctan2(
                        2 * (torso_quat[0] * torso_quat[3] + torso_quat[1] * torso_quat[2]),
                        1 - 2 * (torso_quat[2]**2 + torso_quat[3]**2),
                    )
                    R_yaw = np.array([
                        [np.cos(torso_yaw), -np.sin(torso_yaw), 0.0],
                        [np.sin(torso_yaw),  np.cos(torso_yaw), 0.0],
                        [0.0,                0.0,               1.0],
                    ])

                    # === Compute base target in torso frame (XY follows torso yaw) ===
                    base_target = torso_pos + R_yaw.dot(
                        left_offset_local if role == "left" else right_offset_local
                    )

                    # === Compute VR offset ===
                    if initial_hand_pos[role] is not None:
                        vr_offset = vr_pos - initial_hand_pos[role]
                    else:
                        vr_offset = np.zeros(3)

                    # === Apply VR offset (rotated by torso yaw, fixed Z alignment) ===
                    vr_offset_body = R_yaw.dot(vr_offset)
                    final_pos = base_target + vr_offset_body

                    # --- height and forward tuning ---
                    final_pos[2] += 0.45
                    final_pos[0] += 0.25
                    d.mocap_pos[hands_mid[i]] = final_pos
                    d.mocap_quat[hands_mid[i]] = vr_quat

                    R_hand_flat = np.zeros(9)
                    mujoco.mju_quat2Mat(R_hand_flat, vr_quat)
                    R_hand = R_hand_flat.reshape(3, 3)
                    t_hand = final_pos - torso_pos

                    R_local = R_yaw.T @ R_hand
                    t_local = R_yaw.T @ t_hand
                    t_local *= 2.3
                    hand_quat_local = np.zeros(4)
                    mujoco.mju_mat2Quat(hand_quat_local, R_local.flatten())
                    wxyz_xyz = np.concatenate([hand_quat_local, t_local])
                    hand_target_local = mink.SE3(wxyz_xyz=wxyz_xyz)
                    hand_tasks[i].set_target(hand_target_local)



                    # --- optional debug to verify motion amplitude ---
                    # if i == 0:
                    #     print(f"[DEBUG] hand_z={final_pos[2]:.3f}, hand_x={final_pos[0]:.3f}")





                        # Optional debug print to verify solver sees correct offset
                        # if i == 0:  # only print for right hand
                        #     hand_world = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_palm")]
                        #     err = np.linalg.norm(hand_world - d.mocap_pos[hands_mid[i]])
                        #     print(f"[DEBUG] Hand world error: {err:.3f} m")






                # --- 🟩 Update torso orientation from green tracker ---
                # --- 🟩 Update torso orientation from green tracker (yaw + pitch only) ---
                # --- 🟩 Update torso orientation from green tracker (freeze roll) ---
                # --- 🟩 Update torso orientation from green tracker (freeze roll only) ---
                if tracker_data["tracker"] is not None:
                    torso_pos = np.array(tracker_data["tracker"]["pos"], dtype=float)
                    torso_quat = np.array(tracker_data["tracker"]["quat"], dtype=float)

                    # --- Convert to rotation matrix ---
                    rot_flat = np.zeros(9)
                    mujoco.mju_quat2Mat(rot_flat, torso_quat)
                    rot = rot_flat.reshape(3, 3)

                    # --- Decompose orientation into Euler angles ---
                    # Using ZYX convention (yaw → pitch → roll)
                    yaw = np.arctan2(rot[1, 0], rot[0, 0])      # rotation about Z
                    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1]**2 + rot[2, 2]**2))  # rotation about Y
                    roll = 0.0  # ❄ freeze roll

                    # --- Rebuild rotation matrix with yaw + pitch only ---
                    cy, sy = np.cos(yaw), np.sin(yaw)
                    cp, sp = np.cos(pitch), np.sin(pitch)
                    cr, sr = np.cos(roll), np.sin(roll)

                    rot_yaw_pitch = np.array([
                        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                        [-sp,     cp * sr,                cp * cr]
                    ])

                    # --- Convert back to quaternion ---
                    fixed_quat = np.zeros(4)
                    mujoco.mju_mat2Quat(fixed_quat, rot_yaw_pitch.flatten())

                    # --- Place torso above pelvis ---
                    pelvis_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                    pelvis_pos = d.xpos[pelvis_body_id].copy()
                    torso_world_pos = pelvis_pos + np.array([0.0, 0.0, 0.15])

                    # --- Apply to the IK target ---
                    torso_pose = mink.SE3.from_rotation_and_translation(
                        rotation=mink.SO3(fixed_quat),
                        translation=torso_world_pos
                    )
                    torso_task.set_target(torso_pose)



                # --- 👣 Feet remain static (floor contact) ---
                for i, foot_task in enumerate(feet_tasks):
                    foot_task.set_target(mink.SE3.from_mocap_id(d, feet_mid[i]))

                # --- 🧍 Keep posture & COM consistent ---
                posture_task.set_target_from_configuration(configuration)
                com_task.set_target(d.subtree_com[1].copy())


                # ✅ Update configuration BEFORE IK
                configuration.update(d.qpos)

                # --- Solve IK for upper body ---
                try:
                    vel = mink.solve_ik(configuration, tasks, m.opt.timestep, solver, 5e-2, limits=limits)
                except mink.exceptions.NoSolutionFound:
                    vel = np.zeros_like(configuration.velocity)
                    print("[WARN] IK solver failed this step")

                # --- Integrate IK velocity ---
                q_desired_full = d.qpos.copy()
                q_desired_full[7:] += vel[6:] * m.opt.timestep * integration_gain
                configuration.update(q_desired_full)

                # --- PD torque control for upper body ---
                q_des_upper = q_desired_full[7+num_leg_dof:]
                q_act_upper = d.qpos[7+num_leg_dof:]
                dq_upper = d.qvel[6+num_leg_dof:]
                tau_upper = Kp_up * (q_des_upper - q_act_upper) - Kd_up * dq_upper
                d.ctrl[num_leg_dof:] = tau_upper

                # --- Waist stabilization ---
                for waist_joint, (Kp_w, Kd_w) in {
                    "waist_roll_joint": (120.0, 3.0),
                    "waist_yaw_joint": (150.0, 4.0)
                }.items():
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, waist_joint)
                    act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, waist_joint)
                    if jid != -1 and act_id != -1 and act_id < m.nu:
                        qadr = m.jnt_qposadr[jid]
                        dadr = m.jnt_dofadr[jid]
                        q_err = -d.qpos[qadr]
                        dq = d.qvel[dadr]
                        d.ctrl[act_id] += Kp_w * q_err - Kd_w * dq

                # --- Step simulation ---
                mujoco.mj_step(m, d)

                # --- RL policy update ---
                counter += 1

                if counter % config['control_decimation'] == 0:
                    # --- Time-based command update ---
                    single_obs, _ = compute_observation(d, config, action, cmd, height_cmd, m, OLD_POLICY_JOINTS)
                    obs_history.append(single_obs)
                    for i, hist_obs in enumerate(obs_history):
                        start_idx = i * single_obs_dim
                        end_idx = start_idx + single_obs_dim
                        obs[start_idx:end_idx] = hist_obs
                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    target_dof_pos = action * config['action_scale'] + config['default_angles'][:config['num_actions']]

                # --- Keep time consistent ---
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            viewer.sync()






          

if __name__ == "__main__":
    main()