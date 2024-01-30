import numpy as np

from tf.transformations import quaternion_from_matrix
from geometry_msgs.msg import  Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import matplotlib.pyplot as plt
import yaml
import rospy


# IO utils
def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def has_duplicates(lst):
    seen = set()
    for element in lst:
        if element in seen:
            return True
        seen.add(element)
    return False

def get_poses(poses_data): 
    
    poses = []
    # Accessing individual poses
    for pose in poses_data.get('poses', []):
        for pose_name, pose_values in pose.items():
            pose_msg = Pose()
            pose_msg.position.x = pose_values.get('x', 0.0)
            pose_msg.position.y = pose_values.get('y', 0.0)
            pose_msg.position.z = pose_values.get('z', 0.0)
            pose_msg.orientation.x = pose_values.get('qx', 0.0)
            pose_msg.orientation.y = pose_values.get('qy', 0.0)
            pose_msg.orientation.z = pose_values.get('qz', 0.0)
            pose_msg.orientation.w = pose_values.get('qw', 0.0)
            poses.append(normalize_q(pose_msg))
    return poses

def poseToPoseStamped(pose): 

    pStamped = PoseStamped()
    pStamped.pose.position = pose.position
    pStamped.pose.orientation = pose.orientation
    return pStamped

def sigmoid(x): 

    return 1/1+np.exp(-x)

def scale_sigmoid(a, x): 
    
    return a*sigmoid(x)

# Conversion utils
def poseFromMatrix(matrix):
    goal_pose = Pose()
    quat = quaternion_from_matrix(matrix)

    goal_pose.position.x = matrix[0,3]
    goal_pose.position.y = matrix[1,3]
    goal_pose.position.z = matrix[2,3]
    goal_pose.orientation.x = quat[0]
    goal_pose.orientation.y = quat[1]
    goal_pose.orientation.z = quat[2]
    goal_pose.orientation.w = quat[3]

    return goal_pose

def poseToArray(pose):

    x = pose.position.x
    y = pose.position.y
    z = pose.position.z
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w

    return np.asarray([x, y, z, qx, qy, qz, qw])

def arrayToPose(poseArray):

    pose = Pose()
    pose.position.x = poseArray[0]
    pose.position.y = poseArray[1]
    pose.position.z = poseArray[2]
    pose.orientation.x = poseArray[3]
    pose.orientation.y = poseArray[4]
    pose.orientation.z = poseArray[5]
    pose.orientation.w = poseArray[6]

    return pose

def createGoalPose(x, y, z, qx, qy, qz, qw):

    wanted_pose = Pose()
    wanted_pose.position.x = x
    wanted_pose.position.y = y
    wanted_pose.position.z = z
    wanted_pose.orientation.x = qx
    wanted_pose.orientation.y = qy
    wanted_pose.orientation.z = qz
    wanted_pose.orientation.w = qw
    return wanted_pose

def normalize_q(pose):

    qx = pose.orientation.x
    qy = pose.orientation.y 
    qz = pose.orientation.z 
    qw = pose.orientation.w  
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx = qx/norm; qy = qy/norm; qz = qz/norm; qw = qw/norm
    pose.orientation.x = qx 
    pose.orientation.y = qy 
    pose.orientation.z = qz 
    pose.orientation.w = qw 
    return pose

# FK utils
def TfromDH(theta, d, alpha, a):
    
    T = np.eye(4)
    T[0,0] = np.cos(theta)
    T[0,1] = -np.sin(theta)*np.cos(alpha)
    T[0,2] = np.sin(theta)*np.sin(alpha)
    T[0,3] = a*np.cos(theta)
    T[1,0] = np.sin(theta)
    T[1,1] = np.cos(theta)*np.cos(alpha)
    T[1,2] = -np.cos(theta)*np.sin(alpha)
    T[1,3] = a*np.sin(theta)
    T[2,0] = 0
    T[2,1] = np.sin(alpha)
    T[2,2] = np.cos(alpha)
    T[2,3] = d
    return T

# ORLAB1 forwardKinematics
def forwardKinematics(q, plot=False):
    """ Forward kinematics 

    Args:
        q (np.array): measured joint values in the array [7x1]
        plot (bool): False

    Returns:
        T0e (np.matrix): HTM of the tool in the 0 frame [world frame] [4x4]
    """

    q1 = round(q[0], 4); q2 = round(q[1], 4); q3 = round(q[2], 4); q4 = round(q[3], 4); q5 = round(q[4], 4); q6 = round(q[5],4); q7 = round(q[6],4)

    M_PI = np.pi
    # theta, d, alpha, a
    T01 = TfromDH(q1, 0.333, -M_PI/2, 0.)
    T12 = TfromDH(q2, 0, M_PI/2, 0)
    T23 = TfromDH(q3, 0.316, M_PI/2, 0.0825)
    T34 = TfromDH(q4, 0, -M_PI/2, -0.0825)
    T45 = TfromDH(q5, 0.384, M_PI/2, 0.)
    T56 = TfromDH(q6, 0, M_PI/2, 0.088)
    T67 = TfromDH(q7, 0.107, 0, 0.)
    T7e = np.eye(4); T7e[2, 3] = 0.1; 

    T02 = np.matmul(T01, T12)
    T03 = np.matmul(T02, T23)
    T04 = np.matmul(T03, T34)
    T05 = np.matmul(T04, T45)
    T06 = np.matmul(T05, T56)
    T07 = np.matmul(T06, T67)
    T0e = np.matmul(T07, T7e)


    return poseFromMatrix(T0e)

def createTrajectory(joint_names, q, dq, ddq, t):

    trajectoryMsg = JointTrajectory()
    trajectoryMsg.joint_names = joint_names

    dq = list(dq.T)
    i = 0
    for k, (q, dq, ddq) in enumerate(zip(q, dq, ddq)):
        try:
            i += t[k]
            t_ = rospy.Time.from_sec(i)
        except Exception as e:
            t_ = rospy.Time.from_sec(np.ceil(i))
        trajectoryPoint = JointTrajectoryPoint()
        trajectoryPoint.positions = q
        trajectoryPoint.velocities = dq
        trajectoryPoint.accelerations = ddq
        trajectoryPoint.time_from_start.secs = t_.secs
        trajectoryPoint.time_from_start.nsecs = t_.nsecs
        trajectoryMsg.points.append(trajectoryPoint)

    return trajectoryMsg



def createPredefinedTrajectory(joint_names, q, dq, ddq, t):

    trajectoryMsg = JointTrajectory()
    trajectoryMsg.joint_names = joint_names

    for k, (q, dq, ddq) in enumerate(zip(q, dq, ddq)):
        trajectoryPoint = JointTrajectoryPoint()
        trajectoryPoint.positions = q
        trajectoryPoint.velocities = dq
        trajectoryPoint.accelerations = ddq
        trajectoryPoint.time_from_start.secs = int(np.floor(t[k]))
        trajectoryPoint.time_from_start.nsecs = int((t[k] - np.floor(t[k]))*10e9)
        trajectoryMsg.points.append(trajectoryPoint)

    return trajectoryMsg



def createTaylorTrajectory(joint_names, q_list, dt): 

    trajectoryMsg = JointTrajectory()
    trajectoryMsg.joint_names = joint_names

    for k, q in enumerate(q_list): 
        trajectoryPoint = JointTrajectoryPoint()
        trajectoryPoint.positions = q
        trajectoryPoint.time_from_start.secs = k*dt
        trajectoryMsg.points.append(trajectoryPoint)
        duration = k*dt
    
    return duration, trajectoryMsg

def createSimpleTrajectory(joint_names, q_curr, q_goal, t_move=None): 

    traj = JointTrajectory()
    traj.joint_names = joint_names
    t_q_curr = JointTrajectoryPoint()
    t_q_goal = JointTrajectoryPoint()
    t_q_curr.positions = q_curr
    t_q_goal.positions = q_goal
    if t_move: 
        t_q_goal.time_from_start.secs = t_move
    else: 
        t_q_goal.time_from_start.secs = 5
    traj.points.append(t_q_curr)
    traj.points.append(t_q_goal)
    
    return traj

def calc_cartesian_midpoint(start_pose, end_pose):

    # Calculate position average
    x = (start_pose.position.x + end_pose.position.x)/2
    y = (start_pose.position.y + end_pose.position.y)/2
    z = (start_pose.position.z + end_pose.position.z)/2
    # Keep orientation same as in starting points
    qx = (start_pose.orientation.x)
    qy = (start_pose.orientation.y)
    qz = (start_pose.orientation.z)
    qw = (start_pose.orientation.w)

    return np.asarray([x, y, z, qx, qy, qz, qw])

def calc_joint_midpoint(start_joint, end_joint):
    return (start_joint + end_joint)/2

def norm(q_cmd, q_curr):
    norm = np.sqrt(np.sum((q_cmd[:3]- q_curr[:3])**2))
    return norm

# HoCook utils
def get_time_parametrization(q):

    t = []
    for k, qk in enumerate(q):
        try:
            tk_1 = np.sqrt(np.sum((q[k+1] - q[k])**2))
            t.append(tk_1)
        except Exception as e:
            pass

    rospy.loginfo("Finished time parametrization, segment num, m-1: {}".format(len(t)))

    return t

def createMpmatrix(t):
    # Matrix dimensions are (m - 2) x (m - 4)
    # t dimensions are m-1

    m_1 = len(t)
    Mp = np.zeros((m_1 - 1, m_1 - 3))
    for i, t_ in enumerate(t):
        try:
            Mp[i, i] =  t[i+2]
            Mp[i + 1, i] = 2*(t[i+1] + t[i+2])
            Mp[i + 2, i] = t[i+1]

        except Exception as e:
            pass

    rospy.loginfo("Finished adding elements to the Mp matrix, dimensions (m-2) x (m-4) : {}".format(Mp.shape))

    return Mp

def createMmatrix(Mp, t):
    # Matrix dimensions are (m - 2) x (m - 2)

    m_1 = len(t)
    M1col = np.zeros((m_1 - 1, 1))
    Mlcol = np.zeros((m_1 - 1, 1))

    M1col[0, 0] = 3 /t[0] + 2/t[1]
    M1col[1, 0] = 1 /t[1]
    Mlcol[-2, 0] = 1 /t[-2]
    Mlcol[-1, 0] = 2/t[-2] + 3/t[-1]

    M = np.hstack((M1col, Mp))
    M = np.hstack((M, Mlcol))

    return M

def createApmatrix(n, q, t):
    # Matrix dimensions are n x (m - 4)
    # t dimensions are m-1

    m_1 = len(t)
    Ap = np.zeros((n, m_1 - 3))
    for i , t_ in enumerate(t):
        try:
            c = 3/(t[i+1] * t[i+2])
            bracket = t[i+1]**2*(q[i+3] - q[i+2]) + t[i+2]**2*(q[i+2] - q[i+1])
            Ap[:, i] = c * bracket
        except Exception as e:
            pass

    rospy.loginfo("Finished adding elements to the Ap matrix, dimensions n x (m-4) : {}".format(Ap.shape))

    return Ap

def createAmatrix(n, q, t, Ap):
    # Matrix dimensions are n x (m - 2)
    # t dimensions are m-1

    A1col = 6/t[0]**2 * (q[1] - q[0]) + 3/t[1]**2 * (q[2] - q[1])
    Alcol = 3/t[-2]**2 * (q[-2] - q[-3]) + 6/t[-1]**2*(q[-1] - q[-2])

    A = np.hstack((A1col.reshape(n, 1), Ap))
    A = np.hstack((A, Alcol.reshape(n, 1)))

    return A

# Get B matrices
def getBfirstSeg(q, dq, t):

    T = np.zeros((4, 5))
    T[0, 0] = 1;            T[0, 3] = -4/t[0]**3;   T[0, 4] = 3/t[0]**4
    T[1, 3] = 4/t[0]**3;    T[1, 4] = -3/t[0]**4;
    T[3, 3] = -1/t[0]**2;   T[3, 4] = 1/t[0]**3;

    Q = np.array((q[0], q[1], dq[:, 0], dq[:, 1])).T # 6 x 4

    # (7 x 4) x (4 x 5)
    BfirstSeg = np.matmul(Q, T) # 7 x 5 dimensions

    return BfirstSeg

def getBanySeg(q, dq, t, k):

    T = np.zeros((4, 4))
    T[0, 0] = 1;            T[0, 2] = -3/t[k]**2;   T[0, 3] = 2/t[k]**3;
    T[1, 2] = 3/t[k]**2;    T[1, 3] = -2/t[k]**3;
    T[2, 1] = 1;            T[2, 2] = -2/t[k];      T[2, 3] = 1/t[k]**2;
    T[3, 2] = -1/t[k];      T[3, 3] = 1/t[k]**2;

    # Fix indexing (First seg, k=1, q[k-1] = q[0], but has to be q[1], q[2])
    Q = np.array((q[k], q[k+1], dq[:, k], dq[:, k+1])).T

    BkSeg = np.matmul(Q, T)

    return BkSeg

def getBLastSeg(q, dq, t):

    T = np.zeros((4, 5))
    T[0, 0] = 1;            T[0, 2] = -6/t[-1]**2;  T[0, 3] = 8/t[-1]**3; T[0, 4] = -3/t[-1]**4;
    T[1, 2] = 6/t[-1]**2;   T[1, 3] = -8/t[-1]**3;  T[1, 4] = 3/t[-1]**4;
    T[2, 1] = 1;            T[2, 2] = -3/t[-1];     T[2, 3] = 3/t[-1]**2; T[2, 4] = -1/t[-1]**3;
    Q = np.array((q[-2], q[-1], dq[: , -2], dq[:, -1])).T

    BlastSeg = np.matmul(Q, T)

    return BlastSeg

def getMaxSpeedFirstSeg(B, t):

    dq_max = B[:, 1] + 2*B[:, 2]*t[0] + 3*B[:, 3]*t[0]**2 + 4*B[:, 4]*t[0]**3 

    return dq_max

def getMaxSpeedLastSeg(B, t):

    dq_max = B[:, 1] + 2*B[:, 2]*t[-1] + 3*B[:, 3]*t[-1]**2 + 4*B[:, 4]*t[-1]**3 

    return dq_max

def getMaxSpeedAnySeg(B, t, k):

    dq_max = B[:, 1] + 2*B[:, 2]*t[k] + 3*B[:, 3]*t[k]**2

    return dq_max

def getMaxAccFirstSeg(B, t): 
    
    ddq_max = 2* B[:, 2] + 6*B[:, 3]*t[0] + 12*B[:, 4]*t[0]**2
    
    return ddq_max

def getMaxAccLastSeg(B, t): 
    
    ddq_max = 2*B[:, 2] + 6*B[:, 3]*t[-1] + 12*B[:, 4]*t[-1]**2
    
    return ddq_max

def getMaxAccAnySeg(B, t, k): 
    
    ddq_max = 2*B[:, 2] + 6*B[:, 3]*t[k] 
    
    return ddq_max

def get_dq_max(q, dq, t):

    Bk = []; dqmax = []; 
    for k, t_ in enumerate(t):
        if k == 0:
            Bfirst = getBfirstSeg(q, dq, t)
            dqmax_ = getMaxSpeedFirstSeg(Bfirst, t)
            dqmax.append(dqmax_)
        # Calculate stuff for all segments without first > 0 and last len(t) - 1
        if k > 0 and k < len(t) - 1:
            Bk_ = getBanySeg(q, dq, t, k)
            Bk.append(Bk_)
            dqmax_ = getMaxSpeedAnySeg(Bk_, t,  k)
            dqmax.append(dqmax_)
        if k == len(t) - 1 :
            Blast = getBLastSeg(q, dq, t)
            dqmax_ = getMaxSpeedLastSeg(Blast, t)
            print("t is : {}".format(t))
            print("dqmax_ last seg is: {}".format(dqmax_))
            dqmax.append(dqmax_)

    print("dqmax: {}".format(dqmax))
    dqmax = np.asarray(dqmax).T; # Returns indices (np.argmax(dq_max, axis=1))
    dqmax = np.amax(dqmax, axis=1);  # Returns values axis = 0 -> column-wise, axis=1, row-wise
    
    return dqmax

def get_ddq_max(q, dq, t): 

    Bk = []; ddqmax = []
    for k, t_ in enumerate(t):
        if k == 0:
            Bfirst = getBfirstSeg(q, dq, t)
            ddqmax_ = getMaxAccFirstSeg(Bfirst, t)
            ddqmax.append(ddqmax_)
        # Calculate stuff for all segments without first > 0 and last len(t) - 1
        if k > 0 and k < len(t) - 1:
            Bk_ = getBanySeg(q, dq, t, k)
            Bk.append(Bk_)
            ddqmax_ = getMaxAccAnySeg(Bk_, t, k)
            ddqmax.append(ddqmax_)
        if k == len(t) - 1 :
            Blast = getBLastSeg(q, dq, t)
            ddqmax_ = getMaxAccLastSeg(Blast, t)
            ddqmax.append(ddqmax_)

    return ddqmax

def read_file(file_path):
    a_ = []
    with open(file_path, 'r') as file:
        for row in file:
            a = [float(i) for i in row.split(",")]
            a_.append(a)
    return a_

# Plt utils
def draw(points_gazebo, points_fk, cart_points, eps):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    arr_points = np.stack(points_gazebo, axis=0)
    ax.plot(arr_points[:,0],arr_points[:,1],arr_points[:,2], label='Gazebo path')
    fk_points = np.stack(points_fk, axis=0)
    ax.plot(fk_points[:,0],fk_points[:,1],fk_points[:,2], 'r', label='Forward kinematics')
    x_ = [p[0] for p in cart_points]
    y_ = [p[1] for p in cart_points]
    z_ = [p[2] for p in cart_points]
    ax.scatter(x_, y_, z_ , label="Taylor calculated points")
    ax.set_xlabel('x [m]'); ax.set_xlim([-0.3, 0.3])
    ax.set_ylabel('y [m]'); ax.set_ylim([-0.25, 0.25])
    ax.set_zlabel('z [m]'); ax.set_zlim([0.2, 1.2])
    ax.legend(loc='lower right')
    plt.title('End effector path eps={}'.format(eps))
    plt.show()
