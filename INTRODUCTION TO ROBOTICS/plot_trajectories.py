#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from bagpy import bagreader
from scipy.spatial.transform import Rotation as R

def list_files_with_full_path(folder_path):
    try:
        # Get the list of files in the folder with full paths
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except OSError as e:
        print(f"Error reading files in {folder_path}: {e}")
        return None
    
def get_csv(bag_name):
    
    data_dict = {}
    
    # Read bag
    b = bagreader(bag_name)
    
    for topic in b.topics: 
        data = b.message_by_topic(topic)
        data_dict['{}'.format(topic)] = pd.read_csv(data)
        
    return data_dict

def get_name(search, list_of_names): 
    
    for i, name in enumerate(list_of_names): 
        if search in name: 
            return name

# Transform stuff
def create_T(p, q): 
    
    r = R.from_quat([q[0], q[1], q[2], q[3]])
    r = r.as_matrix()
                     
    T = np.matrix([[r[0, 0], r[0, 1], r[0, 2], p[0]], 
                   [r[1, 0], r[1, 1], r[1, 2], p[1]],
                   [r[2, 0], r[2, 1], r[2, 2], p[2]], 
                   [0, 0, 0, 1]])
    return T
                 
def get_p_from_T(T_0P):
    
    x = [T_0P_[0, 3] for T_0P_ in T_0P]
    y = [T_0P_[1, 3] for T_0P_ in T_0P]
    z = [T_0P_[2, 3] for T_0P_ in T_0P]
    
    return(x, y, z)

def transform_to_marker_tip(x, y, z, qx, qy, qz, qw, T_TP): 
    
    T_0T = [create_T((x_, y_, z_), (qx_, qy_, qz_, qw_)) for x_, y_, z_, qx_, qy_, qz_, qw_ in zip(x, y, z, qx, qy, qz, qw)]
    T_0P = [np.dot(T_0T_, T_TP) for T_0T_ in T_0T]
    
    x, y, z = get_p_from_T(T_0P)
    
    return x, y, z 

def extract_kalipen(csv, trace):
    if trace: 
        topic_name = '/kdno/MARKER_COLOR'
    else: 
        topic_name = '/kdno/MARKER_BW'
        
    t = csv[topic_name]['Time']
    x = csv[topic_name]['pose.position.x']
    y = csv[topic_name]['pose.position.y']
    z = csv[topic_name]['pose.position.z']
    
    return (x, y, z)

def extract_robot(csv, T_TP):
    # Same topic name for both recordings
    x = csv['/franka_state_controller/O_T_EE']['pose.position.x'] 
    y = csv['/franka_state_controller/O_T_EE']['pose.position.y']
    z = csv['/franka_state_controller/O_T_EE']['pose.position.z'] 
    qx = csv['/franka_state_controller/O_T_EE']['pose.orientation.x']
    qy = csv['/franka_state_controller/O_T_EE']['pose.orientation.y']
    qz = csv['/franka_state_controller/O_T_EE']['pose.orientation.z']
    qw = csv['/franka_state_controller/O_T_EE']['pose.orientation.w']
    
    x_, y_, z_ = transform_to_marker_tip(x, y, z, qx, qy, qz, qw, T_TP)
    
    return (x_, y_, z_)


def plot_results(p_mc, p_mnc, p_rc, p_rnc, save_=False):
    # Create a subplot with a 1x2 grid
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    
    # Get data from position vectors
    x_mc, y_mc, z_mc = np.array(p_mc[0]), np.array(p_mc[1]), np.array(p_mc[2])
    x_mnc, y_mnc, z_mnc = np.array(p_mnc[0]), np.array(p_mnc[1]), np.array(p_mnc[2])
    x_rc, y_rc, z_rc = np.array(p_rc[0]), np.array(p_rc[1]), np.array(p_rc[2])
    x_rnc, y_rnc, z_rnc = np.array(p_rnc[0]), np.array(p_rnc[1]), np.array(p_rnc[2])

    # Plot the first graph on the left
    axes[0].plot(x_mc, z_mc, color='red', label='Kalipen trace')
    axes[0].plot(x_mnc, z_mnc, color='green', label='Kalipen no trace')
    axes[0].set_title('Kalipen results')
    axes[0].set_xlabel('X [m]')
    axes[0].set_ylabel('Z [m]')
    axes[0].legend()
    axes[0].grid()

    # Plot the second graph on the right
    axes[1].plot(y_rc, z_rc, color='yellow', label='Robot trace')
    axes[1].plot(y_rnc, z_rnc, color='orange', label='Robot no trace')
    axes[1].set_title('Robot marker results')
    axes[1].set_xlabel('X [m]')
    axes[1].set_ylabel('Z [m]')
    axes[1].legend()
    axes[1].grid()

    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the plot
    plt.show()
    
    if save_:
        plt.savefig('./myresults.png')

def plot_blackboard(p_mc, p_nc, p_rc, p_rnc, save_=False):
    
    # Get data from position vectors
    x_mc, y_mc, z_mc = np.array(p_mc[0]), np.array(p_mc[1]), np.array(p_mc[2])
    x_mnc, y_mnc, z_mnc = np.array(p_mnc[0]), np.array(p_mnc[1]), np.array(p_mnc[2])
    x_rc, y_rc, z_rc = np.array(p_rc[0]), np.array(p_rc[1]), np.array(p_rc[2])
    x_rnc, y_rnc, z_rnc = np.array(p_rnc[0]), np.array(p_rnc[1]), np.array(p_rnc[2])
    
    plt.figure(figsize=(15, 10))
    plt.plot(x_mc, z_mc, label='Kalipen TRACE')
    plt.plot(x_mnc, z_mnc, label='Kalipen NO TRACE')
    plt.plot(y_rc, z_rc, label="Robot TRACE")
    plt.plot(y_rnc, z_rnc, label="Robot NO TRACE")
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    plt.title('Blackboard of the experiment')
    plt.legend()
    plt.grid()
    plt.show()

    if save_: 
        plt.savefig('./myblackboard')


if __name__ == "__main__":

    # TF between tool and pen
    T_TP = np.matrix([[1, 0, 0, 0.0015946], 
                    [0, 1, 0, -0.00069987], 
                    [0, 0, 1, 0.14166], 
                    [0, 0, 0, 1]])
    
    bags = list_files_with_full_path(sys.argv[1])

    # Plots marker movement in space
    csv_k_trace = get_csv(get_name('MARKER_COLOR', bags))
    csv_k_ntrace = get_csv(get_name('MARKER_BW', bags))
    csv_r_trace = get_csv(get_name('color_robot', bags))
    csv_r_ntrace = get_csv(get_name('bw_robot', bags))

    p_mc = extract_kalipen(csv_k_trace, trace=True)
    p_mnc = extract_kalipen(csv_k_ntrace, trace=False)
    p_rc = extract_robot(csv_r_trace, T_TP)
    p_rnc =  extract_robot(csv_r_ntrace, T_TP)

    plot_results(p_mc, p_mnc, p_rc, p_rnc, save_ = True)
    plot_blackboard(p_mc, p_mnc, p_rc, p_rnc, save_ = True)

