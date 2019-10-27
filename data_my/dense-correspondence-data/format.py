import os
import yaml
import numpy as np
from yaml import CLoader
from transforms3d import quaternions

def get_curr_dir():
    return os.path.dirname(__file__)


def getDictFromYamlFilename(filename):
    """
    Read data from a YAML files
    """
    return yaml.load(file(filename), Loader=CLoader)


def saveToYaml(data, filename):
    with open(filename, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def get_current_YYYY_MM_DD_hh_mm_ss():
    """
    Returns a string identifying the current:
    - year, month, day, hour, minute, second

    Using this format:

    YYYY-MM-DD-hh-mm-ss

    For example:

    2018-04-07-19-02-50

    Note: this function will always return strings of the same length.

    :return: current time formatted as a string
    :rtype: string

    """

    now = datetime.datetime.now()
    string = "%0.4d-%0.2d-%0.2d-%0.2d-%0.2d-%0.2d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    return string


def dictFromPosQuat(pos, quat):
    d = dict()
    d['translation'] = dict()
    d['translation']['x'] = pos[0]
    d['translation']['y'] = pos[1]
    d['translation']['z'] = pos[2]

    d['quaternion'] = dict()
    d['quaternion']['w'] = quat[0]
    d['quaternion']['x'] = quat[1]
    d['quaternion']['y'] = quat[2]
    d['quaternion']['z'] = quat[3]

    return d


def getQuaternionFromDict(d):
    quat = None
    quatNames = ['orientation', 'rotation', 'quaternion']
    for name in quatNames:
        if name in d:
            quat = d[name]

    if quat is None:
        raise ValueError("Error when trying to extract quaternion from dict, your dict doesn't contain a key in ['orientation', 'rotation', 'quaternion']")

    return quat


def format_data_for_tsdf(image_folder):
    """
    Processes the data into the format needed for tsdf-fusion algorithm
    """

    def cal_camera_matrix(k_matrix):
        matrix = np.zeros((3, 3))
        matrix[0, :3] = k_matrix[0:3]
        matrix[1, :3] = k_matrix[3:6]
        matrix[2, :3] = K_matrix[6:9]
        print "[INFO] Camera intrinsis matrix:\n", matrix
        return matrix

    def cal_camera_pose(pose_data):
        trans = pose_data['translation']
        quat = pose_data['quaternion']

        trans_xyz = (trans['x'], trans['y'], trans['z'])
        quat_wxyz = (quat['w'], quat['x'], quat['y'], quat['z'])

        print trans_xyz
        print quat_wxyz

        # quaternions to rotation matrix
        rotation_matrix = quaternions.quat2mat(quat_wxyz)
        print rotation_matrix

        # generate homogenous matrix
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = np.array(trans_xyz).T
        matrix[3][3] = 1.0
        print "[INFO] Camera pose matrix:\n", matrix
        return matrix

    # generate camera matrix file
    camera_info_yaml = os.path.join(image_folder, "camera_info.yaml")
    camera_info = getDictFromYamlFilename(camera_info_yaml)
    K_matrix = camera_info['camera_matrix']['data']
    matrix = cal_camera_matrix(K_matrix)
    camera_info_file_full_path = os.path.join(image_folder, "camera-intrinsics.txt")
    np.savetxt(camera_info_file_full_path, matrix)

    # generate camera pose file
    pose_data_yaml = os.path.join(image_folder, "pose_data.yaml")
    with open(pose_data_yaml, 'r') as stream:
        try:
            pose_data_dict = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print pose_data_dict[0]

    for i in pose_data_dict:
        matrix = cal_camera_pose(pose_data_dict[i]['camera_to_world'])

        depth_image_filename = pose_data_dict[i]['depth_image_filename']
        prefix = depth_image_filename.split("depth")[0]
        pose_file_name = prefix + "pose.txt"
        pose_file_full_path = os.path.join(image_folder, pose_file_name)
        np.savetxt(pose_file_full_path, matrix)

    return len(pose_data_dict)


images_dir = os.path.join(os.path.dirname(__file__), 'images')
format_data_for_tsdf(images_dir)
