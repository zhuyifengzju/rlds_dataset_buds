import h5py
import os

import cv2

import numpy as np

new_demo = h5py.File("buds_dataset/buds_real_kitchen_demo.hdf5", "w")
new_demo.create_group("data")
new_demo.attrs["num_demos"] = 50

total = 0

with h5py.File('buds_datasets/Kitchen3/demo.hdf5', 'r') as f:
    for ep_idx in range(50):
        # convert camera_0_color
        ep_grp = new_demo['data'].create_group(f"demo_{ep_idx}")
        images = []
        for image_name in f[f"data/ep_{ep_idx}/camera_0_color"]:
            image_path = image_name.decode("utf-8").replace("datasets", "buds_datasets")
            if not os.path.exists(image_path):
                print(image_path)
            image = cv2.imread(image_path)
            images.append(np.array(image))

        num_samples = len(images)
        total += num_samples
        
        ep_grp.attrs["num_samples"] = num_samples
        ep_grp.create_dataset("agentview_rgb", data=np.stack(images))

        images = []
        # convert camera_1_color
        for image_name in f[f"data/ep_{ep_idx}/camera_1_color"]:
            image_path = image_name.decode("utf-8").replace("datasets", "buds_datasets")
            if not os.path.exists(image_path):
                print(image_path)
            image = cv2.imread(image_path)
            images.append(np.array(image))
        ep_grp.create_dataset("eye_in_hand_rgb", data=np.stack(images))

        # proprio ee
        ee_states = f[f"data/ep_{ep_idx}/proprio_ee"][()]
        ep_grp.create_dataset("ee_states", data=ee_states.astype(np.float32))

        # proprio gripper
        gripper_states = f[f"data/ep_{ep_idx}/proprio_gripper_state"][()]
        ep_grp.create_dataset("gripper_states", data=gripper_states.astype(np.float32))

        # proprio joints
        joint_states = f[f"data/ep_{ep_idx}/proprio_joints"][()]
        ep_grp.create_dataset("joint_states", data=joint_states.astype(np.float32))

        # actions
        actions = f[f"data/ep_{ep_idx}/actions"][()]
        ep_grp.create_dataset("actions", data=actions)


new_demo.attrs["total"] = total

new_demo.close()