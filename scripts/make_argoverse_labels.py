import os
import sys
import numpy as np
from PIL import Image
from progressbar import ProgressBar

from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader \
    import ArgoverseTrackingLoader
from argoverse.utils.camera_stats import RING_CAMERA_LIST

sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))

from src.utils.configs import get_default_configuration
from src.data.utils import get_visible_mask, get_occlusion_mask, \
    encode_binary_labels
from src.data.argoverse.utils import get_object_masks, get_map_mask

from scipy.io import loadmat

colors = loadmat(os.getcwd() + '/scripts/data/color150.mat')['colors']

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def colorEncode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb



def process_split(split, map_data, config):

    # Create an Argoverse loader instance
    path = os.path.join(os.path.expandvars(config.dataroot), split)
    print("Loading Argoverse tracking data at " + path)
    loader = ArgoverseTrackingLoader(path)

    for scene in loader:
        process_scene(split, scene, map_data, config)


def process_scene(split, scene, map_data, config):

    print("\n\n==> Processing scene: " + scene.current_log)

    i = 0
    progress = ProgressBar(
        max_value=len(RING_CAMERA_LIST) * scene.num_lidar_frame)
    
    # Iterate over each camera and each frame in the sequence
    for frame in range(scene.num_lidar_frame):
        for camera in RING_CAMERA_LIST:
            progress.update(i)
            process_frame(split, scene, camera, frame, map_data, config)
            i += 1
            

def process_frame(split, scene, camera, frame, map_data, config):

    # Compute object masks
    masks = get_object_masks(scene, camera, frame, config.map_extents,
                             config.map_resolution)
    
    # Compute drivable area mask
    masks[0] = get_map_mask(scene, camera, frame, map_data, config.map_extents,
                            config.map_resolution)
    
    # Ignore regions of the BEV which are outside the image
    calib = scene.get_calibration(camera)
    masks[-1] |= ~get_visible_mask(calib.K, calib.camera_config.img_width,
                                   config.map_extents, config.map_resolution)
    
    # Ignore regions of the BEV which are occluded (based on LiDAR data)
    lidar = scene.get_lidar(frame)
    cam_lidar = calib.project_ego_to_cam(lidar)
    masks[-1] |= get_occlusion_mask(cam_lidar, config.map_extents, 
                                    config.map_resolution)
    
    # Encode masks as an integer bitmask
    labels = encode_binary_labels(masks)

    # Flip labels about z-axis since ego should be at bottom-center
    labels = labels[::-1, :]

    # idx_map = np.zeros(masks.shape[1:], dtype=np.uint8)
    # for idx in range(masks.shape[0]):
    #     idx_map[masks[idx] != False] = idx + 1
    # color_labels = colorEncode(idx_map, colors)
    # Image.fromarray(color_labels.astype(np.uint8)).save(f'/scratch/shantanu/{camera}_bev.png')

    # Create a filename and directory
    timestamp = str(scene.image_timestamp_list_sync[camera][frame])
    output_path = os.path.join(config.label_root, split, 
                               scene.current_log, camera, 
                               f'{camera}_{timestamp}.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save encoded label file to disk
    Image.fromarray(labels.astype(np.int32), mode='I').save(output_path)
    

if __name__ == '__main__':

    config = get_default_configuration()
    config.merge_from_file('configs/datasets/argoverse.yml')

    # Create an Argoverse map instance
    map_data = ArgoverseMap()

    for split in ['train1', 'train2', 'train3', 'train4', 'val']:
        process_split(split, map_data, config)


