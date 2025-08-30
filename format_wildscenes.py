"""
Format WildScenes dataset for VPR evaluation
raw_data/V-03 -> test/database
raw_data/V-01, raw_data/V-02 -> test/queries
"""

import os
import shutil
from glob import glob
from tqdm import tqdm
from os.path import join
import pandas as pd
import numpy as np
from PIL import Image
import yaml
import utm
import math
import map_builder

# Dataset structure
default_scenes = {
    'test': ["V-01", "V-02", "V-03"]
}

datasets_folder = join(os.curdir, "datasets")
dataset_name = "wildscenes"
dataset_folder = join(datasets_folder, dataset_name)
raw_data_folder = join(datasets_folder, dataset_name, "raw_data")

def load_poses_from_csv(poses_file):
    """Load poses from WildScenes CSV format"""
    try:
        df = pd.read_csv(poses_file)
        numeric_cols = ['timestamp', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_cols)        
        if len(df) == 0:
            print(f"Warning: No valid poses found in {poses_file}")
            return None
        return df
    except Exception as e:
        print(f"Error loading poses from {poses_file}: {e}")
        return None

def quaternion_to_euler(qw, qx, qy, qz):
    """Convert quaternion to euler angles (yaw, pitch, roll)"""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # Convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    return yaw_deg, pitch_deg, roll_deg

def copy_images_with_poses(src_folder, dst_folder, poses_df):
    """Copy images using pose coordinates for naming"""
    os.makedirs(dst_folder, exist_ok=True)
    
    image_dir = os.path.join(src_folder, 'image')
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    image_files.sort()

    for idx, img_file in enumerate(tqdm(image_files, desc=f"Copy to {dst_folder}")):
        img_path = os.path.join(image_dir, img_file)
        timestamp = img_file.replace('.png', '')

        # NOTE(gogojjh): Set fake longitude and latitude
        base_longitude = 153.18
        base_latitude = -27.62        
        pose = poses_df.iloc[idx]
        # Convert UTM offsets to longitude and latitude based on the base point
        utm_east = pose['x']
        utm_north = pose['y']
        height = pose['z']
        meters_per_deg_lat = 111320
        meters_per_deg_lon = 111320 * math.cos(math.radians(base_latitude))
        latitude = base_latitude + (utm_north / meters_per_deg_lat)
        longitude = base_longitude + (utm_east / meters_per_deg_lon)
        yaw, pitch, roll = quaternion_to_euler(pose['qw'], pose['qx'], pose['qy'], pose['qz'])
        dst_image_name = f"@{utm_east:.6f}@{utm_north:.6f}@17@T@{latitude:.6f}@{longitude:.6f}@@@{yaw:.2f}@{pitch:.2f}@{roll:.2f}@{height:.3f}@{timestamp}@@.jpg"
        
        dst_path = os.path.join(dst_folder, dst_image_name)
        shutil.copy2(img_path, dst_path)

def main():
    """Main conversion function"""
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(raw_data_folder, exist_ok=True)
    
    # Create output directories
    test_dir = join(dataset_folder, 'images', 'test')
    os.makedirs(join(test_dir, 'database'), exist_ok=True)
    os.makedirs(join(test_dir, 'queries'), exist_ok=True)
    
    # Process all scenes
    scenes_config = {
        'V-03': join(test_dir, 'database'),
        'V-01': join(test_dir, 'queries'),
        'V-02': join(test_dir, 'queries')
    }
    
    for scene, dst_folder in scenes_config.items():
        print(f"Processing {scene} -> {dst_folder}...")
        scene_dir = join(raw_data_folder, scene)
        
        if os.path.exists(scene_dir):
            poses_file = os.path.join(scene_dir, 'poses2d.csv')
            poses_df = load_poses_from_csv(poses_file)
            if poses_df is not None:
                copy_images_with_poses(scene_dir, dst_folder, poses_df)
            else:
                print(f"Failed to load poses for {scene}")
        else:
            print(f"{scene} directory not found")

    map_builder.build_map_from_dataset(dataset_folder)
    print("Conversion completed!")

if __name__ == "__main__":
    main()
