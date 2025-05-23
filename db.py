from pathlib import Path
import os
import sys
sys.path.append('/home/luffy/continue/repos/Hierarchical-Localization')
from hloc import (
    # extract_features_new,
    # match_features,
    reconstruction,
    # visualization,
    # pairs_from_retrieval,
    # conv_new
)
import struct
import sqlite3
WIDTH = 1600
HEIGHT = 800

def change_camera_model_and_parameters(db_path, new_model_id, new_params):

    new_params_binary = struct.pack(f'{len(new_params)}d', *new_params)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT camera_id FROM cameras;")
    cameras = cursor.fetchall()
    for camera_id in cameras:
        cursor.execute("""
            UPDATE cameras
            SET model = ?, params = ?
            WHERE camera_id = ?;
        """, (new_model_id, new_params_binary, camera_id[0]))

    conn.commit()
    conn.close()
    print(f"Updated all cameras to model '{new_camera_model}' with new parameters.")

pairs = Path("/home/luffy/continue/repos/COLMAP-ELoFTR/image_pairs.txt")  # Replace with the actual path
image_dir = Path("//home/luffy/data/VID_20240622_155518_00_007_processed_1600_800")  # Replace with the actual path
export_dir = Path(".")  # Replace with the actual path
matches = Path("matches-loftr_image_pairs.h5")
features = Path("feats_matches-loftr.h5")

model = reconstruction.main(export_dir, image_dir, pairs, features, matches)
db_path = export_dir / "database.db"
new_camera_model = 11  
new_params = [WIDTH*1.2, WIDTH/2,HEIGHT/2]  
change_camera_model_and_parameters(db_path, new_camera_model, new_params)