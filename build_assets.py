import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
import glob

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================

# List of pose folders in input_images
# Note: These names must match your folder names exactly
POSES_LIST = ["Serve", "DriveForehand", "DriveBackhand", "Smash", "Volley"]


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# INPUT_ROOT = os.path.join(BASE_DIR, "input_images")
# ASSETS_ROOT = os.path.join(BASE_DIR, "assets")

INPUT_ROOT = "input_images"  
ASSETS_ROOT = "assets"      

# Initialize MediaPipe (Static mode, high accuracy)
mp_pose = mp.solutions.pose
pose_static = mp_pose.Pose(
    static_image_mode=True, 
    model_complexity=2, 
    enable_segmentation=True,
    min_detection_confidence=0.5
)

def build_all_assets():
    print("========================================")
    print("   STARTING ASSETS BUILD PROCESS")
    print("========================================")
    
    # 1. Clean up old assets folder
    if os.path.exists(ASSETS_ROOT):
        print(f"-> Deleting old folder: {ASSETS_ROOT}")
        shutil.rmtree(ASSETS_ROOT)
    
    os.makedirs(ASSETS_ROOT)
    print(f"-> Created new folder: {ASSETS_ROOT}")

    total_poses_processed = 0

    # 2. Iterate through each pose in the list
    for pose_name in POSES_LIST:
        input_dir = os.path.join(INPUT_ROOT, pose_name)
        output_dir = os.path.join(ASSETS_ROOT, pose_name)
        
        print(f"\nChecking: {pose_name}...")
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            print(f"ERROR: Folder '{input_dir}' not found.")
            print("   -> Please create this folder and add 4 images.")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # --- FIND FILES (Support multiple extensions & Deduplicate) ---
        valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]
        images = []
        for ext in valid_extensions:
            found = glob.glob(os.path.join(input_dir, ext))
            images.extend(found)
        
        # Deduplicate and sort
        images = sorted(list(set(images)))
        
        # Check image count
        if len(images) != 4:
            print(f"WARNING: Exactly 4 images required for '{pose_name}'. Found {len(images)}.")
            print(f"   -> File list: {images}")
            print("   -> Skipping this pose.")
            continue
            
        print(f"   Found {len(images)} images. Processing...")
        
        # Process each image
        processed_count = 0
        for i, img_path in enumerate(images):
            frame = cv2.imread(img_path)
            if frame is None: 
                print(f"   Cannot read file: {img_path}")
                continue
            
            # --- IMPORTANT: FLIP IMAGE (MIRROR) ---
            # To match the user's webcam mirror view
            frame = cv2.flip(frame, 1) 
            # ------------------------------------
            
            # MediaPipe Process
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_static.process(img_rgb)
            
            if results.pose_landmarks and results.segmentation_mask is not None:
                # A. Create Ghost Image
                mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
                ghost = np.zeros_like(frame)
                ghost[:] = (0, 255, 0) # Green color
                
                # Cut person from background
                ghost_masked = cv2.bitwise_and(ghost, ghost, mask=mask)
                
                # Draw white contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ghost_masked, contours, -1, (255, 255, 255), 2)
                
                # Save Ghost Image
                cv2.imwrite(f"{output_dir}/ghost_{i}.png", ghost_masked)
                
                # B. Calculate Metadata (For adaptive scaling)
                lms = results.pose_landmarks.landmark
                # Torso height (Shoulder -> Hip)
                ys = (lms[11].y + lms[12].y) / 2
                yh = (lms[23].y + lms[24].y) / 2
                torso_h = abs(ys - yh)
                # Hip center
                xh = (lms[23].x + lms[24].x) / 2
                yh_c = (lms[23].y + lms[24].y) / 2
                
                np.save(f"{output_dir}/meta_{i}.npy", [torso_h, xh, yh_c])
                
                # C. Save Target Landmarks
                landmarks_xy = []
                for lm in lms:
                    landmarks_xy.append([lm.x, lm.y])
                np.save(f"{output_dir}/target_{i}.npy", np.array(landmarks_xy))
                
                processed_count += 1
            else:
                print(f"   No person detected in image: {os.path.basename(img_path)}")
        
        if processed_count == 4:
            total_poses_processed += 1
            print(f"   -> Finished pose: {pose_name}")

    pose_static.close()
    
    print("\n========================================")
    if total_poses_processed > 0:
        print(f"COMPLETED! Successfully processed {total_poses_processed} poses.")
        print(f"Data saved at: {os.path.abspath(ASSETS_ROOT)}")
        print("You can now run the training scripts.")
    else:
        print("NOTHING PROCESSED. Please check input_images folder.")
    print("========================================")

if __name__ == "__main__":
    build_all_assets()