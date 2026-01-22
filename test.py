import cv2
import mediapipe as mp
import numpy as np
import os
import time
import math

# ==========================================
# CẤU HÌNH BÀI TẬP
# ==========================================
TARGET_POSE_NAME = "Serve"  # Đổi tên thư mục cho từng bài
# ==========================================

# Cấu hình hệ thống
ASSETS_ROOT = "assets"      
SIMILARITY_THRESH = 0.85    

# Khởi tạo MediaPipe
mp_pose = mp.solutions.pose

# --- CÁC HÀM TIỆN ÍCH (UTILITY FUNCTIONS) ---

def overlay_smart_v2(background, overlay_img, ghost_meta, user_lms):
    """Tạo bóng ma Ghost tự động co giãn"""
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay_img.shape[:2]
    
    # 1. User Stats
    u_ys = (user_lms[11].y + user_lms[12].y) / 2
    u_yh = (user_lms[23].y + user_lms[24].y) / 2
    u_xh = (user_lms[23].x + user_lms[24].x) / 2 
    
    user_torso_px = abs(u_ys - u_yh) * h_bg
    
    # 2. Ghost Stats
    ghost_torso_ratio = ghost_meta[0] 
    ghost_hip_x_ratio = ghost_meta[1]
    ghost_hip_y_ratio = ghost_meta[2]
    
    if ghost_torso_ratio == 0 or user_torso_px == 0: 
        return background, (1.0, 0, 0, 0)

    # 3. Calculate Target Scale
    target_h = int(user_torso_px / ghost_torso_ratio)
    aspect_ratio = w_ov / h_ov
    target_w = int(target_h * aspect_ratio)
    
    # Safety Clamp
    if target_h > h_bg * 2.5:
        target_h = int(h_bg * 2.5)
        target_w = int(target_h * aspect_ratio)

    if target_w <= 0 or target_h <= 0: return background, (1.0, 0, 0, 0)
    
    try:
        resized_ghost = cv2.resize(overlay_img, (target_w, target_h))
    except:
        return background, (1.0, 0, 0, 0)

    # 4. Alignment
    u_px_x = int(u_xh * w_bg)
    u_px_y = int(u_yh * h_bg)
    g_px_x = int(ghost_hip_x_ratio * target_w)
    g_px_y = int(ghost_hip_y_ratio * target_h)
    
    top_left_x = u_px_x - g_px_x
    top_left_y = u_px_y - g_px_y
    
    # 5. Draw
    y1, y2 = max(0, top_left_y), min(h_bg, top_left_y + target_h)
    x1, x2 = max(0, top_left_x), min(w_bg, top_left_x + target_w)
    
    gy1 = max(0, -top_left_y)
    gy2 = gy1 + (y2 - y1)
    gx1 = max(0, -top_left_x)
    gx2 = gx1 + (x2 - x1)
    
    if (y2 > y1) and (x2 > x1) and (gy2 > gy1) and (gx2 > gx1):
        alpha = 0.4 
        bg_slice = background[y1:y2, x1:x2]
        ghost_slice = resized_ghost[gy1:gy2, gx1:gx2]
        
        gray_ghost = cv2.cvtColor(ghost_slice, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_ghost, 5, 255, cv2.THRESH_BINARY)
        
        bg_slice_masked = cv2.bitwise_and(bg_slice, bg_slice, mask=mask)
        ghost_slice_masked = cv2.bitwise_and(ghost_slice, ghost_slice, mask=mask)
        blended = cv2.addWeighted(bg_slice_masked, 1-alpha, ghost_slice_masked, alpha, 0)
        
        inv_mask = cv2.bitwise_not(mask)
        bg_slice_bg = cv2.bitwise_and(bg_slice, bg_slice, mask=inv_mask)
        final_slice = cv2.add(bg_slice_bg, blended)
        
        background[y1:y2, x1:x2] = final_slice

    return background, (target_w, target_h, top_left_x, top_left_y)

def draw_transformed_keypoints(img, target_lms_norm, transform_data, user_lms_norm):
    """Vẽ vòng tròn mục tiêu (Target Circles)"""
    h, w = img.shape[:2]
    target_w, target_h, shift_x, shift_y = transform_data
    
    # Các khớp quan trọng
    indices = [11, 12, 13, 14, 15, 16, 25, 26]
    
    score_hits = 0
    total_checks = 0
    THRESHOLD = 0.1 
    
    for idx in indices:
        tx = int(target_lms_norm[idx][0] * target_w + shift_x)
        ty = int(target_lms_norm[idx][1] * target_h + shift_y)
        ux = int(user_lms_norm[idx].x * w)
        uy = int(user_lms_norm[idx].y * h)
        
        dist_px = math.hypot(ux - tx, uy - ty)
        dist_norm = dist_px / w
        
        color = (0, 0, 255) # Đỏ (Chưa khớp)
        if dist_norm < THRESHOLD:
            color = (0, 255, 0) # Xanh (Khớp)
            score_hits += 1
        
        total_checks += 1
        
        if 0 <= tx < w and 0 <= ty < h:
            # Vẽ vòng tròn rỗng
            cv2.circle(img, (tx, ty), 15, color, 2)
            # Vẽ đường nối gợi ý nếu sai
            if color == (0, 0, 255): 
                cv2.line(img, (ux, uy), (tx, ty), (255, 255, 255), 1)

    return score_hits / total_checks if total_checks > 0 else 0

def draw_user_joints(img, user_lms):
    """Vẽ chấm cam (Orange Dots) trên người dùng"""
    h, w = img.shape[:2]
    indices = [11, 12, 13, 14, 15, 16] # Vai, Khuỷu, Cổ tay
    
    for idx in indices:
        lm = user_lms[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        
        # Vẽ viền trắng + nhân cam
        cv2.circle(img, (cx, cy), 9, (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 6, (0, 165, 255), -1) 

def calculate_cosine_similarity(user_lms, target_lms_array):
    """Tính điểm góc xương"""
    CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16), 
        (11, 23), (12, 24), (23, 25), (24, 26)
    ]
    total_score = 0
    valid_connections = 0
    for idx1, idx2 in CONNECTIONS:
        u1 = np.array([user_lms[idx1].x, user_lms[idx1].y])
        u2 = np.array([user_lms[idx2].x, user_lms[idx2].y])
        t1 = target_lms_array[idx1]
        t2 = target_lms_array[idx2]
        
        u_vec, t_vec = u2 - u1, t2 - t1
        norm_u, norm_t = np.linalg.norm(u_vec), np.linalg.norm(t_vec)
        
        if norm_u > 0 and norm_t > 0:
            score = np.dot(u_vec, t_vec) / (norm_u * norm_t)
            total_score += score
            valid_connections += 1
    return total_score / valid_connections if valid_connections > 0 else 0

# --- CHƯƠNG TRÌNH CHẠY CHÍNH (MAIN) ---

def run_trainer():
    # Kiểm tra dữ liệu
    pose_dir = os.path.join(ASSETS_ROOT, TARGET_POSE_NAME)
    if not os.path.exists(pose_dir):
        print(f"❌ Error: Không tìm thấy dữ liệu cho '{TARGET_POSE_NAME}'. Chạy build_assets.py trước!")
        return

    print(f"--- BẮT ĐẦU BÀI TẬP: {TARGET_POSE_NAME} ---")
    window_name = f"Trainer: {TARGET_POSE_NAME}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # Khởi động Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Không thể mở Camera.")
        return

    pose_tracker = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    # Biến trạng thái Game
    current_stage = 0 
    last_match_time = 0
    match_duration = 0 
    REQUIRED_HOLD_TIME = 0.5
    
    # Biến trạng thái Chuẩn bị (Prep Phase)
    # 3 giây đếm ngược khi mới mở file
    POSE_PREP_TIME = 3.0 
    prep_start_time = time.time()
    in_prep_phase = True 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(img_rgb)
        
        # --- LOGIC KIỂM SOÁT CHƯƠNG TRÌNH ---

        # 1. GIAI ĐOẠN CHUẨN BỊ (3s đếm ngược)
        if in_prep_phase:
            elapsed = time.time() - prep_start_time
            remaining = int(POSE_PREP_TIME - elapsed) + 1
            
            # Vẽ khung tối thông báo
            cv2.rectangle(frame, (0,0), (450, 180), (0,0,0), -1)
            cv2.putText(frame, f"BAI TAP: {TARGET_POSE_NAME}", (10, 50), font, 1, (0,255,255), 2)
            cv2.putText(frame, "CHUAN BI...", (10, 100), font, 1, (255,255,255), 2)
            cv2.putText(frame, f"{remaining}", (10, 160), font, 2, (0,255,0), 3)
            
            if elapsed > POSE_PREP_TIME:
                in_prep_phase = False
                last_match_time = time.time()
        
        # 2. GIAI ĐOẠN HOÀN THÀNH (EXERCISE COMPLETE)
        elif current_stage >= 4:
            # Hiện thông báo chữ Xanh to ở giữa
            cv2.putText(frame, "EXERCISE COMPLETE!", (50, h//2), font, 2.5, (0, 255, 0), 5)
            cv2.putText(frame, "Press 'r' to replay", (50, h//2+80), font, 1, (255,255,255), 2)

        # 3. GIAI ĐOẠN ĐANG TẬP (TRAINING PHASE)
        else:
            # Đường dẫn file dữ liệu
            target_file_idx = current_stage
            
            ghost_path = f"{pose_dir}/ghost_{target_file_idx}.png"
            meta_path = f"{pose_dir}/meta_{target_file_idx}.npy"
            target_path = f"{pose_dir}/target_{target_file_idx}.npy"

            # Kiểm tra file tồn tại (Quan trọng)
            if os.path.exists(ghost_path) and os.path.exists(meta_path) and os.path.exists(target_path):
                ghost_img = cv2.imread(ghost_path)
                ghost_meta = np.load(meta_path)
                target_lms = np.load(target_path)
                
                if results.pose_landmarks:
                    user_lms = results.pose_landmarks.landmark
                    
                    # 1. Vẽ Ghost & Lấy thông tin biến đổi
                    frame, transform_data = overlay_smart_v2(frame, ghost_img, ghost_meta, user_lms)
                    
                    # 2. Vẽ điểm khớp trên người dùng (Màu Cam)
                    draw_user_joints(frame, user_lms)

                    # 3. Vẽ điểm mục tiêu trên Ghost (Vòng tròn) & Tính điểm vị trí
                    acc_score = draw_transformed_keypoints(frame, target_lms, transform_data, user_lms)
                    
                    # 4. Tính điểm góc xương
                    skel_score = calculate_cosine_similarity(user_lms, target_lms)
                    
                    # 5. Điểm tổng hợp
                    final_score = (acc_score * 0.4) + (skel_score * 0.6)

                    # --- LOGIC CHẤM ĐIỂM & QUA MÀN ---
                    if final_score > SIMILARITY_THRESH:
                        match_duration = time.time() - last_match_time
                        
                        # Thanh Loading xanh lá
                        ratio = match_duration / REQUIRED_HOLD_TIME
                        bw = min(300, int(ratio * 300))
                        pt2_x = int(50 + bw)
                        cv2.rectangle(frame, (50, h-50), (pt2_x, h-20), (0, 255, 0), -1)
                        
                        if match_duration > REQUIRED_HOLD_TIME:
                            current_stage += 1
                            match_duration = 0
                            last_match_time = time.time()
                    else:
                        match_duration = 0
                        last_match_time = time.time()
                        
                    # Hiển thị điểm số
                    color = (0, 255, 0) if final_score > SIMILARITY_THRESH else (0, 0, 255)
                    cv2.putText(frame, f"DIEM: {int(final_score*100)}%", (50, 80), font, 1.2, color, 3)
                    cv2.putText(frame, f"BUOC {current_stage + 1}/4", (w-300, 60), font, 1, (255,255,0), 2)
            else:
                # Nếu thiếu file, chỉ in lỗi ra console, không hiện lên màn hình
                print(f"⚠️ Warning: Missing data files for stage {current_stage}")
                # Có thể reset hoặc bỏ qua tùy ý

        # Hiển thị màn hình
        cv2.imshow(window_name, frame)
        
        # Phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Quit
            break
        if key == ord('r'): # Reset
            current_stage = 0
            in_prep_phase = True
            prep_start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_trainer()