import cv2
import numpy as np
import pandas as pd

# ============================================
# SETUP
# ============================================

cap = cv2.VideoCapture('assets/Samplecvfootage.mp4')
data = []
frame_count = 0

ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    exit()

# Select car to track
print("Select the car to track")
boundingbox = cv2.selectROI("Select Car", first_frame, True)
cv2.destroyAllWindows()

# Create tracker
params = cv2.TrackerCSRT_Params()
params.use_color_names = True
params.use_hog = True
params.number_of_scales = 50
params.scale_step = 1.05
params.use_segmentation = True

tracker = cv2.TrackerCSRT_create(params)
tracker.init(first_frame, boundingbox)

# Get initial car size
x, y, w, h = [int(v) for v in boundingbox]
initial_area = w * h
print(f"Initial car area: {initial_area} px²")
print("Tracking started! Press SPACE to quit.")

# ============================================
# Sizes
# ============================================

BLOCK_SIZE = 5          # Frames per block
current_block = []     
block_averages = []     

#Overtake chances based on size ratios
OVERTAKE_RATIO_THRESHOLD = 1.5  
MAX_RATIO = 3.0                 

approach_start_frame = None
approach_duration = 0
max_approach_speed = 0
is_approaching = False

chance_history = []
SMOOTH_WINDOW = 10

# ============================================

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    success, boundingbox = tracker.update(frame)
    
    if success:
        # ----- GET CURRENT SIZE -----
        x, y, w, h = [int(v) for v in boundingbox]
        box_area = w * h
        
        # ----- ADD TO CURRENT BLOCK -----
        current_block.append(box_area)
        
        # ----- WHEN BLOCK IS FULL -----
        if len(current_block) >= BLOCK_SIZE:
            # Calculate average of this block
            block_avg = np.mean(current_block)
            block_averages.append(block_avg)
            
            # Keep only last 2 blocks
            if len(block_averages) > 2:
                block_averages.pop(0)
            
            # Reset block
            current_block = []
            
            # ----- COMPARE PAST 5 FRAME BLOCKS AGAINST 5 BLOCKS  -----
            if len(block_averages) >= 2:
                prev_avg = block_averages[-2]
                current_avg = block_averages[-1]
                block_diff = current_avg - prev_avg
                
                # ----- DETECT APPROACHING VS MOVING AWAY -----
                if block_diff > 0:
                    status = "APPROACHING"
                    status_color = (0, 255, 0)
                    is_approaching = True
                    
                    if approach_start_frame is None:
                        approach_start_frame = frame_count
                    
                    approach_duration = frame_count - approach_start_frame
                    max_approach_speed = max(max_approach_speed, block_diff)
                else:
                    status = "MOVING AWAY"
                    status_color = (0, 0, 255)
                    is_approaching = False
                    approach_start_frame = None
                    approach_duration = 0
                    max_approach_speed = 0
            
            # ----- SIZE RATIO -----
            area_ratio = block_avg / initial_area
            
        else:
            # Not enough frames yet
            area_ratio = box_area / initial_area
            if 'status' not in locals():
                status = "COLLECTING DATA..."
                status_color = (255, 255, 255)
        
        # ============================================
        # CALCULATE OVERTAKE CHANCE
        # ============================================
        
        # 1. Size Score (0-50%)
        if area_ratio >= MAX_RATIO:
            size_score = 50
        elif area_ratio >= OVERTAKE_RATIO_THRESHOLD:
            size_score = ((area_ratio - OVERTAKE_RATIO_THRESHOLD) / 
                         (MAX_RATIO - OVERTAKE_RATIO_THRESHOLD)) * 50
        elif area_ratio >= 1.2:
            size_score = ((area_ratio - 1.2) / 0.3) * 15
        else:
            size_score = 0
        
        # 2. Speed Score (0-30%)
        speed_score = 0
        if is_approaching and max_approach_speed > 0 and area_ratio > 1.3:
            speed_score = min(30, (max_approach_speed / 200) * 30)
        
        # 3. Duration Penalty (0-20%)
        duration_penalty = 0
        if is_approaching and approach_duration > 10 and area_ratio > 1.3:
            if approach_duration < 30:
                duration_penalty = ((approach_duration - 10) / 20) * 10
            elif approach_duration < 50:
                duration_penalty = 10 + ((approach_duration - 30) / 20) * 10
            else:
                duration_penalty = 20
        
        # 4. Minimum chance based on size
        min_chance = 0
        if area_ratio >= 2.0:
            min_chance = 15
        elif area_ratio >= 1.7:
            min_chance = 10
        elif area_ratio >= 1.5:
            min_chance = 5
        
        # Final chance
        raw_chance = size_score + speed_score - duration_penalty
        raw_chance = max(raw_chance, min_chance)
        if area_ratio > 1.2:
            raw_chance += 10
        
        overtake_chance = max(0, min(100, int(raw_chance)))
        
        # Smooth the chance
        chance_history.append(overtake_chance)
        if len(chance_history) > SMOOTH_WINDOW:
            chance_history.pop(0)
        
        if len(chance_history) >= SMOOTH_WINDOW:
            smooth_chance = int(np.mean(chance_history))
        else:
            smooth_chance = overtake_chance
        
        # ----- OVERTAKE STATUS -----
        if smooth_chance >= 70:
            overtake_status = "HIGH OVERTAKE CHANCE!"
            overtake_color = (0, 255, 0)
        elif smooth_chance >= 45:
            overtake_status = "MEDIUM OVERTAKE CHANCE"
            overtake_color = (0, 255, 255)
        elif smooth_chance >= 25:
            overtake_status = "LOW OVERTAKE CHANCE"
            overtake_color = (0, 165, 255)
        elif smooth_chance >= 10:
            overtake_status = "WATCHING FOR OVERTAKE"
            overtake_color = (255, 165, 0)
        else:
            overtake_status = "NO OVERTAKE OPPORTUNITY"
            overtake_color = (0, 0, 255)
        
        # ----- SAVE DATA -----
        data.append({
            'frame': frame_count,
            'area_ratio': area_ratio,
            'block_diff': block_diff if len(block_averages) >= 2 else 0,
            'status': status,
            'overtake_chance': smooth_chance,
            'duration_penalty': duration_penalty,
        })
        
        # ============================================
        # DRAW ON SCREEN
        # ============================================
        
        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Text info
        y = 25
        cv2.putText(frame, f"Status: {status}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, overtake_status, (10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, overtake_color, 2)
        cv2.putText(frame, f"Size: {area_ratio:.2f}x", (10, y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # Overtake bar
        bar_x, bar_y, bar_w, bar_h = 10, 150, 400, 40
        fill_w = int((smooth_chance / 100) * bar_w)
        
        if smooth_chance < 25:
            bar_color = (0, 0, 255)
        elif smooth_chance < 50:
            bar_color = (0, 165, 255)
        elif smooth_chance < 70:
            bar_color = (0, 255, 255)
        else:
            bar_color = (0, 255, 0)
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
        cv2.putText(frame, f"{smooth_chance}% OVERTAKE", (bar_x + bar_w//2 - 60, bar_y + 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Block progress
        block_progress = len(current_block) / BLOCK_SIZE
        cv2.rectangle(frame, (bar_x + bar_w + 20, bar_y), 
                     (bar_x + bar_w + 120, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x + bar_w + 20, bar_y), 
                     (bar_x + bar_w + 20 + int(block_progress * 100), bar_y + 10), 
                     (0, 255, 255), -1)
        cv2.putText(frame, f"Block {len(current_block)}/{BLOCK_SIZE}", 
                   (bar_x + bar_w + 20, bar_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
        # Legend
        lx = bar_x + bar_w + 20
        ly = bar_y + 35
        cv2.rectangle(frame, (lx, ly), (lx + 15, ly + 10), (0, 255, 0), -1)
        cv2.putText(frame, "Approaching", (lx + 20, ly + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        cv2.rectangle(frame, (lx, ly + 15), (lx + 15, ly + 25), (0, 0, 255), -1)
        cv2.putText(frame, "Moving Away", (lx + 20, ly + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
        
    else:
        cv2.putText(frame, "Tracking Lost!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        data.append({'frame': frame_count, 'tracking': 'lost'})
    
    cv2.imshow('Overtake Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# ============================================
# CLEANUP
# ============================================

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv('overtake_data.csv', index=False)

print(f"\nTracking complete! {len(df)} frames recorded.")
print(f"Data saved to: overtake_data.csv")