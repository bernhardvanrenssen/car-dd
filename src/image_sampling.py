import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
video_path = "/home/bernhard/projects/car-dd/data/video/ecosport_4k.MOV"  # Adjust if needed
output_folder = "selected_frames"
k = 20                 # number of clusters (final frames)
frame_skip = 10        # skip every N frames for faster processing
max_frames = 1000      # optional safeguard - only process first N*frame_skip frames

print("File exists?", os.path.exists(video_path))

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# ---------------------------------------------------------
# 1. READ & DOWNSAMPLE FRAMES
# ---------------------------------------------------------
cap = cv2.VideoCapture(video_path)
all_frames = []         # will store tuples of (frame_index, frame)
feature_vectors = []    # will store histogram vectors

frame_index = 0
read_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # If we're skipping frames, only process if index % frame_skip == 0
    if frame_index % frame_skip == 0:
        # Optional: downscale the frame to speed up histogram calculation
        # (e.g., to 640x360) – you can comment out if you want full-res
        frame_small = cv2.resize(frame, (640, 360))
        
        # Compute a simple 3D color histogram over the *entire* frame
        hist = cv2.calcHist([frame_small], [0,1,2], None, [8,8,8],
                            [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        
        all_frames.append((frame_index, frame))  # store the original frame
        feature_vectors.append(hist)
        
        read_count += 1
        # Safety check if you only want to process up to max_frames
        if read_count >= max_frames:
            break

    frame_index += 1

cap.release()

if not feature_vectors:
    print("No frames were loaded. Check your video path or frame_skip settings.")
    exit(0)

feature_vectors = np.array(feature_vectors, dtype=np.float32)

print(f"Loaded {len(feature_vectors)} frames (after skipping).")

# ---------------------------------------------------------
# 2. CLUSTER FRAMES (K-MEANS)
# ---------------------------------------------------------
print("Clustering frames, please wait...")
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(feature_vectors)

# ---------------------------------------------------------
# 3. PICK A REPRESENTATIVE FRAME FROM EACH CLUSTER
# ---------------------------------------------------------
final_frames = []

for cluster_id in range(k):
    # Indices of all frames in this cluster
    cluster_indices = np.where(labels == cluster_id)[0]
    
    # Find frame whose feature vector is nearest to the cluster centroid
    centroid = kmeans.cluster_centers_[cluster_id]
    
    best_index = None
    best_dist = float('inf')
    
    for ci in cluster_indices:
        dist = np.linalg.norm(feature_vectors[ci] - centroid)
        if dist < best_dist:
            best_dist = dist
            best_index = ci
    
    final_frames.append(best_index)

print("Selected frame indices (in the downsampled list):", final_frames)

# ---------------------------------------------------------
# 4. SAVE THE SELECTED FRAMES
# ---------------------------------------------------------
for i, idx_in_list in enumerate(final_frames):
    frame_id, original_frame = all_frames[idx_in_list]
    outpath = os.path.join(output_folder, f"cluster_{i:02d}_frame_{frame_id}.jpg")
    cv2.imwrite(outpath, original_frame)
    
print(f"Saved {len(final_frames)} representative frames to '{output_folder}'.")
