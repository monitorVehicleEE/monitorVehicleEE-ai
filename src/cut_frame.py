import cv2
import os

video_path = "./dataset/vehicle/videos/15.mp4"
video_name = os.path.splitext(os.path.basename(video_path))[0]

output_folder = f"./dataset/vehicle/frames/{video_name}/"

os.makedirs(output_folder, exist_ok=True)
cap = cv2.VideoCapture(video_path)

skip_frame = 5       # lấy mỗi N frame
resize = None    # resize ảnh (None nếu không cần)
diff_threshold = 5     # ngưỡng khác biệt để bỏ frame trùng (0-255)



prev_gray = None
frame_count = 0
save_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frame
    if frame_count % skip_frame != 0:
        frame_count += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # So sánh với frame trước để loại trùng
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        mean_diff = diff.mean()

        if mean_diff < diff_threshold:
            frame_count += 1
            continue  # bỏ frame gần giống

    prev_gray = gray

    # Resize nếu cần
    if resize is not None:
        frame = cv2.resize(frame, resize)

    # Lưu ảnh
    filename = os.path.join(output_folder, f"{video_name}_{save_count:05d}.jpg")
    cv2.imwrite(filename, frame)

    save_count += 1
    frame_count += 1

cap.release()

print(f"Done! {video_name} Saved {save_count} frames.")