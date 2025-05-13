import cv2
import numpy as np
import time

# Mở video
cap = cv2.VideoCapture(0)  # Thay bằng đường dẫn video của bạn

# Biến thời gian để tính FPS
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tính FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # Lấy kích thước khung hình
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Dùng HoughCircles để tìm hình tròn
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=5, maxRadius=100)

    if circles is not None:
        for circle in circles[0]:
            x, y, r = circle.astype("int")

            # Tính tọa độ tương đối
            relative_x = x - center_x
            relative_y = y - center_y
            # print(f"Tọa độ hình tròn (so với tâm): ({relative_x}, {relative_y})")

            # Vẽ hình tròn và tâm
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)

    # Vẽ tâm khung hình
    cv2.circle(frame, (center_x, center_y), 2, (255, 0, 0), 3)

    # Vẽ FPS lên khung hình
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Hiển thị khung hình
    cv2.imshow("Tracking Circle", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
