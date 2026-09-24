"""
test_camera.py
Standalone test — NOT part of the main app.
Opens your webcam and prints any QR code it detects.
Run this first to confirm camera + QR decoding actually works
on your machine BEFORE we wire it into the Streamlit app.

Run with:  python test_camera.py
Press 'q' in the camera window to quit.
"""

import cv2
from pyzbar.pyzbar import decode

def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    if not cap.isOpened():
        print("❌ Could not open webcam. Check that no other app (Zoom, Teams, etc.) is using it.")
        return

    print("✅ Webcam opened. Point a UPI QR code at the camera. Press 'q' to quit.")

    last_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam.")
            break

        decoded_objects = decode(frame)

        for obj in decoded_objects:
            data = obj.data.decode("utf-8")
            if data != last_data:
                print(f"🔍 QR Code detected: {data}")
                last_data = data

            # Draw a box around the detected QR code
            points = obj.polygon
            if len(points) == 4:
                pts = [(p.x, p.y) for p in points]
                for i in range(4):
                    cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)

        cv2.imshow("QR Scanner Test - press q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()