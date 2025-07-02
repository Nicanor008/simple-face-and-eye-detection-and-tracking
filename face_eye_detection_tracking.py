import cv2

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize tracker
tracker = cv2.TrackerCSRT_create()
tracking = False
blink_count = 0
eye_closed_frames = 0
blink_threshold = 3  # Number of frames with no eyes to count as a blink

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
    exit()

print("✅ Webcam started. Press 'cmd/ctrl' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not tracking:
        # Detect face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            tracking = True
    else:
        # Track face
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Face (Tracked)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10)

            if len(eyes) == 0:
                eye_closed_frames += 1
            else:
                if eye_closed_frames >= blink_threshold:
                    blink_count += 1
                    print(f"Blink Detected! Total: {blink_count}")
                eye_closed_frames = 0

            # Sort eyes by horizontal position
            eyes = sorted(eyes, key=lambda e: e[0])
            for idx, (ex, ey, ew, eh) in enumerate(eyes):
                label = "Left Eye" if idx == 0 else "Right Eye" if idx == 1 else f"Eye {idx+1}"
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, label, (ex, ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            tracking = False

    # Show blink count
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display result
    cv2.imshow("Face Tracker + Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
