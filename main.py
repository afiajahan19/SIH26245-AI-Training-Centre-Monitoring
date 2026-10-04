from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
video = cv2.VideoCapture("test_video.mp4")

if not video.isOpened():
    print("❌ Could not open video")
    exit()

while True:

    ret, frame = video.read()

    if not ret:
        break

    # YOLO tracking
    results = model.track(
        frame,
        persist=True,
        classes=[0],          # Person only
        conf=0.35,
        tracker="bytetrack.yaml",
        verbose=False
    )

    person_count = 0

    if results[0].boxes is not None:

        boxes = results[0].boxes

        for box in boxes:

            confidence = float(box.conf[0])

            # Get tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                track_id = -1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            person_count += 1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Display ID + confidence
            label = f"ID {track_id} | {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # Display count
    cv2.rectangle(
        frame,
        (10, 10),
        (350, 65),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"People Present: {person_count}",
        (20, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "SIH26245 - Attendance Monitoring",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()