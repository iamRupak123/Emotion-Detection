import cv2
import requests

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

emotion = ""
confidence = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # Draw faces
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        # Show prediction above rectangle
        if emotion:

            text = f"{emotion.upper()} {confidence}%"

            cv2.putText(
                frame,
                text,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Press S to Analyze",
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    # Display webcam
    cv2.imshow(
        "Emotion Detection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # --------------------------------
    # S → Predict
    # --------------------------------

    if key == ord("s"):

        if len(faces) > 0:

            x, y, w, h = faces[0]

            # Crop face
            face = frame[y:y+h, x:x+w]

            # BGR → RGB
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            # Resize
            face = cv2.resize(
                face,
                (128, 128)
            )

            # RGB → BGR for JPEG
            face = cv2.cvtColor(
                face,
                cv2.COLOR_RGB2BGR
            )

            # Encode
            success, encoded_image = cv2.imencode(
                ".jpg",
                face
            )

            if success:

                files = {
                    "image": (
                        "face.jpg",
                        encoded_image.tobytes(),
                        "image/jpeg"
                    )
                }

                response = requests.post(
                    "http://127.0.0.1:5000/predict",
                    files=files
                )

                if response.status_code == 200:

                    result = response.json()

                    emotion = result["emotion"]
                    confidence = result["confidence"]

                    print(
                        f"Prediction: {emotion}"
                    )

                    print(
                        f"Confidence: {confidence}%"
                    )

                else:

                    print(
                        "Backend Error:",
                        response.text
                    )

        else:

            print("No face detected!")

    # --------------------------------
    # Q → Quit
    # --------------------------------

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()