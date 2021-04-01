import cv2

if __name__ == '__main__':
    model_path = 'models/haarcascade_frontalface_default.xml'

    video = cv2.VideoCapture(0)
    classifier = cv2.CascadeClassifier(model_path)

    while True:
        (ret, frame) = video.read()

        bounding_boxes = classifier.detectMultiScale(frame)

        for box in bounding_boxes:
            (x, y, width, height) = box

            frame = cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (255, 0, 255),
                    5
            )

            cv2.imshow('webcam feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
