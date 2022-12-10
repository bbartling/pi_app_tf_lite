import cv2
import time
import numpy as np

from flask import Flask
import threading
import argparse
import paho.mqtt.client as mqtt
import socket
import json

from tensorflow.lite.python.interpreter import Interpreter
import utils


ap = argparse.ArgumentParser()
ap.add_argument(
    "--confidence",
    type=float,
    default=0.6,
    help="minimum probability to filter weak detections",
)

ap.add_argument("--use-flask", default=True, action="store_true")
ap.add_argument("--no-flask", dest="use-flask", action="store_false")

ap.add_argument("--use-mqtt", default=False, action="store_true")
ap.add_argument("--no-mqtt", dest="use-mqtt", action="store_false")


args = ap.parse_args()
CONFIDENCE = args.confidence
print("CONFIDENCE SET FOR PEOPLE DETECTION IS: ", CONFIDENCE)

# MQTT server environment variables
USE_MQTT = args.use_mqtt
print("MQTT APP ENABLED IS: ", USE_MQTT)

BROKER = "test.mosquitto.org"
IPADDRESS = socket.gethostbyname(BROKER)
MQTT_HOST = IPADDRESS
MQTT_PORT = 1883
MQTT_KEEPALIVE_INTERVAL = 60


# flask web app configs
USE_FLASK = args.use_flask
print("WEB APP ENABLED IS: ", USE_FLASK)


# create deep copy for text on cv results
# text gets garbled with out this deep copy
def insert_people_count(image, people, fps):

    cv2.putText(
        image,
        f"People: {people}",
        (1, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

    cv2.putText(
        image,
        f"Fps: {str(round(fps, 2))}",
        (1, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
    )

    return image



def detect(interpreter, videostream, routes):

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    floating_model = input_details[0]["dtype"] == np.float32

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]["name"]

    if "StatefulPartitionedCall" in outname:  # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else:  # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:

        people = 0

        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame = videostream.read()

        # rotated for pi camera
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_rgb_resized, axis=0)

        # for web app rendering
        frame_bgr_resized = cv2.resize(frame, (width, height))

        # for drawing boxes around people
        frame_width = frame_rgb_resized.shape[0]
        frame_height = frame_rgb_resized.shape[1]      
        
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]["index"])[
            0
        ]  # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]["index"])[
            0
        ]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]["index"])[
            0
        ]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            
            if (scores[i] > CONFIDENCE) and (scores[i] <= 1.0):
                
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions
                # need to force them to be within image using max() and min()

                ymin = int(max(1,(boxes[i][0] * frame_height)))
                xmin = int(max(1,(boxes[i][1] * frame_width)))
                ymax = int(min(frame_height,(boxes[i][2] * frame_height)))
                xmax = int(min(frame_width,(boxes[i][3] * frame_width)))

                cv2.rectangle(frame_bgr_resized, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)

                # For the text background
                # Finds space required
                (w, h), _ = cv2.getTextSize(
                    f"{scores[i]:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                # Prints the text.
                cv2.rectangle(
                    frame_bgr_resized,
                    (xmin, ymin + h + 5),
                    (xmin + w + 5, ymin),
                    (255, 255, 255),
                    -1,
                )

                cv2.putText(
                    frame_bgr_resized,
                    f"{scores[i]:.2f}",
                    (xmin, ymin + h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

                people += 1

        # print("FPS: ",frame_rate_calc)
        final_frame = insert_people_count(frame_bgr_resized, people, frame_rate_calc)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # generate frame for web app
        routes.framecopy = final_frame
        routes.current_fps = round(frame_rate_calc, 4)
        routes.net_people_count = people

        if USE_MQTT:

            # publish on mqtt bus or not
            if routes.net_people_count == None or routes.net_people_count != people:
                print("publishing people count: ", people)
                client.publish(
                    "people",
                    json.dumps({"people-count": people, "fps": frame_rate_calc}),
                )


if __name__ == "__main__":

    """
    MQTT APP SETUP
    """
    if USE_MQTT:
        client = mqtt.Client()
        client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    """
    FLASK APP SETUP
    """

    routes = utils.WebAppUtils()

    if USE_FLASK:
        flask_app = Flask(__name__)
        app = utils.FlaskAppWrapper(flask_app)

        app.add_endpoint("/favicon.ico/", "favicon", routes.favicon)
        app.add_endpoint("/people-count/", "get_people", routes.get_people)
        app.add_endpoint("/fps/", "get_fps", routes.get_fps)
        app.add_endpoint("/video-feed/", "video_feed", routes.video_feed)
        app.add_endpoint("/", "index", routes.index)

        threaded_flask_app = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=5000, use_reloader=False)
        )

        threaded_flask_app.setDaemon(True)
        threaded_flask_app.start()

    """
    LOAD MODEL 
    """
    MODEL = "./model/person_model0.tflite"

    interpreter = Interpreter(model_path=MODEL)

    # Create a video player
    videostream = utils.VideoStream().start()

    try:
        detect(interpreter, videostream, routes)

    except KeyboardInterrupt:
        print("trying to exit gracefully")
        print("last PFS measured at: ", routes.current_fps)
        videostream.stop()
        print("Killing Flask App Thread")
        exit(0)
