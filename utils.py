
from threading import Thread
import time
from pathlib import Path
import cv2

from flask import jsonify, request, make_response, render_template, Response


class EndpointHandler(object):
    def __init__(self, action):
        self.action = action 

    def __call__(self, *args, **kwargs):
        response = self.action(*args, **request.view_args)
        return make_response(response)

class FlaskAppWrapper(object):
    def __init__(self, app, **configs):
        self.app = app
        self.configs(**configs)

    def configs(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value

    def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, methods=['GET'], *args, **kwargs):
        self.app.add_url_rule(endpoint, endpoint_name, EndpointHandler(handler), methods=methods, *args, **kwargs)

    def run(self, **kwargs):
        self.app.run(**kwargs)
        
        
class WebAppUtils:
    def __init__(self):
        self.framecopy = None
        self.net_people_count = None
        self.current_fps = None
    
    def favicon(self):
        return 'dummy', 200
    
    # used to render computer vision in browser
    def gen_frames(self):

        while True:
            if self.framecopy is None:
                continue

            ret, buffer = cv2.imencode('.jpg', self.framecopy)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def get_people(self):
        if self.net_people_count != None:
            return jsonify(self.net_people_count)
        else:
            return jsonify("server error"),500

    def get_fps(self):
        if self.current_fps != None:
            return jsonify(self.current_fps)
        else:
            return jsonify("server error"),500

    def video_feed(self):
        # Video streaming route. Put this in the src attribute of an img tag
        return Response(self.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def index(self):
        # Video streaming home page
        return render_template('index.html')


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True




