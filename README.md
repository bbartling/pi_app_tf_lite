# pi_app_tf_lite
Full Stack Tensor Flow Lite Rasp Pi App for People Detecting tested on headless Buster B+ Model. The idea is to deploy apps remotely and monitor computer vision results through the web app features and have IoT log rest API endpoints or MQTT.


* Install TF Lite on Rasp Pi and setup on virtual env:
https://youtu.be/vekblEk6UPc

* pip install flask:
`$ python -m pip install flask`

* pip install paho mqtt:
`$ python -m pip install paho-mqtt`

* Start web app
`$ python app.py`

* If started successfully in the console:
'''
CONFIDENCE SET FOR PEOPLE DETECTION IS:  0.6
MQTT APP ENABLED IS:  False
WEB APP ENABLED IS:  True
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
 * Serving Flask app 'bens_app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://10.100.100.17:5000
Press CTRL+C to quit
'''

* Dial into computer vision output on port 5000

![exampleSnip](/images/cap.PNG)

* Rest API endpoint for detected people

![exampleSnip](/images/peoplecount.PNG)

* Rest API endpoint for frame rate per second

![exampleSnip](/images/fps.PNG)

# optional args:
`--use-mqtt` (use MQTT)
`--no-flask` (disable web app features)
`--confidence` (detection confidence)


