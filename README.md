# pi_app_tf_lite
Full Stack Tensor Flow Lite Rasp Pi App for People Detecting tested on Buster B+ Model


* Install TF Lite on Rasp Pi and setup on virtual env:
https://youtu.be/vekblEk6UPc

* pip install flask:
`$ python -m pip install flask`

* pip install paho mqtt:
`$ python -m pip install paho-mqtt`

* Start web app
`$ python app.py`

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


