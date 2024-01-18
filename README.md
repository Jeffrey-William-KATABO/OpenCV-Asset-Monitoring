# OpenCV-Asset-Monitoring
Intelligent video asset surveillance system for detecting human presence, alteration, and offensive actions. 

For human presence detection only, use PresenceDetection.py. For Alteration detection only, use AlterationDetection.py. 
For Offensive activity detection only, use OffenseDetection.py. For a combination of the 3 detection modes, use Detector.py.
Note that the latter has a significantly higher latency in the video stream than the others.
The offensive activity detection is based on a model trained on the UCF Crime dataset. The model is developed here: https://www.kaggle.com/code/annakzvereva/crime-video-detection.

The presence detection system is based on a mediapipe model. You can find more info about this here: https://developers.google.com/mediapipe/solutions/vision/object_detector/python
