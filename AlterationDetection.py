# Importing the necessary libraries
import cv2
import tensorflow as tf
import paho.mqtt.client as mqtt
from datetime import datetime
import time
import base64

# Loading the trained model from the .h5 file
model = tf.keras.models.load_model('modele.h5')

# Initializing ORB object
orb = cv2.ORB_create()

# Setting up video capture from the webcam
video_capture = cv2.VideoCapture(0)


# Creating  a Multiple Instance Learning (MIL) tracker for the Region Of Interest (ROI)
tracker = cv2.TrackerMIL_create()


# Associating each range of scores with a severity level
def score2severity(score):
    if 9<=score<=10:
        return ("Critical",5)
    elif 8<=score<9:
        return ("Very High",4)
    elif 7<score< 8:
        return ("High",3)
    elif 4<=score<7:
        return ("Medium",2)
    elif 0<=score<4:
        return ("Low",1)
    

# Defining a function that sends notifications using the MQTT protocol
def send_mqtt_notification(severity, timestamp, image_data):
    broker_address = "maqiatto.com"  # MQTT broker address
    broker_port = 1883  # MQTT broker port
    topic_notification = "womendjia.ivan@gmail.com/anomaly"  # MQTT topic for notifications
    topic_image = "womendjia.ivan@gmail.com/images"  # MQTT topic for the image

    client = mqtt.Client()  # Create a new MQTT client instance
    client.username_pw_set("womendjia.ivan@gmail.com","mqttpass")
    
    client.connect(broker_address, broker_port)  # Connect to the MQTT broker

    # Publish the notification message to the MQTT topic for notifications
    notification_message = f"Severity Level: {severity}_Time: {datetime.fromtimestamp(timestamp).strftime('%d-%m-%y %H:%M')}_Image:{image_data}"
    print("========sending notification==========")
    #print(notification_message)
    client.publish(topic_notification, payload=notification_message)  # Publish the MQTT message to the specified topic

    # Publish the image data to the MQTT topic for the image
    #client.publish(topic_image, payload=image_data)  # Publish the image data as the payload

    client.disconnect()  # Disconnect from the MQTT broker


# Read the initial frame and select the region of interest (ROI)
ret, frame = video_capture.read()
bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
# Initializing the tracker with the defined ROI
tracker.init(frame, bbox)

# Extracting features from the initial state (ROI)
roi = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
kp1, des1 = orb.detectAndCompute(roi, None)

# Creating a Brute Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

score2 = 0
while True:
    # Reading the next frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Updating the object tracker
    ret, bbox = tracker.update(frame)
    if ret:
        # Extracting features from the current state (ROI)
        roi = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        kp2, des2 = orb.detectAndCompute(roi, None)

        # Performing feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)


        # Calculate similarity
        if len(kp1) == 0:
            similarity = 0
        else:
            similarity = len(matches) / len(kp1)  # Using the ratio of matches to keypoints as the similarity metric
        
        # Define a threshold to determine if the asset has been altered
        similarity_threshold = 0.40
        detection_time=0
        alteration_severity = ""
        if 0.7*similarity_threshold <=similarity <=0.85*similarity_threshold:
            score2 = 2 # Minor alteration
            alteration_severity = "Minor alteration"
            detection_time = time.time() # Time at which the incident was detected
        elif 0.4*similarity_threshold <=similarity <0.7*similarity_threshold:
            score2 = 5 # Moderate alteration
            alteration_severity = "Moderate alteration"
            detection_time = time.time() # Time at which the incident was detected
        elif 0 <= similarity < 0.4*similarity_threshold:
            score2 = 9 # Major alteration
            alteration_severity = "Major alteration"
            detection_time = time.time() # Time at which the incident was detected

        # Display the similarity and detected alteration status
        sim = f"Similarity: {similarity}"
        cv2.putText(frame, sim, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
        #print(f"Similarity: {similarity}")
         # Determining the level of severity from the score
        severity = score2severity(score2)
        #print(score2, severity)

            
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        # Convert the last frame to JPEG format
        _, encoded_image = cv2.imencode(".jpg", frame,encode_param)
        
        # Encode the binary data to base64 string
        image_b64 = str(base64.b64encode(encoded_image))


        # Sending a notification in the MQTT  format depending on the severity:
        # Uncomment if you have an MQTT broket set up

        if alteration_severity != "":
                send_mqtt_notification(severity, detection_time, image_b64)
        
        
    # Display the frame with the bounding box
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
    cv2.imshow("Asset Alteration Detection", frame)

    # Checking for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
