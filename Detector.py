# Importing the necessary libraries
import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
import paho.mqtt.client as mqtt
from datetime import datetime
import time

# Loading the trained model from the .h5 file
model = tf.keras.models.load_model('modele.h5')

# Initializing ORB object
orb = cv2.ORB_create()

# Setting up video capture from the webcam
video_capture = cv2.VideoCapture(0)

# Defining data hyperparameters for image preprocessing
data_hyperparams = {
    'img_size': (64,64),
    'batch_size': 8,
    'preprocessing_function': tf.keras.applications.densenet.preprocess_input
    }

# Defining the image batch preprocessing function as required by the model:
def preprocess(batch):
  prep_batch = []
  for img in batch:
    preprocessed_img = tf.image.resize(img, data_hyperparams['img_size']) # Resizing the image to 64x64
    preprocessed_img = data_hyperparams['preprocessing_function'](preprocessed_img) # Preprocessing the image with keras
    preprocessed_img = preprocessed_img / 255.0 # Normalizing the pixel values
    prep_batch.append(preprocessed_img) # Append the preprocessed image to the batch of processed images
  return prep_batch

# Defining a function to determine the class of activity based on the model prediction on an individual image
def ImageClassDecoder(prediction):
    """
    Function to get the type of activity detected from the model prediction.
    Takes an array of predictions and returns the corresponding class label and textual description.
    """
    if prediction.argmax() == 0:
        return (0, 'Abuse')
    elif prediction.argmax() == 1:
        return (1, 'Arrest')
    elif prediction.argmax() == 2:
        return (2, 'Arson')
    elif prediction.argmax() == 3:
        return (3, 'Assault')
    elif prediction.argmax() == 4:
        return (4, 'Burglary')
    elif prediction.argmax() == 5:
        return (5, 'Explosion')
    elif prediction.argmax() == 6:
        return (6, 'Fighting')
    elif prediction.argmax() == 7:
        return (7, 'Normal')
    elif prediction.argmax() == 8:
        return (8, 'Road Accident')
    elif prediction.argmax() == 9:
        return (9, 'Robbery')
    elif prediction.argmax() == 10:
        return (10, 'Shooting')
    elif prediction.argmax() == 11:
        return (11, 'Shoplifting')
    elif prediction.argmax() == 12:
        return (12, 'Stealing')
    elif prediction.argmax() == 13:
        return (13, 'Vandalism')

# Same thing, but for a batch of images
def BatchClassDecoder(batch):
    """
    Function to determine the the class of a batch of images.
    The batch class is the most recurrent class among the predictions on individual images.
    """
    classes = []
    counts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for img in batch:
        classes.append(ImageClassDecoder(img)[0])
    for cls in classes:
        counts[cls] += 1
    c = np.array(counts).argmax()
    if c == 0:
        return (0, 'Abuse') # 6/10
    elif c == 1:
        return (1, 'Arrest') # 5/10
    elif c == 2:
        return (2, 'Arson') # 10/10
    elif c == 3:
        return (3, 'Assault') # 8/10
    elif c == 4:
        return (4, 'Burglary') # 10/10
    elif c == 5:
        return (5, 'Explosion') # 9/10
    elif c == 6:
        return (6, 'Fighting') # 5/10
    elif c == 7:
        return (7, 'Normal') # 0/10
    elif c == 8:
        return (8, 'Road Accident') # 8/10
    elif c == 9:
        return (9, 'Robbery') # 8/10
    elif c == 10:
        return (10, 'Shooting') # 10/10
    elif c == 11:
        return (11, 'Shoplifting') #7/10
    elif c == 12:
        return (12, 'Stealing') # 7/10
    elif c == 13:
        return (13, 'Vandalism') # 7/10
    
def BatchProb(batch):
    """
    Function to determine the prediction probability for a batch of images.
    Takes a batch of images and returns its mean probability vector
    """
    S=[]
    for i in range(len(batch[0])):
        s=0
        for elt in batch:
            s+=elt[i]
        S.append(s/len(batch))
    return S

# Creating  a Multiple Instance Learning (MIL) tracker for the Region Of Interest (ROI)
tracker = cv2.TrackerMIL_create()

"""### A. Module I: Human Presence Detection"""

# Detecting the presence of people in the images using the media pipe library

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)  # green


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualized.
  Returns:
    Image with bounding boxes.
  """
  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return image

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')

DRS = DetectionResult(detections=[]) # Initialization of the global detection results object

# Callback function invoked each time an object is detected in the video stream
def get_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    #print('detection result: {}'.format(result))
    global DRS # Global detection result object, accessible from outside the function
    DRS=result

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='efficientdet.tflite'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=1, # maximum number of objects to be detected
    result_callback=get_result,
    category_allowlist = 'person' # We are only interested in detecting the presence of humans
                                  # ?? Every other category will be ignored ??
    )

i=1 # Initial timestamp



"""### E. Modules I, II and III: Detection of Human Presence, Offensive Behaviour and Alteration + Scoring and Notification"""

# Mapping the class labels to the corresponding severity scores:
def scoreMapper(n):
    scores = [6,5,10,8,10,9,5,0,8,8,10,7,7,7]
    return scores[n]

# Associating each range of scores with a severity level
def score2severity(score):
    if 9<=score<=10:
        return ("Critical",5)
    elif 8<=score<9:
        return ("Very High",4)
    elif 7<score<=8:
        return ("High",3)
    elif 4<=score<7:
        return ("Medium",2)
    elif 0<score<4:
        return ("Low",1)
    elif score == 0:
        return ("No Incident",0)
    
# The idea of the following function is that the severity of the General incident is equal to the highest severity
# between the offensive activity incident and the alteration incident. For example, if no offensive activity is
# detected (score1 = 0) but the asset is severely altered (score2 = 9), the incident should be considered critical
# even if the total score (score1+score2) is just 9/20. This is why we use the max().
def Severity(score1, score2):
    level = np.max([score2severity(score1)[1], score2severity(score2)[1]])
    levels = ["No Incident", "Low", "Medium", "High", "Very High", "Critical"]
    return levels[level]

# Defining a function that sends notifications using the MQTT protocol
def send_mqtt_notification(incident_type, severity, human_presence, timestamp, duration, image_data):
    broker_address = "mqtt.example.com"  # MQTT broker address
    broker_port = 1883  # MQTT broker port
    topic_notification = "topic/incident_notification"  # MQTT topic for notifications
    topic_image = "topic/incident_image"  # MQTT topic for the image

    client = mqtt.Client()  # Create a new MQTT client instance
    client.connect(broker_address, broker_port)  # Connect to the MQTT broker

    # Publish the notification message to the MQTT topic for notifications
    notification_message = f"Incident Type: {incident_type}\nSeverity Level: {severity}\nHuman Presence:{human_presence}\nTime: {datetime.fromtimestamp(timestamp).strftime('%d-%m-%y %H:%M')}\nDuration: {duration}"
    client.publish(topic_notification, payload=notification_message)  # Publish the MQTT message to the specified topic

    # Publish the image data to the MQTT topic for the image
    client.publish(topic_image, payload=image_data)  # Publish the image data as the payload

    client.disconnect()  # Disconnect from the MQTT broker

# Initialize a batch to store the frames
frame_batch = []

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

# Initializing the timestamp counter
i=1
score1, score2 = 0,0
presence = False
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Update the object tracker
    ret, bbox = tracker.update(frame)
    if ret:
        ### HUMAN PRESENCE DETECTION: ##########################################################
        ########################################################################################

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(frame))
        with ObjectDetector.create_from_options(options) as detector:
            detector.detect_async(mp_image, i)
            i+=1

            # Defining a binary state variable that indicates whether there is a person near the asset
        if DRS.detections[0].categories[0].category_name == 'person':
            presence = True
        """
        NOTE: Be very careful with the detector.detect_async() function. Its second argument (frame timestamp)
        MUST be GREATER than what the function has processed previously. In other words, If you use incremented 
        counters as timestamps when you first run the function, the first counter of the next execution MUST be GREATER
        than the last counter of the function. The same applies if you use timestamps from the time module.
        """
        
        ### ALTERATION DETECTION: ##############################################################
        ########################################################################################

        # The key idea is to calculate the similarity score for each frame in the batch, and then
        # averaging the results to get the mean alteration score for the batch.

        batch_similarities = [] # Collection of the similarity scores for the frames in the batch

        # Extract features from the current state (ROI)
        roi = frame[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        kp2, des2 = orb.detectAndCompute(roi, None)

        # Perform feature matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)


        
        # Calculate similarity
        if len(kp1) == 0:
            similarity = 0
        else:
            similarity = len(matches) / len(kp1)  # Using the ratio of matches to keypoints as the similarity metric
        batch_similarities.append(similarity)

        ### OFFENSIVE ACTTVITY DETECTION: ######################################################
        ########################################################################################

        frame_batch.append(frame)

        # Check if the frame batch size has reached the desired batch size
        if len(frame_batch) == data_hyperparams['batch_size']:
            # Convert the frame batch to a numpy array
            frame_batch = np.array(frame_batch)

            # Perform preprocessing on the frame batch
            frame_batch = np.array(preprocess(frame_batch))

            # Pass the frame batch to the computer vision model for offensive action detection
            predictions = model.predict(frame_batch)

            # Get the class of the prediction
            Class = BatchClassDecoder(predictions)[1]
            #cv2.putText(frame, str(Class), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0),2)

            # Getting the probability for a specific class
            #probs = BatchProb(predictions)
            #print('Fighting score: ',probs[6])
            
            # computing the severity score for offensive behaviour
            score1 = scoreMapper(BatchClassDecoder(predictions)[0])

            # Calculating the average similarity score
            batch_similarity = np.mean(batch_similarities)

            detection_time = time.time() # Time at which the incidents were detected

            # Defining a threshold to determine if the asset has been altered
            similarity_threshold = 0.4

            # Display the similarity and detected alteration status
            # cv2.putText(frame, sim, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
            print(f"Batch Similarity: {batch_similarity}")
            alteration_severity = ""
            if 0.7*similarity_threshold <=batch_similarity <=0.85*similarity_threshold:
                score2 = 2 # Minor alteration
                alteration_severity = "Minor alteration"
            elif 0.4*similarity_threshold <=batch_similarity <0.7*similarity_threshold:
                score2 = 5 # Moderate alteration
                alteration_severity = "Moderate alteration"
            elif 0 <= batch_similarity < 0.4*similarity_threshold:
                score2 = 9 # Major alteration
                alteration_severity = "Major alteration"
                
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            # Convert the last frame to JPEG format
            _, encoded_image = cv2.imencode(".jpg", frame_batch[-1],encode_param)
            
            # Encode the binary data to base64 string
            image_b64 = str(base64.b64encode(encoded_image))
            # Clear the frame batch and similarity scores list for the next batch of frames
            frame_batch = []
            batch_similarities = []
  
            # Computing the final score
            score = score1 + score2

            # Determining the level of severity from the score
            severity = Severity(score1, score2)
            human_presence = "No"
            if presence:
                human_presence = "Yes" # Human Presence
            # Sending a notification in the MQTT  format depending on the severity: 
            # Uncomment if you have an MQTT broker set up
            if Class != "Normal" and alteration_severity != "":
                send_mqtt_notification(Class + " and " + alteration_severity, severity, human_presence, detection_time, image_b64)
            if Class == "Normal" and alteration_severity != "":
                pass
                send_mqtt_notification(alteration_severity, severity, human_presence, detection_time, image_b64)
        # Display the frame with the bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
        # Displaying the frame
        cv2.imshow('Video', frame)
    # Checking for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break