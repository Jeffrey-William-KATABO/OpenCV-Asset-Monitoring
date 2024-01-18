# Importing the necessary libraries
import cv2
import tensorflow as tf
import numpy as np
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


"""### B. Module II: Offensive Activity Detection"""

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

# Defining a function that sends notifications using the MQTT protocol
def send_mqtt_notification(incident_type, severity, timestamp, image_data):
    broker_address = "maqiatto.com"  # MQTT broker address
    broker_port = 1883  # MQTT broker port
    topic_notification = "womendjia.ivan@gmail.com/anomaly"  # MQTT topic for notifications
    topic_image = "womendjia.ivan@gmail.com/images"  # MQTT topic for the image

    client = mqtt.Client()  # Create a new MQTT client instance
    client.username_pw_set("womendjia.ivan@gmail.com","mqttpass")
    
    client.connect(broker_address, broker_port)  # Connect to the MQTT broker

    # Publish the notification message to the MQTT topic for notifications
    notification_message = f"Incident Type: {incident_type}_Severity Level: {severity}_Time: {datetime.fromtimestamp(timestamp).strftime('%d-%m-%y %H:%M')}_Image:{image_data}"
    print("========sending notification==========")
    print(notification_message)
    client.publish(topic_notification, payload=notification_message)  # Publish the MQTT message to the specified topic

    # Publish the image data to the MQTT topic for the image
    #client.publish(topic_image, payload=image_data)  # Publish the image data as the payload

    client.disconnect()  # Disconnect from the MQTT broker


# Initialize a batch to store the frames
frame_batch = []
score1 = 0
while True:
    ret, frame = video_capture.read()
    if ret:
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
            probs = BatchProb(predictions)
            print('Fighting score: ', probs[6])

            # computing the severity score for offensive behaviour
            score1 = scoreMapper(BatchClassDecoder(predictions)[0])

            detection_time = time.time() # Time at which the incident was detected

                
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
            # Convert the last frame to JPEG format
            _, encoded_image = cv2.imencode(".jpg", frame_batch[-1],encode_param)
            
            # Encode the binary data to base64 string
            image_b64 = str(base64.b64encode(encoded_image))

            # Determining the level of severity from the score
            severity = score2severity(score1)
            print(score1, severity)

                
            # Sending a notification in the MQTT  format depending on the severity: 
            # Uncomment if you have an MQTT broket set up
            
            
            if Class != "Normal":
                send_mqtt_notification(Class, severity, detection_time, image_b64)
            

            # Clear the frame batch and similarity scores list for the next batch of frames
            frame_batch = []
        cv2.imshow('Video', frame)
    # Check for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()