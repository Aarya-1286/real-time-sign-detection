import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import time
from gtts import gTTS
from io import BytesIO
import pygame

def speak(text, language='en'):
    mp3_fo = BytesIO()
    tts = gTTS(text, lang=language)
    tts.write_to_fp(mp3_fo)
    mp3_fo.seek(0)

    pygame.mixer.init()
    pygame.mixer.music.load(mp3_fo, 'mp3')
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)



# Load your trained TensorFlow model
model = keras.models.load_model('model.h5')

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

word_buffer = ""
text_buffer = ""
last_predicted_character = None
start_time = None

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == model.input_shape[1]:
            prediction = model.predict(np.array([data_aux]))
            predicted_character_index = np.argmax(prediction[0])
            predicted_character = labels_dict[predicted_character_index]

            if predicted_character == last_predicted_character:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 1:
                    word_buffer += predicted_character
                    last_predicted_character = None
                    start_time = None
            else:
                last_predicted_character = predicted_character
                start_time = None

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)
    else:
        last_predicted_character = None
        start_time = None

    # Display the text buffer on the frame
    cv2.putText(frame, word_buffer, (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, text_buffer, (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    cv2.imshow('frame', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('a'): 
        speak(word_buffer)
        text_buffer += word_buffer + " "
        word_buffer=""
    elif key == ord('z'): # Backspace
        word_buffer = word_buffer[:-1]

cap.release()
cv2.destroyAllWindows()