from keras.models import load_model
import cv2
import time
import numpy as np
from random import choice

REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"

    if move1 == "rock":
        if move2 == "scissors":
            return "User"
        if move2 == "paper":
            return "Computer"

    if move1 == "paper":
        if move2 == "rock":
            return "User"
        if move2 == "scissors":
            return "Computer"

    if move1 == "scissors":
        if move2 == "paper":
            return "User"
        if move2 == "rock":
            return "Computer"


model = load_model("rock-paper-scissors-model.h5")

cap = cv2.VideoCapture(0)

last_computer_move = "none"
last_user_move = "none"

waiting_time = 0.8
target_time = time.time() + waiting_time
countdown = 4
play = False
while True:
    
    # launch the game
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    if k == ord('a'):
        play = True
        target_time = time.time() + waiting_time
    
    ret, frame = cap.read()
    if not ret:
        continue

    # rectangle for user to play
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 255, 255), 2)
    # rectangle for computer to play
    cv2.rectangle(frame, (450, 300), (550, 400), (255, 255, 255), 2)
    
    computer_move_name = "none"
    
    if play :
        current_time = time.time()
        if current_time>target_time :
            countdown-=1
            target_time += waiting_time
            if countdown==0 :
                countdown=4
                play = False
                
                # extract the region of image within the user rectangle
                roi = frame[100:400, 100:400]
                img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (227, 227))
            
                # predict the move made
                pred = model.predict(np.array([img]))
                move_code = np.argmax(pred[0])
                user_move_name = mapper(move_code)
                last_user_move = user_move_name
            
                # predict the winner (human vs computer)
                if user_move_name != "none":
                    l = ['rock', 'paper', 'scissors']
                    i = l.index(user_move_name)
                    i = (i+1)%3
                    computer_move_name = l[i]
                    last_computer_move = computer_move_name
        
        

        # display the information
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "Your Move: " + last_user_move,
                    (50, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Computer's Move: " + last_computer_move,
                    (750, 50), font, 1.2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "{}".format(countdown),
            (500, 60), font, 2, (255, 0, 255), 5, cv2.LINE_AA)

    if last_computer_move != "none":
        icon = cv2.imread(
            "images/{}.png".format(last_computer_move))
        icon = cv2.resize(icon, (100, 100))
        frame[300:400, 450:550] = icon
        

    cv2.imshow("Rock Paper Scissors", frame)

cap.release()
cv2.destroyAllWindows()
