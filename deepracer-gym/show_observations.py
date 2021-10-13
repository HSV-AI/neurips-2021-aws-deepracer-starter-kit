import pickle
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
import cv2
import os
from PIL import Image
import sys
  
print("This is the name of the program:", sys.argv[0])
  
print("Argument List:", str(sys.argv))

with open(sys.argv[1],'rb') as f:
    obs = pickle.load(f)

    print('Loading',len(obs),'observations')

    size = 320, 120
    fps = 15
    position = (20,20)

    out = cv2.VideoWriter(sys.argv[1]+'.avi', 0, fps, (size[0], size[1]))
    for idx, state in enumerate(obs):
        state = np.swapaxes(state, 0, 2)

        state = np.vstack((state[0], state[1]))
        state = np.rot90(state, 3)

        edges = cv2.Canny(image=state, threshold1=100, threshold2=200) # Canny Edge Detection
        edges = np.dstack([edges, edges, edges])

        state = np.dstack([state, state, state])

        state = cv2.putText(
            state, #numpy array on which text is written
            "Huntsville AI", #text
            position, #position at which writing has to start
            cv2.FONT_HERSHEY_PLAIN, #font family
            1, #font size
            (209, 80, 0, 255), #font color
            1) #font stroke
        
        out.write(state)
    
    #closing all open windows 
    cv2.destroyAllWindows()

    out.release()
