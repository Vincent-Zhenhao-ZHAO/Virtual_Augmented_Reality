############# File description #############
# Requird submitted files:
#   - 1. "Material/RenderPy-master/render.py"
#   - 2. "./result_video.mov"
#   - 3. "./Readme.txt"
#   - 4. "./Virtual_Augmented_Reality_Report_cvhm34.pdf"
#   - 5. "./Material/RenderPy-master/Problem_2_3_5_test_file.ipynb" -> not required but for testing purpose, it was used to test the program.
#   - 6. "./Material/RenderPy-master/image.py -> I have modifed the class to adjust the opencv frame output. -> add numpy array to opencv frame.

############# How to run #############

# The main file is "render.py". To run the program, please use the following command:
#   - python3 render.py
# Note: Please make sure you have name "IMU_data_path" correctly in "render.py" before running the program, it at line 41.
# Note: If you wish to save the file, please change line 476 to the path you wish to save.
# Note: If you wish not to save the file, please comment the line 476 and line 477.

# Note: The file contained comments about how I approach the problem and also included the source paper and references I have read.
# note: The file also commented the which problem is which part, it aims to provide convinence when marking.
# Note: For some commments is to help me to understand the current process, also helps to debug when something went wrong.

# Note: For genering the video, I provided the code in "./Material/RenderPy-master/Problem_2_3_5_test_file.ipynb", please run the code in the jupyter notebook.
# Note: The generated video is good to run on VLC player, but the QuickTime player will not show the video correctly.

# This would not be the most correct way to solve the problem, but I have tried the best to solve the problems.

# Kindly remind: The program will take around 20 hours to finish all sequences, please be patient, I have provided the command line to show the progress.
# Thanks for reading, I hope you enjoy the program.

############# Library I have used #############
import cv2
import math
import numpy as np
import pandas as pd
import time
###############################################

# Problems I have solved: I have finish Problem 1, 2, 3, 4, 5.1; 
# I have not been able to solve the Problem 5.2 and I have not let the multi-headset shown on the screen at the same time.

# For video: It is the version without the physics engine, as when I ran the physics, it was working initially, but after a while, the obejct will 
# "disappear" because it was "falling down", so I have decided to use the version without physics engine to show the result.

##############################################

Thanks so much for reading, I hope you enjoy the program :)
