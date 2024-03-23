# File description

**Required submitted files:**
- `Material/RenderPy-master/render.py`
- `./result_video.mov`
- `./Readme.txt`
- `./Virtual_Augmented_Reality_Report_cvhm34.pdf`
- `./Material/RenderPy-master/Problem_2_3_5_test_file.ipynb` - Not required but for testing purposes, it was used to test the programme.
- `./Material/RenderPy-master/image.py` - I have modified the class to adjust the OpenCV frame output. -> add numpy array to OpenCV frame.

## How to run

The main file is `render.py`. To run the programme, please use the following command:
- `python3 render.py`

**Note:** Please ensure you have named `IMU_data_path` correctly in `render.py` before running the programme, it at line 41.

**Note:** If you wish to save the file, please change line 476 to the path you wish to save.

**Note:** If you wish not to save the file, please comment out line 476 and line 477.

**Note:** The file contains comments about how I approached the problem and also includes the source paper and references I have read.

**Note:** The file also comments on which problem is which part, it aims to provide convenience when marking.

**Note:** Some comments help me to understand the current process, also helps to debug when something went wrong.

**Note:** For generating the video, I provided the code in `./Material/RenderPy-master/Problem_2_3_5_test_file.ipynb`, please run the code in the Jupyter notebook.

**Note:** The generated video is good to run on VLC player, but the QuickTime player will not show the video correctly.

This might not be the most correct way to solve the problem, but I have tried my best to solve the problems.

Kindly remind: The programme will take around 20 hours to finish all sequences, please be patient, I have provided the command line to show the progress.

Thanks for reading, I hope you enjoy the programme.

## Library I have used

```python
import cv2
import math
import numpy as np
import pandas as pd
import time
```
## Regarding the Video

The video version provided is without the physics engine. Initially, when running with the physics engine, it was functioning; however, after some time, the object would "disappear" because it was "falling down." Therefore, I decided to use the version without the physics engine to showcase the result.

Thank you so much for reading. I hope you enjoy the program :)

