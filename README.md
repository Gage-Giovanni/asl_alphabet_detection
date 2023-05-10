# ASL Alphabet Detection

Gage Giovanni  
Final Project for CPSC 491  
CSU Fullerton  
Dr. Lidia Morrison  

# .exe download location and instructions
This repository contains all source code and files used to build the ASL Alphabet Detector  
To run this program more easily, download the deployed project with .exe from the link below

https://drive.google.com/file/d/1_kx8-ozKwrqrlTd-62Q2bJ_LykSuqT6d/view?usp=share_link

After downloading, run asl_alphabet_detector.exe to launch the program

# Python environment instructions
Dependencies and setup:

- Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019

If running python file directly, run the following commands to install dependencies and run the main program (tested on Python 3.10.7):

- python install_tf.py

- pip install tensorflow --upgrade
- pip install dotenv
- pip install load_dotenv
- pip install matplotlib
- pip install pyyaml
- pip install gin-config
- pip install tensorflow-addons
- pip install opencv-python
- pip install PySimpleGUI

- python asl_alphabet_detector.py

Only needed if training a new model with pictures and labels in the 'train' and 'test' folders:

- pip install pytz
- pip install pycocotools

- python verify_tf.py

If the above returns OK, continue. Otherwise, troubleshoot Tensorflow installation

- python train_model.py
- Run the command that train_model.py prints


# Settings

Detection Mode - will determine the video input source for object detection:
- 'Real Time Detection' will draw input from the selected video input device and provide random prompts for training
- 'Demo' will use the provided demo.mp4 video to detect objects from. This mode will prompt the letters 'a' through 'z' in order

Input Device - determines the video input device to be used in 'Real Time Detection' mode.
- If video input problems are encountered, try selecting different Input Device numbers

Confidence Threshold - selects a number 1 through 100 to be used as a threshold at which boxes will be drawn and a correct sign will provide a point
- If the program is having trouble detecting signs, lowering this setting will make object detection more forgiving

Max Boxes - selects a number 1 through 5 to be used as the maximum number of boxes shown at any given time
