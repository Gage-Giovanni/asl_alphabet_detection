import os
import random
import PySimpleGUI as sg
from dotenv import load_dotenv

import tensorflow as tf

from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import visualization_utils
from object_detection.utils import label_map_util

import cv2
import numpy as np
from matplotlib import pyplot as plt

load_dotenv()

settings = {'mode': 'Real Time Detection', 'input_device_id': 0, 'confidence_threshold': 80, 'box_limit': 2}

CONFIG = os.path.join(os.getenv('CUSTOM_MODELS'), os.getenv('NEW_MODEL'), 'pipeline.config')

# Load config settings for custom object detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG)
obj_det_model = model_builder.build(model_config = configs['model'], is_training = False)

# Load custom object detection model checkpoint
checkpoint = tf.compat.v2.train.Checkpoint(model = obj_det_model)
checkpoint.restore(os.path.join(os.getenv('CUSTOM_MODELS'), os.getenv('NEW_MODEL'), 'ckpt-' + os.getenv('CHECKPOINT_NUM'))).expect_partial()

# Retrieve list of object labels
label_list = []
category_index = label_map_util.create_category_index_from_labelmap(os.getenv('LABELMAP'))
for label in category_index: label_list.append(category_index[label]['name'])

# Initialize global variables
score = 0
label_index_counter = 0

# GUI theme
sg.theme('DarkBlue')

# Return all valid video input devices
def getValidCameras():
    valid_sources = []
    for source in range(0, 8):
        cap = cv2.VideoCapture(source) 
        if cap is not None and cap.isOpened():
            valid_sources.append(source)
    if not valid_sources:
        valid_sources.append(0)
    return valid_sources

# Function to detect objects in a frame
@tf.function
def detect_objs(image):
    frame, shapes = obj_det_model.preprocess(image)
    predictions = obj_det_model.predict(frame, shapes)
    return obj_det_model.postprocess(predictions, shapes)

# Update game logic variables
def update_prompt():
    if settings['mode'] == 'Real Time Detection':
        prompt = random.choice(label_list)
    else:
        global label_index_counter
        prompt = label_list[label_index_counter]
        label_index_counter += 1
    return prompt

# Initialize valid input devices and default device
input_device_list = getValidCameras()
settings['input_device_id'] = input_device_list[0]

# Begin object detection session
def begin_detection():
    score = 0
    global label_index_counter
    label_index_counter = 0
    prompt = update_prompt()

    # Detection window GUI layout
    layout = [
        [sg.Text('Prompt: ' + prompt.upper(), font=('Book', 14), enable_events = True, key = "-PROMPT-"), sg.Text('Score: ' + str(score), font=('Book', 14), key = "-SCORE-")],
        [sg.Image(key = '-IMAGE-')]
    ]
    window = sg.Window('ASL Training', layout)

    if settings['mode'] == 'Demo Video':
        cap = cv2.VideoCapture('demo.mp4')
    else:
        cap = cv2.VideoCapture(settings['input_device_id'])

    while cap.isOpened():
        event,values = window.read(timeout = 0)
        if event == sg.WIN_CLOSED:
            break
        try:
            _, frame = cap.read()
            image_np = np.array(frame)
            
            # Convert frame to tensor object and get detections
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_objs(input_tensor)
        
            num_detections = int(detections.pop('num_detections'))

            image_np_with_detections = image_np.copy()
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            conf_thresh = settings['confidence_threshold'] / 100

            # Draw boxes using visulization_utils
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections, detections['detection_boxes'],
                detections['detection_classes'] + 1, detections['detection_scores'],
                category_index, use_normalized_coordinates=True, max_boxes_to_draw = settings['box_limit'],
                min_score_thresh = conf_thresh, agnostic_mode=False)

            imgbytes = cv2.imencode('.png',image_np_with_detections)[1].tobytes()
            window['-IMAGE-'].update(data = imgbytes)

            det_classes = detections['detection_classes'] + 1
            det_scores = detections['detection_scores']

            # Check for correctness and update score & prompt
            for index,value in enumerate(det_classes):
                if (det_scores[index] > conf_thresh) & (category_index[value]['name'] == prompt):
                    score = score + 1
                    prompt = update_prompt()
                    window['-PROMPT-'].update('Prompt: ' + prompt.upper())
                    window['-SCORE-'].update('Score: ' + str(score))
        except:
            window.close()

    window.close()

# Settings GUI layout
def open_settings():
    layout_settings = [
        [sg.Text('Settings', font=('Bookman Old Style', 40), pad=((0, 0), 16))],
        [sg.Text('Detection Mode: ', font=('Bookman Old Style', 14), pad=((0, 0), 10)), sg.Combo(['Real Time Detection', 'Demo Video'], default_value = settings['mode'], key = 'mode')],
        [sg.Text('Input Device: ', font=('Bookman Old Style', 14), pad=((0, 0), 10)), sg.Combo(input_device_list, default_value = settings['input_device_id'], key = 'input_device_id')],
        [sg.Text('Confidence Threshold: ', font=('Bookman Old Style', 14), pad=((0, 0), 16)), sg.Slider(range = (1, 100), default_value = settings['confidence_threshold'], orientation = 'horizontal', key = 'confidence_threshold')],
        [sg.Text('Max Boxes: ', font=('Bookman Old Style', 14)), sg.Slider(range = (1, 5), default_value = settings['box_limit'], orientation = 'horizontal', key = 'box_limit')],
        [sg.Button('Save', size=(10, 1), font=('Book', 14), button_color=('white', '#4E4E4E'), pad = ((0,0), 24)), sg.Button('Cancel', size=(10, 1), font=('Book', 14), button_color=('white', '#4E4E4E'))]
    ]
    window_settings = sg.Window('ASL Trainer Settings', layout_settings, size=(500, 400), resizable=False, finalize=True, element_justification='c')

    # Settings window event loop
    while True:
        event,values = window_settings.read(timeout = 0)
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Cancel':
            break
        elif event == 'Save':
            settings['mode'] = values['mode']
            settings['confidence_threshold'] = values['confidence_threshold']
            settings['box_limit'] = values['box_limit']
            settings['input_device_id'] = values['input_device_id']
            break
    
    window_settings.close()

# Main menu GUI layout
def main_menu():
    layout_main = [
        [sg.Text('ASL Virtual', font=('Bookman Old Style', 40), pad=((0, 0), 0))],
        [sg.Text('Trainer', font=('Bookman Old Style', 40), pad=((0, 0), 20))],
        [sg.Button('Begin', size=(20, 2), font=('Book', 14), button_color=('white', '#4E4E4E'), pad=((0, 0), 20))],
        [sg.Button('Settings', size=(20, 2), font=('Book', 14), button_color=('white', '#4E4E4E'), pad=((0, 0), 20))]
    ]

    window_main = sg.Window('ASL Virtual Trainer', layout_main, size=(500, 400), resizable=False, finalize=True, element_justification='c')

    # Main menu window event loop
    while True:
        event,values = window_main.read(timeout = 0)
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Begin':
            begin_detection()
        elif event == 'Settings':
            open_settings()

    window_main.close()

main_menu()