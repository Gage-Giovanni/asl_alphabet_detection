import os
import random

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

import cv2
import numpy as np
from matplotlib import pyplot as plt

import PySimpleGUI as sg

settings = {'mode': 'Real Time Detection', 'confidence_threshold': 80, 'box_limit': 2}

CUSTOM_MODEL_NAME = 'new_model'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

label_list = []
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
for label in category_index: label_list.append(category_index[label]['name'])

# GUI theme
sg.theme('DarkBlue')

def begin_detection():
    prompt = random.choice(label_list)
    score = 0

    layout = [
        [sg.Text('Prompt: ' + prompt.upper(), font=('Book', 14), enable_events = True, key = "-PROMPT-"), sg.Text('Score: ' + str(score), font=('Book', 14), key = "-SCORE-")],
        [sg.Image(key = '-IMAGE-')]
    ]
    window = sg.Window('ASL Training', layout)

    if settings['mode'] == 'Demo Video':
        cap = cv2.VideoCapture('abc.mp4')
    else:
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        event,values = window.read(timeout = 0)
        if event == sg.WIN_CLOSED:
            break
        try:
            _, frame = cap.read()
            image_np = np.array(frame)
            
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)
        
            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            conf_thresh = settings['confidence_threshold'] / 100

            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw = settings['box_limit'],
                        min_score_thresh = conf_thresh,
                        agnostic_mode=False)

            imgbytes = cv2.imencode('.png',image_np_with_detections)[1].tobytes()
            window['-IMAGE-'].update(data = imgbytes)

            my_classes = detections['detection_classes'] + label_id_offset
            my_scores = detections['detection_scores']

            for index,value in enumerate(my_classes):
                if (my_scores[index] > conf_thresh) & (category_index[value]['name'] == prompt):
                    score = score + 1
                    prompt = random.choice(label_list)
                    window['-PROMPT-'].update('Prompt: ' + prompt.upper())
                    window['-SCORE-'].update('Score: ' + str(score))
        except:
            window.close()

    window.close()

def open_settings():
    layout_settings = [
        [sg.Text('Settings', font=('Bookman Old Style', 40), pad=((0, 0), 20))],
        [sg.Text('Detection Mode: ', font=('Bookman Old Style', 14), pad=((0, 0), 20)), sg.Combo(['Real Time Detection', 'Demo Video'], default_value = settings['mode'], key = 'mode')],
        [sg.Text('Confidence Threshold: ', font=('Bookman Old Style', 14), pad=((0, 0), 20)), sg.Slider(range = (1, 100), default_value = settings['confidence_threshold'], orientation = 'horizontal', key = 'confidence_threshold')],
        [sg.Text('Max Boxes: ', font=('Bookman Old Style', 14)), sg.Slider(range = (1, 5), default_value = settings['box_limit'], orientation = 'horizontal', key = 'box_limit')],
        [sg.Button('Save', size=(10, 1), font=('Book', 14), button_color=('white', '#4E4E4E'), pad = ((0,0), 30)), sg.Button('Cancel', size=(10, 1), font=('Book', 14), button_color=('white', '#4E4E4E'))]
    ]
    window_settings = sg.Window('ASL Trainer Settings', layout_settings, size=(500, 400), resizable=False, finalize=True, element_justification='c')

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
            break
    
    window_settings.close()


def main_menu():
    layout_main = [
        [sg.Text('ASL Virtual', font=('Bookman Old Style', 40), pad=((0, 0), 0))],
        [sg.Text('Trainer', font=('Bookman Old Style', 40), pad=((0, 0), 20))],
        [sg.Button('Begin', size=(20, 2), font=('Book', 14), button_color=('white', '#4E4E4E'), pad=((0, 0), 20))],
        [sg.Button('Settings', size=(20, 2), font=('Book', 14), button_color=('white', '#4E4E4E'), pad=((0, 0), 20))]
    ]

    window_main = sg.Window('ASL Virtual Trainer', layout_main, size=(500, 400), resizable=False, finalize=True, element_justification='c')

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