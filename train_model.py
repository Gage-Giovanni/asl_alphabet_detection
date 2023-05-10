import os
from dotenv import load_dotenv
import object_detection
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
   
load_dotenv()

NEW_MODEL_DIR = os.path.join(os.getenv('CUSTOM_MODELS'), os.getenv('NEW_MODEL'))
PIPELINE = os.path.join(NEW_MODEL_DIR, 'pipeline.config')
PT_MODEL = os.path.join('Tensorflow\\workspace\\pre-trained-models', 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8')

LABELS = [{'name':'a', 'id':1}, {'name':'b', 'id':2}, {'name':'c', 'id':3}, {'name':'d', 'id':4}, {'name':'e', 'id':5},
          {'name':'f', 'id':6}, {'name':'g', 'id':7}, {'name':'h', 'id':8}, {'name':'i', 'id':9}, {'name':'j', 'id':10},
          {'name':'k', 'id':11}, {'name':'l', 'id':12}, {'name':'m', 'id':13}, {'name':'n', 'id':14}, {'name':'o', 'id':15},
          {'name':'p', 'id':16}, {'name':'q', 'id':17}, {'name':'r', 'id':18}, {'name':'s', 'id':19}, {'name':'t', 'id':20},
          {'name':'u', 'id':21}, {'name':'v', 'id':22}, {'name':'w', 'id':23}, {'name':'x', 'id':24}, {'name':'y', 'id':25},
          {'name':'z', 'id':26}]

with open(os.getenv('LABELMAP'), 'w') as f:
    for label in LABELS:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# Create new model directories
if not os.path.exists(NEW_MODEL_DIR):
    os.mkdir(NEW_MODEL_DIR)
    os.mkdir(os.path.join(NEW_MODEL_DIR, 'export'))
    os.mkdir(os.path.join(NEW_MODEL_DIR, 'tfjsexport'))
    os.mkdir(os.path.join(NEW_MODEL_DIR, 'tfliteexport'))

# Create tf records
os.system('python ' + os.getenv('TF_RECORD_SCRIPT') + ' -x ' + os.path.join(os.getenv('IMAGES'), 'train') + ' -l ' + os.getenv('LABELMAP') + ' -o ' + os.path.join(os.getenv('ANNOTATIONS'), 'train.record'))
os.system('python ' + os.getenv('TF_RECORD_SCRIPT') + ' -x ' + os.path.join(os.getenv('IMAGES'), 'test') + ' -l ' + os.getenv('LABELMAP') + ' -o ' + os.path.join(os.getenv('ANNOTATIONS'), 'test.record'))


# Copy over pipeline.config file and update for training
os.system('copy ' + os.path.join(PT_MODEL, 'pipeline.config') + ' ' + NEW_MODEL_DIR)

config = config_util.get_configs_from_pipeline_file(PIPELINE)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(PIPELINE, "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)

if pipeline_config:
    pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
    pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PT_MODEL, 'checkpoint', 'ckpt-0')
    pipeline_config.model.ssd.num_classes = len(LABELS)
    pipeline_config.train_input_reader.label_map_path= os.getenv('LABELMAP')
    pipeline_config.eval_input_reader[0].label_map_path = os.getenv('LABELMAP')
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(os.getenv('ANNOTATIONS'), 'train.record')]
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(os.getenv('ANNOTATIONS'), 'test.record')]
    pipeline_config.train_config.batch_size = 4

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(PIPELINE, "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)

# Print out the training command
TRAINING_SCRIPT = os.path.join(os.getenv('TF_MODELS'), 'research', 'object_detection', 'model_main_tf2.py')
command = 'python ' + TRAINING_SCRIPT + ' --model_dir=' + NEW_MODEL_DIR + ' --pipeline_config_path=' + PIPELINE + ' --num_train_steps=20000'
print(command)