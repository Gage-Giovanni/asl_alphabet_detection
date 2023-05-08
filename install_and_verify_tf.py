import os

# Install Tensorflow
os.system('cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install')
os.system('cd Tensorflow/models/research/slim && pip install -e . ')

# Verify Installation
VERIFICATION_SCRIPT = os.path.join(str(os.getenv('TF_MODELS')), 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
os.system(VERIFICATION_SCRIPT)