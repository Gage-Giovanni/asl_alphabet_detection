import os
from dotenv import load_dotenv

load_dotenv()

# Verify Installation
VERIFICATION_SCRIPT = os.path.join(str(os.getenv('TF_MODELS')), 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')
os.system('python ' + VERIFICATION_SCRIPT)