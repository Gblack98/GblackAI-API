import sys
import os

# Add root directory to PYTHONPATH for app/ imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
