from flask import Flask
from app.routes import main
import os

# Get the absolute path to the template folder
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static'))

# Create the Flask app with specified template and static folder
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.register_blueprint(main)

# Import other components if needed
from app import resume_evaluator, utils