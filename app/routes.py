from flask import Blueprint, render_template, request, jsonify
from app.resume_evaluator import ResumeEvaluator
from app.utils import format_content, extract_text_from_pdf
import os
import tempfile

main = Blueprint('main', __name__)
resume_evaluator = ResumeEvaluator()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part'
        }), 400
        
    file = request.files['resume']
    target_role = request.form.get('target_role', '')
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No selected file'
        }), 400
        
    # Handle PDF upload
    try:
        # Save to temporary file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)
        
        # Extract text from PDF
        resume_text = extract_text_from_pdf(file_path)
        
        # Clean up temp file
        os.remove(file_path)
        os.rmdir(temp_dir)
        
        # Evaluate resume
        evaluation = resume_evaluator.evaluate_resume(resume_text, target_role)
        formatted_evaluation = format_content(evaluation)
        
        return jsonify({
            'status': 'success',
            'evaluation': formatted_evaluation
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500