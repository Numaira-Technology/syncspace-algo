"""Flask web application for document processing and updates.

This module provides a web interface for:
1. Uploading Word documents and Excel files
2. Processing documents to find and update financial metrics
3. Downloading updated documents with confidence scores

API Endpoints:
    POST /upload:
        Upload Word and Excel files for processing
        Files: docx_file, excel_file (multipart/form-data)
        Returns: HTML template with processing results
        
    GET /download:
        Download the updated document
        Returns: Modified .docx file with confidence scores
        
Data Structures:
    Document Info (session):
        {
            'original_path': str,  # Path to original document
            'updated_path': str,   # Path to modified document
            'filename': str        # Output filename
        }
        
    Results Format:
        [
            {
                'original': str,   # Original sentence
                'modified': str,   # Updated sentence
                'confidence': float # Update confidence (0.0-1.0)
            }
        ]
"""

import os
import logging
import tempfile
from flask import Flask, request, render_template, flash, send_file, session, redirect, url_for
from werkzeug.utils import secure_filename
from main import process_files, update_document
from utils.logging import setup_logging

app = Flask(__name__)
# Use a fixed secret key for development
app.secret_key = 'your-fixed-secret-key-here'  # Change this in production

# Set up logging
logger = setup_logging(level=logging.DEBUG)  # Changed to DEBUG for detailed logging

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'xlsx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def cleanup_files(document_info):
    """Clean up temporary files after successful download"""
    updated_path = document_info.get('updated_path')
    original_path = document_info.get('original_path')
    
    if updated_path and os.path.exists(updated_path):
        try:
            os.remove(updated_path)
            os.rmdir(os.path.dirname(updated_path))
        except Exception as e:
            logger.error(f"Error cleaning up updated file: {str(e)}")
    
    if original_path and os.path.exists(original_path):
        try:
            os.remove(original_path)
        except Exception as e:
            logger.error(f"Error cleaning up original file: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if both files are present
        if 'docx_file' not in request.files or 'excel_file' not in request.files:
            flash('Both files are required')
            return render_template('index.html')
        
        docx_file = request.files['docx_file']
        excel_file = request.files['excel_file']
        
        # If user does not select files
        if docx_file.filename == '' or excel_file.filename == '':
            flash('No selected files')
            return render_template('index.html')
        
        if (docx_file and allowed_file(docx_file.filename) and 
            excel_file and allowed_file(excel_file.filename)):
            
            try:
                # Save files
                docx_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(docx_file.filename))
                excel_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(excel_file.filename))
                
                docx_file.save(docx_path)
                excel_file.save(excel_path)
                
                # Process files and get results
                results = process_files(docx_path, excel_path)
                
                # Create updated document
                if results and isinstance(results, list):
                    # Create a temporary file for the updated document
                    temp_dir = tempfile.mkdtemp()
                    updated_filename = 'updated_' + secure_filename(docx_file.filename)
                    updated_path = os.path.join(temp_dir, updated_filename)
                    
                    # Update the document with modifications
                    update_document(docx_path, results, updated_path)
                    
                    # Store paths in session
                    session['document_info'] = {
                        'original_path': docx_path,
                        'updated_path': updated_path,
                        'filename': updated_filename
                    }
                    session.modified = True  # Ensure session is saved
                
                # Clean up excel file
                try:
                    os.remove(excel_path)
                except:
                    pass
                
                # Format results for template
                formatted_results = []
                if isinstance(results, list):
                    for result in results:
                        if isinstance(result, tuple) and len(result) == 3:
                            formatted_results.append({
                                'original': result[0],
                                'modified': result[1],
                                'confidence': result[2]
                            })
                        else:
                            formatted_results.append({
                                'original': result,
                                'modified': result
                            })
                
                return render_template('index.html', results=formatted_results)
            except Exception as e:
                logger.error(f"Error processing files: {str(e)}", exc_info=True)
                flash(f'Error processing files: {str(e)}')
                return render_template('index.html')
        else:
            flash('Allowed file types are .docx and .xlsx')
            return render_template('index.html')
    
    return render_template('index.html')

@app.route('/download')
def download_docx():
    document_info = session.get('document_info')
    if not document_info:
        logger.warning("No document info in session")
        flash('No updated document available for download. Please process a document first.')
        return redirect(url_for('upload_file'))
    
    updated_path = document_info.get('updated_path')
    filename = document_info.get('filename')
    
    if updated_path and os.path.exists(updated_path):
        try:
            response = send_file(
                updated_path,
                mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                as_attachment=True,
                download_name=filename,
                max_age=0
            )
            
            # Clean up files after successful download
            @response.call_on_close
            def cleanup():
                cleanup_files(document_info)
                session.pop('document_info', None)
                session.modified = True
            
            return response
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}", exc_info=True)
            flash(f'Error downloading file: {str(e)}')
            return redirect(url_for('upload_file'))
    else:
        logger.warning(f"File not found at path: {updated_path}")
        flash('No updated document available for download. Please process a document first.')
        return redirect(url_for('upload_file'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)               