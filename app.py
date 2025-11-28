"""Flask web application for document processing and updates.

This module provides a DEBUG-FRIENDLY web interface for:
1. Step 1: Excel to Image - Upload Excel and see image rendering
2. Step 2: Vision Extraction - Upload image and see AI comprehension
3. Step 3: Full Workflow - Complete document processing
4. Download updated documents

API Endpoints:
    GET /: Main debug interface
    POST /step1_excel_to_image: Convert Excel to image
    POST /step2_vision_extract: Extract data from image
    POST /step3_full_workflow: Complete processing
    GET /download: Download updated document
    GET /download_image/<filename>: Download generated image
"""

import os
import logging
import tempfile
import json
import base64
from datetime import datetime
from flask import Flask, request, render_template, flash, send_file, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from utils.document_processing.vision_processor import process_files_with_vision, apply_updates_to_document
from utils.excel_to_image import convert_excel_to_image
from utils.vision_data_extractor import extract_data_from_excel_image
from utils.logging import setup_logging

app = Flask(__name__)
# Use a fixed secret key for development
app.secret_key = 'your-fixed-secret-key-here'  # Change this in production

# Set up logging
logger = setup_logging(level=logging.DEBUG)  # Changed to DEBUG for detailed logging

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
DEBUG_FOLDER = 'debug_output'
ALLOWED_EXTENSIONS = {'docx', 'xlsx', 'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DEBUG_FOLDER'] = DEBUG_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

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

@app.route('/', methods=['GET'])
def index():
    """Main debug interface page."""
    return render_template('debug_interface.html')

@app.route('/step1_excel_to_image', methods=['POST'])
def step1_excel_to_image():
    """Step 1: Convert Excel to Image and display it."""
    logger.info("="*80)
    logger.info("STEP 1: EXCEL TO IMAGE CONVERSION")
    logger.info("="*80)
    
    if 'excel_file' not in request.files:
        logger.error("No Excel file provided")
        return jsonify({'error': 'No Excel file provided'}), 400
    
    excel_file = request.files['excel_file']
    
    if excel_file.filename == '':
        logger.error("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(excel_file.filename):
        logger.error(f"Invalid file type: {excel_file.filename}")
        return jsonify({'error': 'Only .xlsx files allowed'}), 400
    
    try:
        # Save Excel file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_filename = f"{timestamp}_{secure_filename(excel_file.filename)}"
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
        excel_file.save(excel_path)
        logger.info(f"Excel file saved: {excel_path}")
        
        # Convert to image
        logger.info("Converting Excel to image...")
        img, img_base64 = convert_excel_to_image(excel_path)
        logger.info(f"Image created: {img.width}x{img.height} pixels")
        
        # Save image to debug folder
        image_filename = f"{timestamp}_excel_render.png"
        image_path = os.path.join(app.config['DEBUG_FOLDER'], image_filename)
        img.save(image_path)
        logger.info(f"Image saved: {image_path}")
        
        # Store metadata in session (not the large base64 string)
        # The base64 will be read from file when needed
        session['excel_image_filename'] = image_filename
        session['excel_path'] = excel_path
        session.modified = True
        
        logger.info("Step 1 complete - Excel converted to image")
        
        return jsonify({
            'success': True,
            'message': 'Excel converted to image successfully',
            'image_url': f'/download_image/{image_filename}',
            'image_width': img.width,
            'image_height': img.height,
            'image_size_kb': len(img_base64) // 1024
        })
        
    except Exception as e:
        logger.error(f"Error in Step 1: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/step2_vision_extract', methods=['POST'])
def step2_vision_extract():
    """Step 2: Extract data from image using vision AI."""
    logger.info("="*80)
    logger.info("STEP 2: VISION DATA EXTRACTION")
    logger.info("="*80)
    
    # Check if we have an image from Step 1 or if user uploaded new image
    image_base64 = None
    
    if 'image_file' in request.files and request.files['image_file'].filename != '':
        # User uploaded a new image
        logger.info("Using uploaded image file")
        image_file = request.files['image_file']
        
        if not allowed_file(image_file.filename):
            logger.error(f"Invalid file type: {image_file.filename}")
            return jsonify({'error': 'Only .png, .jpg, .jpeg files allowed'}), 400
        
        try:
            # Read and encode image
            image_data = image_file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"Image uploaded: {len(image_base64)} bytes (base64)")
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return jsonify({'error': f'Error reading image: {str(e)}'}), 500
    
    elif 'excel_image_filename' in session:
        # Use image from Step 1 - read from file
        logger.info("Using image from Step 1")
        image_filename = session['excel_image_filename']
        image_path = os.path.join(app.config['DEBUG_FOLDER'], image_filename)
        
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
            logger.info(f"Loaded image from file: {len(image_base64)} bytes (base64)")
        else:
            logger.error(f"Image file not found: {image_path}")
            return jsonify({'error': 'Image file not found. Please run Step 1 again.'}), 400
    
    else:
        logger.error("No image available - run Step 1 first or upload an image")
        return jsonify({'error': 'No image available. Please run Step 1 first or upload an image.'}), 400
    
    try:
        # Extract data using vision AI
        logger.info("Extracting data from image using vision AI...")
        extracted_data = extract_data_from_excel_image(image_base64, timeout=60)
        
        logger.info(f"Extraction complete - found {len(extracted_data.get('metrics', {}))} metrics")
        logger.debug(f"Extracted data: {json.dumps(extracted_data, indent=2)}")
        
        # Store in session for Step 3
        session['extracted_data'] = extracted_data
        session.modified = True
        
        # Format for display
        metrics_summary = []
        for metric_name, values in extracted_data.get('metrics', {}).items():
            metrics_summary.append({
                'name': metric_name,
                'value_count': len(values),
                'values': values
            })
        
        logger.info("Step 2 complete - Data extracted from image")
        
        return jsonify({
            'success': True,
            'message': 'Data extracted successfully',
            'metrics_count': len(extracted_data.get('metrics', {})),
            'metrics': metrics_summary,
            'time_periods': extracted_data.get('time_periods', []),
            'metadata': extracted_data.get('metadata', {}),
            'raw_data': extracted_data
        })
        
    except Exception as e:
        logger.error(f"Error in Step 2: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/step3_full_workflow', methods=['POST'])
def step3_full_workflow():
    """Step 3: Complete workflow - process Word document with extracted data."""
    logger.info("="*80)
    logger.info("STEP 3: FULL WORKFLOW")
    logger.info("="*80)
    
    # Check if both files are present
    if 'docx_file' not in request.files or 'excel_file' not in request.files:
        logger.error("Both files required")
        return jsonify({'error': 'Both Word and Excel files are required'}), 400
    
    docx_file = request.files['docx_file']
    excel_file = request.files['excel_file']
    
    if docx_file.filename == '' or excel_file.filename == '':
        logger.error("Empty filenames")
        return jsonify({'error': 'No files selected'}), 400
    
    if not (allowed_file(docx_file.filename) and allowed_file(excel_file.filename)):
        logger.error("Invalid file types")
        return jsonify({'error': 'Only .docx and .xlsx files allowed'}), 400
    
    try:
        # Save files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        docx_filename = f"{timestamp}_{secure_filename(docx_file.filename)}"
        excel_filename = f"{timestamp}_{secure_filename(excel_file.filename)}"
        
        docx_path = os.path.join(app.config['UPLOAD_FOLDER'], docx_filename)
        excel_path = os.path.join(app.config['UPLOAD_FOLDER'], excel_filename)
        
        docx_file.save(docx_path)
        excel_file.save(excel_path)
        logger.info(f"Files saved: {docx_path}, {excel_path}")
        
        # Process files using vision AI
        logger.info("Starting full vision-based processing...")
        results = process_files_with_vision(docx_path, excel_path)
        logger.info(f"Processing complete - found {len(results)} updates")
        
        # Create updated document
        if results and isinstance(results, list):
            temp_dir = tempfile.mkdtemp()
            updated_filename = f'updated_{secure_filename(docx_file.filename)}'
            updated_path = os.path.join(temp_dir, updated_filename)
            
            logger.info(f"Applying {len(results)} updates to document...")
            changes_made = apply_updates_to_document(docx_path, updated_path, results, min_confidence=0.7)
            logger.info(f"Applied {changes_made} changes to document")
            
            # Store paths in session
            session['document_info'] = {
                'original_path': docx_path,
                'updated_path': updated_path,
                'filename': updated_filename
            }
            session.modified = True
        
        # Clean up excel file
        try:
            os.remove(excel_path)
        except:
            pass
        
        # Format results for response
        formatted_results = []
        if isinstance(results, list):
            for result in results:
                if isinstance(result, tuple) and len(result) == 4:
                    formatted_results.append({
                        'original': result[0],
                        'modified': result[1],
                        'confidence': result[2],
                        'description': result[3]
                    })
                elif isinstance(result, tuple) and len(result) == 3:
                    formatted_results.append({
                        'original': result[0],
                        'modified': result[1],
                        'confidence': result[2],
                        'description': 'N/A'
                    })
        
        logger.info("Step 3 complete - Full workflow finished")
        
        return jsonify({
            'success': True,
            'message': f'Processing complete - {len(formatted_results)} updates made',
            'results_count': len(formatted_results),
            'results': formatted_results,
            'download_available': True
        })
        
    except Exception as e:
        logger.error(f"Error in Step 3: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/download_image/<filename>')
def download_image(filename):
    """Download a generated image from debug folder."""
    try:
        image_path = os.path.join(app.config['DEBUG_FOLDER'], filename)
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return "Image not found", 404
        
        logger.info(f"Serving image: {image_path}")
        return send_file(image_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error serving image: {str(e)}")
        return str(e), 500

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