# Create new file
from flask import Flask, jsonify, send_file, abort
from flask.json.provider import JSONProvider
from flask_cors import CORS
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

from src.core.database import init_database
from src.storage.operations import DatabaseOperations, serialize_datetime
from src.storage.models import PromptTypeModel, PromptModel

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize database operations lazily
db_ops = None

# Configure Flask's JSON provider to handle datetime
class CustomJSONProvider(JSONProvider):
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, default=self._default)
    
    def loads(self, s, **kwargs):
        return json.loads(s)
    
    def _default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

# Register the custom JSON provider
app.json = CustomJSONProvider(app)

def get_db():
    """Get or initialize database operations."""
    global db_ops
    if db_ops is None:
        db_ops = DatabaseOperations()
    return db_ops

def ensure_prompt_types():
    """Ensure necessary prompt types exist in the database."""
    try:
        with get_db().db.atomic():
            # Create default prompt types if they don't exist
            default_types = ['user_interaction', 'system', 'assistant']
            prompt_types = {}
            for type_name in default_types:
                prompt_type, created = PromptTypeModel.get_or_create(prompt_type_name=type_name)
                prompt_types[type_name] = prompt_type
                if created:
                    app.logger.info(f"Created prompt type: {type_name}")
            
            # Add some test prompts if none exist
            if PromptModel.select().count() == 0:
                app.logger.info("No prompts found, creating test prompts...")
                test_prompts = [
                    {
                        'type': 'user_interaction',
                        'text': 'Please help me debug this code',
                        'context': json.dumps({'application_name': 'VS Code', 'window_title': 'debug.py'}),
                        'model_name': 'gpt-4',
                        'quality_score': 0.95
                    },
                    {
                        'type': 'assistant',
                        'text': 'I can help you debug the code. What seems to be the issue?',
                        'context': json.dumps({'application_name': 'VS Code', 'window_title': 'debug.py'}),
                        'model_name': 'gpt-4',
                        'quality_score': 0.92
                    },
                    {
                        'type': 'system',
                        'text': 'Processing code analysis request',
                        'context': json.dumps({'application_name': 'VS Code', 'window_title': 'debug.py'}),
                        'model_name': 'gpt-4',
                        'quality_score': 1.0
                    }
                ]
                
                for prompt in test_prompts:
                    PromptModel.create(
                        prompt_text=prompt['text'],
                        prompt_type=prompt_types[prompt['type']],
                        model_name=prompt['model_name'],
                        context=prompt['context'],
                        quality_score=prompt['quality_score'],
                        timestamp=datetime.now()
                    )
                app.logger.info("Created test prompts successfully")
                
    except Exception as e:
        app.logger.error(f"Error ensuring prompt types and test data: {e}")

def normalize_path(path):
    """Normalize a file path to work with both Windows and Unix systems."""
    # Convert URL-encoded path to normal path
    path = path.replace('%5C', os.path.sep).replace('\\', os.path.sep)
    # Convert forward slashes to system separator
    path = path.replace('/', os.path.sep)
    # Resolve the path relative to the data directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    full_path = os.path.normpath(os.path.join(base_dir, path))
    # Ensure the path is within the allowed directory
    if not full_path.startswith(base_dir):
        abort(403)  # Forbidden
    return full_path

@app.route('/api/screenshots', methods=['GET'])
def get_screenshots():
    """Get all screenshots from the last 30 days."""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        screenshots = get_db().get_screenshots_in_range(start_time, end_time)
        return jsonify(screenshots)
    except Exception as e:
        app.logger.error(f"Error in get_screenshots: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/screenshot/<path:image_path>')
def serve_screenshot(image_path):
    """Serve a screenshot image."""
    try:
        full_path = normalize_path(image_path)
        if not os.path.exists(full_path):
            app.logger.error(f"Screenshot not found: {full_path}")
            return f"Screenshot not found: {image_path}", 404
        return send_file(full_path, mimetype='image/jpeg')
    except Exception as e:
        app.logger.error(f"Error serving screenshot {image_path}: {e}")
        return str(e), 500

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Get all prompts."""
    try:
        # Get prompts of all types, ordered by timestamp
        prompts = []
        for prompt_type in ['user_interaction', 'system', 'assistant']:
            app.logger.debug(f"Fetching prompts of type: {prompt_type}")
            type_prompts = get_db().get_prompts_by_type(prompt_type)
            app.logger.debug(f"Found {len(type_prompts)} prompts of type {prompt_type}")
            prompts.extend(type_prompts)
        
        # Sort all prompts by timestamp in descending order
        prompts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Take the most recent 100 prompts
        prompts = prompts[:100]
        
        app.logger.info(f"Returning {len(prompts)} total prompts")
        return jsonify(prompts)
    except Exception as e:
        app.logger.error(f"Error in get_prompts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/code-insights', methods=['GET'])
def get_code_insights():
    """Get screenshots with code insights."""
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(days=30)
        screenshots = get_db().get_screenshots_in_range(start_time, end_time)
        
        # Filter screenshots that have code insights
        code_insights = [
            {
                'timestamp': s['timestamp'],
                'insights': s.get('code_insights', None),
                'context': s['context']
            }
            for s in screenshots
            if s.get('code_insights')
        ]
        
        return jsonify(code_insights)
    except Exception as e:
        app.logger.error(f"Error in get_code_insights: {e}")
        return jsonify({"error": str(e)}), 500

def start_server(host='localhost', port=5667):
    """Start the Flask server."""
    # Initialize database before starting server
    init_database(testing=True)  # Use testing mode to match AW server
    # Ensure prompt types exist
    ensure_prompt_types()
    app.run(host=host, port=port)

if __name__ == '__main__':
    start_server() 