"""Cursor chat tracking functionality."""

import logging
import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.storage.operations import DatabaseOperations
from src.storage.models import CursorProjectModel

logger = logging.getLogger(__name__)

class CursorTracker:
    """Tracks and processes Cursor chat data."""
    
    def __init__(self, db_ops: DatabaseOperations):
        """Initialize cursor tracker.
        
        Args:
            db_ops: Database operations instance
        """
        self.db_ops = db_ops
        logger.info("Initialized cursor tracker")
        
    def get_cursor_workspace_path(self) -> Optional[Path]:
        """Get the path to the Cursor workspace storage directory.
        
        Returns:
            Optional[Path]: Path to the Cursor workspace storage directory or None if not found
        """
        try:
            # Get the path based on the operating system
            if os.name == 'nt':  # Windows
                base_path = Path(os.path.expandvars('%APPDATA%')) / 'Cursor' / 'User' / 'workspaceStorage'
            elif os.name == 'posix':  # macOS/Linux
                if os.path.exists(os.path.expanduser('~/Library')):  # macOS
                    base_path = Path(os.path.expanduser('~/Library/Application Support/Cursor/User/workspaceStorage'))
                else:  # Linux
                    base_path = Path(os.path.expanduser('~/.config/Cursor/User/workspaceStorage'))
            else:
                logger.error(f"Unsupported operating system: {os.name}")
                return None
                
            if not base_path.exists():
                logger.error(f"Cursor workspace storage directory not found: {base_path}")
                return None
                
            logger.info(f"Cursor workspace storage directory found: {base_path}")
            return base_path
            
        except Exception as e:
            logger.error(f"Error getting Cursor workspace path: {e}")
            return None
            
    def get_latest_workspace_db_path(self) -> Optional[str]:
        """Get the path to the latest Cursor workspace database.
        
        Returns:
            Optional[str]: Path to the latest Cursor workspace database or None if not found
        """
        try:
            # Get the base path
            base_path = self.get_cursor_workspace_path()
            if not base_path:
                logger.error("Cursor workspace path not found")
                return None
                
            logger.info(f"Searching for workspace folders in: {base_path}")
                
            # Get all workspace folders
            workspace_folders = list(base_path.glob("*"))
            if not workspace_folders:
                logger.error("No workspace folders found")
                return None
                
            # Log the number of workspace folders found
            logger.info(f"Found {len(workspace_folders)} workspace folders")
            
            # Get the most recently modified workspace folder
            try:
                workspace_folder = max(workspace_folders, key=os.path.getmtime)
                logger.info(f"Latest workspace folder: {workspace_folder}")
            except Exception as e:
                logger.error(f"Error finding latest workspace folder: {e}")
                # Try an alternative approach - just use the first folder
                if workspace_folders:
                    workspace_folder = workspace_folders[0]
                    logger.info(f"Using first workspace folder as fallback: {workspace_folder}")
                else:
                    return None
            
            # Check if the database file exists
            db_path = workspace_folder / "state.vscdb"
            if not db_path.exists():
                logger.error(f"state.vscdb not found in {workspace_folder}")
                
                # Try to find any state.vscdb file in any workspace folder
                all_dbs = list(base_path.glob("*/state.vscdb"))
                if all_dbs:
                    db_path = all_dbs[0]
                    logger.info(f"Using alternative database file: {db_path}")
                else:
                    return None
                
            logger.info(f"Latest workspace database found: {db_path}")
            return str(db_path)
            
        except Exception as e:
            logger.error(f"Error getting latest workspace database path: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
            
    def query_cursor_db(self, db_path: str, query: str) -> List[Any]:
        """Execute a SQL query on the Cursor database.
        
        Args:
            db_path: Path to the Cursor database
            query: SQL query to execute
            
        Returns:
            List[Any]: Query results
        """
        try:
            # Check if the database file exists
            if not os.path.exists(db_path):
                logger.error(f"Database file does not exist: {db_path}")
                return []
                
            # Log the query
            logger.info(f"Executing query on {db_path}: {query}")
            
            # Connect to the database in read-only mode
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            
            # Execute the query
            cursor.execute(query)
            
            # Fetch all rows
            rows = cursor.fetchall()
            
            # Close the connection
            conn.close()
            
            # Log the results
            logger.info(f"Query executed successfully, fetched {len(rows)} rows")
            
            return rows
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Database path: {db_path}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Database path: {db_path}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
            
    def query_specific_chat_data(self, db_path: str) -> List[Any]:
        """Query specific chat data from the Cursor database.
        
        This method queries the 'workbench.panel.aichat.view.aichat.chatdata' key
        which contains the structured chat data with tabs and messages.
        
        Args:
            db_path: Path to the Cursor database
            
        Returns:
            List[Any]: Chat data results
        """
        query = "SELECT value FROM ItemTable WHERE [key] = 'workbench.panel.aichat.view.aichat.chatdata'"
        rows = self.query_cursor_db(db_path, query)
        
        if not rows or not rows[0] or len(rows[0]) < 1:
            logger.warning("No chat data found in the database")
            return []
            
        return rows
            
    def get_cursor_chat_data(self, db_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Get prompts and generations from the Cursor database.
        
        Args:
            db_path: Path to the Cursor database
            
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Prompts and generations
        """
        try:
            # Query prompts
            logger.info(f"Querying prompts from Cursor database: {db_path}")
            prompt_rows = self.query_cursor_db(
                db_path, 
                "SELECT [key], value FROM ItemTable WHERE [key] = 'aiService.prompts'"
            )
            
            # Query generations
            logger.info(f"Querying generations from Cursor database: {db_path}")
            generation_rows = self.query_cursor_db(
                db_path, 
                "SELECT [key], value FROM ItemTable WHERE [key] = 'aiService.generations'"
            )
            
            # Also query chat data which might contain both prompts and responses
            logger.info(f"Querying chat data from Cursor database: {db_path}")
            chat_rows = self.query_specific_chat_data(db_path)
            
            # Log raw results for debugging
            logger.info(f"Raw prompt rows: {len(prompt_rows)}")
            logger.info(f"Raw generation rows: {len(generation_rows)}")
            logger.info(f"Raw chat rows: {len(chat_rows)}")
            
            # Parse prompts
            prompts = []
            if prompt_rows and prompt_rows[0] and len(prompt_rows[0]) > 1:
                try:
                    logger.info("Parsing prompts JSON")
                    prompts_data = json.loads(prompt_rows[0][1])
                    
                    # Handle different possible formats
                    if isinstance(prompts_data, list):
                        prompts = prompts_data
                        logger.info(f"Successfully parsed {len(prompts)} prompts (list format)")
                    elif isinstance(prompts_data, dict):
                        # If it's a dictionary, check for common keys that might contain the prompts
                        for key in ['prompts', 'items', 'data', 'results']:
                            if key in prompts_data and isinstance(prompts_data[key], list):
                                prompts = prompts_data[key]
                                logger.info(f"Successfully parsed {len(prompts)} prompts (dict format, key={key})")
                                break
                        
                        # If we still don't have prompts, try to convert the dict to a list
                        if not prompts:
                            # Try to convert the dict to a list of items
                            prompts = [{"id": k, **v} for k, v in prompts_data.items() if isinstance(v, dict)]
                            logger.info(f"Successfully parsed {len(prompts)} prompts (dict->list conversion)")
                    else:
                        logger.warning(f"Prompts data is not a list or dict: {type(prompts_data)}")
                        
                    # If we have prompts but they don't have IDs, add timestamps to them
                    # This is for the format we're seeing in the logs
                    if prompts and 'text' in prompts[0] and 'commandType' in prompts[0] and 'timestamp' not in prompts[0]:
                        logger.info("Adding timestamps to prompts")
                        for i, prompt in enumerate(prompts):
                            # Add a timestamp based on the index (newer prompts have higher indices)
                            # Use a base timestamp of now minus one day per prompt
                            base_timestamp = datetime.now().timestamp() * 1000  # Convert to milliseconds
                            prompt['timestamp'] = base_timestamp - (len(prompts) - i) * 60 * 1000  # One minute per prompt
                            logger.debug(f"Added timestamp to prompt: {prompt['timestamp']}")
                            
                            # Extract file paths from the prompt text if possible
                            if 'text' in prompt and not prompt.get('files'):
                                text = prompt.get('text', '')
                                import re
                                file_paths = re.findall(r'[\'"]([^\'"]*.(?:py|js|ts|html|css|md|txt|json|yaml|yml))[\'"]', text)
                                if file_paths:
                                    prompt['files'] = [{'path': path} for path in file_paths]
                                    logger.info(f"Extracted {len(prompt['files'])} file paths from prompt text")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse prompts JSON: {e}")
                    # Try to log a sample of the data for debugging
                    if prompt_rows[0][1]:
                        sample = prompt_rows[0][1][:100] + "..." if len(prompt_rows[0][1]) > 100 else prompt_rows[0][1]
                        logger.error(f"Sample of prompts data: {sample}")
            else:
                logger.warning("No prompt rows found or invalid format")
            
            # Parse generations
            generations = []
            if generation_rows and generation_rows[0] and len(generation_rows[0]) > 1:
                try:
                    logger.info("Parsing generations JSON")
                    generations_data = json.loads(generation_rows[0][1])
                    
                    # Handle different possible formats
                    if isinstance(generations_data, list):
                        generations = generations_data
                        logger.info(f"Successfully parsed {len(generations)} generations (list format)")
                    elif isinstance(generations_data, dict):
                        # If it's a dictionary, check for common keys that might contain the generations
                        for key in ['generations', 'items', 'data', 'results']:
                            if key in generations_data and isinstance(generations_data[key], list):
                                generations = generations_data[key]
                                logger.info(f"Successfully parsed {len(generations)} generations (dict format, key={key})")
                                break
                        
                        # If we still don't have generations, try to convert the dict to a list
                        if not generations:
                            # Try to convert the dict to a list of items
                            generations = [{"id": k, **v} for k, v in generations_data.items() if isinstance(v, dict)]
                            logger.info(f"Successfully parsed {len(generations)} generations (dict->list conversion)")
                    else:
                        logger.warning(f"Generations data is not a list or dict: {type(generations_data)}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse generations JSON: {e}")
                    # Try to log a sample of the data for debugging
                    if generation_rows[0][1]:
                        sample = generation_rows[0][1][:100] + "..." if len(generation_rows[0][1]) > 100 else generation_rows[0][1]
                        logger.error(f"Sample of generations data: {sample}")
            else:
                logger.warning("No generation rows found or invalid format")
                
            # Parse chat data to extract additional prompts and responses
            if chat_rows and len(chat_rows) > 0:
                try:
                    logger.info("Parsing chat data JSON")
                    # The chat data is in the first column of the first row
                    chat_data_json = chat_rows[0][0] if len(chat_rows[0]) > 0 else None
                    if chat_data_json:
                        chat_data = json.loads(chat_data_json)
                        
                        # Extract tabs from chat data
                        if isinstance(chat_data, dict) and 'tabs' in chat_data and isinstance(chat_data['tabs'], list):
                            tabs = chat_data['tabs']
                            logger.info(f"Found {len(tabs)} tabs in chat data")
                            
                            # Process each tab
                            for tab in tabs:
                                # Try different keys for messages
                                messages = None
                                for key in ['messages', 'bubbles']:
                                    if key in tab and isinstance(tab[key], list):
                                        messages = tab[key]
                                        break
                                        
                                if not messages:
                                    continue
                                    
                                logger.debug(f"Found {len(messages)} messages in tab {tab.get('id', 'unknown')}")
                                
                                # Process each message
                                for i, message in enumerate(messages):
                                    try:
                                        # Check if this is a user message
                                        is_user = False
                                        if 'role' in message and message['role'] == 'user':
                                            is_user = True
                                        elif 'type' in message and message['type'] == 'user':
                                            is_user = True
                                            
                                        if not is_user:
                                            continue
                                            
                                        # Extract the prompt text
                                        prompt_text = None
                                        for key in ['content', 'text', 'initText', 'rawText']:
                                            if key in message and message[key]:
                                                prompt_text = message[key]
                                                break
                                                
                                        if not prompt_text:
                                            # Try to extract from delegate
                                            if 'delegate' in message and message['delegate'] and 'a' in message['delegate']:
                                                prompt_text = message['delegate']['a']
                                                
                                        if not prompt_text:
                                            continue
                                            
                                        # Create a prompt object
                                        prompt_id = f"chat_{tab.get('id', 'unknown')}_{i}"
                                        prompt = {
                                            'id': prompt_id,
                                            'text': prompt_text,
                                            'timestamp': message.get('timestamp', tab.get('timestamp', 0)),
                                            'files': []
                                        }
                                        
                                        # Extract files if available
                                        if 'selections' in message and message['selections']:
                                            prompt['files'] = [{'path': s.get('text', '')} for s in message['selections'] if 'text' in s]
                                            
                                        # Add to prompts list
                                        prompts.append(prompt)
                                        
                                        # Look for the next message as a response
                                        if i + 1 < len(messages):
                                            next_message = messages[i + 1]
                                            
                                            # Check if this is an assistant message
                                            is_assistant = False
                                            if 'role' in next_message and next_message['role'] == 'assistant':
                                                is_assistant = True
                                            elif 'type' in next_message and next_message['type'] == 'ai':
                                                is_assistant = True
                                                
                                            if is_assistant:
                                                # Extract the response text
                                                response_text = None
                                                for key in ['content', 'text', 'markdown']:
                                                    if key in next_message and next_message[key]:
                                                        response_text = next_message[key]
                                                        break
                                                        
                                                if response_text:
                                                    # Create a generation object
                                                    generation = {
                                                        'promptId': prompt_id,
                                                        'response': response_text,
                                                        'timestamp': next_message.get('timestamp', tab.get('timestamp', 0)),
                                                        'model': next_message.get('model', '')
                                                    }
                                                    
                                                    # Add to generations list
                                                    generations.append(generation)
                                    except Exception as e:
                                        logger.error(f"Error processing message: {e}")
                                        continue
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse chat data JSON: {e}")
                    if chat_rows and len(chat_rows) > 0 and len(chat_rows[0]) > 0:
                        sample = chat_rows[0][0][:100] + "..." if len(chat_rows[0][0]) > 100 else chat_rows[0][0]
                        logger.error(f"Sample of chat data: {sample}")
                except Exception as e:
                    logger.error(f"Error processing chat data: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            logger.info(f"Retrieved {len(prompts)} prompts and {len(generations)} generations")
            return prompts, generations
            
        except Exception as e:
            logger.error(f"Error getting cursor chat data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [], []
            
    def match_prompts_with_generations(self, prompts: List[Dict[str, Any]], generations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match prompts with their corresponding generations.
        
        Args:
            prompts: List of prompts
            generations: List of generations
            
        Returns:
            List[Dict[str, Any]]: Matched prompts and generations
        """
        try:
            # Log sample prompt and generation for debugging
            if prompts and len(prompts) > 0:
                sample_prompt = prompts[0]
                logger.info(f"Sample prompt keys: {list(sample_prompt.keys())}")
                logger.info(f"Sample prompt: {json.dumps(sample_prompt)[:500]}...")
            
            if generations and len(generations) > 0:
                sample_generation = generations[0]
                logger.info(f"Sample generation keys: {list(sample_generation.keys())}")
                logger.info(f"Sample generation: {json.dumps(sample_generation)[:500]}...")
            
            # Create a dictionary of generations by id
            # Try different possible key names for the prompt ID in generations
            generation_dict = {}
            for gen in generations:
                # Try different possible key names
                for key in ['promptId', 'prompt_id', 'id']:
                    prompt_id = gen.get(key)
                    if prompt_id:
                        generation_dict[prompt_id] = gen
                        break
            
            # Also create a dictionary of generations by timestamp for matching
            generation_by_timestamp = {}
            for gen in generations:
                # Try different possible key names for timestamp
                for key in ['timestamp', 'unixMs', 'time', 'date']:
                    timestamp = gen.get(key)
                    if timestamp:
                        # Convert to string for dictionary key
                        timestamp_str = str(timestamp)
                        generation_by_timestamp[timestamp_str] = gen
                        break
            
            logger.info(f"Created generation dictionary with {len(generation_dict)} entries by ID and {len(generation_by_timestamp)} by timestamp")
            
            # Match prompts with generations
            matched_data = []
            for index, prompt in enumerate(prompts):
                # Try different possible key names for the prompt ID
                prompt_id = None
                for key in ['id', 'promptId', 'prompt_id']:
                    if key in prompt:
                        prompt_id = prompt.get(key)
                        break
                
                # If no ID found, generate one based on the prompt text and index
                if not prompt_id:
                    # Try to get text from the prompt
                    prompt_text = None
                    for key in ['prompt', 'text', 'content', 'message', 'initText', 'rawText']:
                        if key in prompt:
                            prompt_text = prompt.get(key)
                            if prompt_text:
                                # If it's a complex structure, try to extract the text
                                if isinstance(prompt_text, dict) and 'delegate' in prompt_text:
                                    prompt_text = prompt_text.get('delegate', {}).get('a', '')
                                elif isinstance(prompt_text, str) and prompt_text.startswith('{'):
                                    try:
                                        # Try to parse as JSON
                                        json_data = json.loads(prompt_text)
                                        if 'root' in json_data and 'children' in json_data['root']:
                                            # Try to extract text from the JSON structure
                                            prompt_text = json_data['root']['children'][0]['children'][0].get('text', '')
                                    except:
                                        # Keep the original text if parsing fails
                                        pass
                                break
                    
                    if prompt_text:
                        # Generate a hash from the prompt text and index
                        import hashlib
                        prompt_hash = hashlib.md5(f"{prompt_text}_{index}".encode()).hexdigest()
                        prompt_id = f"generated_{prompt_hash[:10]}"
                        logger.info(f"Generated ID for prompt in matching: {prompt_id}")
                    else:
                        logger.warning(f"Prompt missing both ID and text: {json.dumps(prompt)[:100]}...")
                        continue
                
                # Try to find the corresponding generation by ID
                generation = generation_dict.get(prompt_id)
                
                # If not found by ID, try to match by timestamp
                if not generation:
                    # Try different possible key names for timestamp
                    for key in ['timestamp', 'unixMs', 'time', 'date']:
                        timestamp = prompt.get(key)
                        if timestamp:
                            # Convert to string for dictionary lookup
                            timestamp_str = str(timestamp)
                            generation = generation_by_timestamp.get(timestamp_str)
                            if generation:
                                logger.info(f"Matched prompt and generation by timestamp: {timestamp_str}")
                                break
                
                # If still not found, try to find the closest timestamp
                if not generation and 'timestamp' in prompt:
                    prompt_timestamp = prompt.get('timestamp')
                    closest_generation = None
                    closest_diff = float('inf')
                    
                    for gen in generations:
                        for key in ['timestamp', 'unixMs', 'time', 'date']:
                            gen_timestamp = gen.get(key)
                            if gen_timestamp:
                                # Calculate time difference
                                time_diff = abs(prompt_timestamp - gen_timestamp)
                                if time_diff < closest_diff:
                                    closest_diff = time_diff
                                    closest_generation = gen
                    
                    # If the closest generation is within 5 seconds, use it
                    if closest_generation and closest_diff < 5000:  # 5 seconds in milliseconds
                        generation = closest_generation
                        logger.info(f"Matched prompt and generation by closest timestamp: diff={closest_diff}ms")
                
                if generation:
                    logger.debug(f"Found matching generation for prompt ID: {prompt_id}")
                else:
                    logger.debug(f"No matching generation found for prompt ID: {prompt_id}")
                
                # Try different possible key names for the prompt text
                prompt_text = None
                for key in ['prompt', 'text', 'content', 'message', 'initText', 'rawText']:
                    if key in prompt:
                        prompt_text = prompt.get(key)
                        if prompt_text:
                            # If it's a complex structure, try to extract the text
                            if isinstance(prompt_text, dict) and 'delegate' in prompt_text:
                                prompt_text = prompt_text.get('delegate', {}).get('a', '')
                            elif isinstance(prompt_text, str) and prompt_text.startswith('{'):
                                try:
                                    # Try to parse as JSON
                                    json_data = json.loads(prompt_text)
                                    if 'root' in json_data and 'children' in json_data['root']:
                                        # Try to extract text from the JSON structure
                                        prompt_text = json_data['root']['children'][0]['children'][0].get('text', '')
                                except:
                                    # Keep the original text if parsing fails
                                    pass
                            break
                
                # Try different possible key names for the response text
                response_text = None
                if generation:
                    for key in ['response', 'text', 'content', 'message', 'textDescription', 'markdown']:
                        if key in generation:
                            response_text = generation.get(key)
                            if response_text:
                                break
                
                # Try different possible key names for the model
                model_name = None
                if generation:
                    for key in ['model', 'model_name', 'modelName', 'type']:
                        if key in generation:
                            model_name = generation.get(key)
                            if model_name:
                                break
                
                # Extract files from prompt
                files = prompt.get('files', [])
                
                # If no files in prompt but generation has them, use those
                if not files and generation and 'files' in generation:
                    files = generation.get('files', [])
                
                # If we have a textDescription but no files, try to extract file info from it
                if not files and generation and 'textDescription' in generation:
                    text_desc = generation.get('textDescription', '')
                    if text_desc:
                        # Try to extract file paths from the text description
                        import re
                        file_paths = re.findall(r'[\'"]([^\'"]*.(?:py|js|ts|html|css|md|txt|json|yaml|yml))[\'"]', text_desc)
                        if file_paths:
                            files = [{'path': path} for path in file_paths]
                            logger.info(f"Extracted {len(files)} file paths from textDescription")
                
                # If we have selections in the prompt, use those as files
                if 'selections' in prompt and prompt['selections']:
                    selection_files = [{'path': s.get('text', '')} for s in prompt['selections'] if 'text' in s]
                    if selection_files:
                        files.extend(selection_files)
                        logger.info(f"Added {len(selection_files)} files from selections")
                
                # Only add items that have prompt text
                if prompt_text:
                    matched_item = {
                        'prompt_id': prompt_id,
                        'prompt_text': prompt_text,
                        'timestamp': prompt.get('timestamp', 0),
                        'files': files,
                        'response_text': response_text,
                        'model_name': model_name
                    }
                    
                    matched_data.append(matched_item)
                    logger.debug(f"Added matched item with prompt ID: {prompt_id}")
                
            logger.info(f"Matched {len(matched_data)} prompts with generations")
            return matched_data
            
        except Exception as e:
            logger.error(f"Error matching prompts with generations: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
            
    def process_cursor_chat_data(self, project_id: Optional[int] = None) -> int:
        """Process Cursor chat data and store in the database.
        
        Args:
            project_id: ID of the cursor project (optional)
            
        Returns:
            int: Number of new chats added
        """
        try:
            logger.info(f"Processing Cursor chat data for project_id: {project_id}")
            
            # Get the latest workspace database path
            db_path = self.get_latest_workspace_db_path()
            if not db_path:
                logger.error("Failed to get latest workspace database path")
                return 0
                
            logger.info(f"Using Cursor database at: {db_path}")
                
            # Get prompts and generations
            prompts, generations = self.get_cursor_chat_data(db_path)
            logger.info(f"Retrieved {len(prompts)} prompts and {len(generations)} generations")
            
            if not prompts:
                logger.info("No prompts found in Cursor database")
                return 0
                
            # Process prompts and generations
            return self._process_prompts_and_generations(project_id, prompts, generations)
            
        except Exception as e:
            logger.error(f"Error processing cursor chat data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0
            
    def get_all_cursor_chat_data(self, db_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all chat-related data from the Cursor database.
        
        Args:
            db_path: Path to the Cursor database
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of chat-related data
        """
        try:
            # Query all chat-related data
            logger.info(f"Querying all chat-related data from Cursor database: {db_path}")
            
            # Define the keys to query
            chat_keys = [
                'aiService.prompts',
                'aiService.generations',
                'workbench.panel.aichat.view.aichat.chatdata',
                'aiService.conversations',
                'aiService.messages'
            ]
            
            # Build the query
            placeholders = ', '.join(['?'] * len(chat_keys))
            query = f"SELECT [key], value FROM ItemTable WHERE [key] IN ({placeholders})"
            
            # Execute the query
            conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
            cursor = conn.cursor()
            cursor.execute(query, chat_keys)
            rows = cursor.fetchall()
            conn.close()
            
            logger.info(f"Query executed successfully, fetched {len(rows)} rows")
            
            # Process the results
            result = {}
            for row in rows:
                if len(row) < 2:
                    continue
                    
                key = row[0]
                value = row[1]
                
                try:
                    # Parse the JSON value
                    data = json.loads(value)
                    result[key] = data
                    logger.info(f"Successfully parsed data for key: {key}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON for key: {key}")
                    
            return result
            
        except Exception as e:
            logger.error(f"Error getting all cursor chat data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {}
            
    def process_all_cursor_chat_data(self, project_id: Optional[int] = None) -> int:
        """Process all cursor chat data and store in the database.
        
        Args:
            project_id: ID of the cursor project (optional)
            
        Returns:
            int: Number of new chats added
        """
        try:
            logger.info(f"Processing all Cursor chat data for project_id: {project_id}")
            
            # Get the latest workspace database path
            db_path = self.get_latest_workspace_db_path()
            if not db_path:
                logger.error("Failed to get latest workspace database path")
                return 0
                
            logger.info(f"Using Cursor database at: {db_path}")
                
            # Get all chat-related data
            all_chat_data = self.get_all_cursor_chat_data(db_path)
            if not all_chat_data:
                logger.info("No chat data found in Cursor database")
                return 0
                
            # Process each type of chat data
            new_chats_count = 0
            
            # First try the standard prompts and generations
            if 'aiService.prompts' in all_chat_data and 'aiService.generations' in all_chat_data:
                logger.info("Processing prompts and generations")
                prompts = all_chat_data['aiService.prompts']
                generations = all_chat_data['aiService.generations']
                
                # Convert to lists if they're dictionaries
                if isinstance(prompts, dict):
                    prompts = [{"id": k, **v} for k, v in prompts.items() if isinstance(v, dict)]
                    
                if isinstance(generations, dict):
                    generations = [{"id": k, **v} for k, v in generations.items() if isinstance(v, dict)]
                    
                # Process the prompts and generations
                if isinstance(prompts, list) and len(prompts) > 0:
                    # Call the existing method to process prompts and generations
                    count = self._process_prompts_and_generations(project_id, prompts, generations)
                    new_chats_count += count
                    logger.info(f"Added {count} chats from prompts and generations")
            
            # Then try the chat data
            if 'workbench.panel.aichat.view.aichat.chatdata' in all_chat_data:
                logger.info("Processing chat data")
                chat_data = all_chat_data['workbench.panel.aichat.view.aichat.chatdata']
                
                # Process the chat data
                if isinstance(chat_data, dict) and 'tabs' in chat_data and isinstance(chat_data['tabs'], list):
                    count = self._process_chat_data(project_id, chat_data)
                    new_chats_count += count
                    logger.info(f"Added {count} chats from chat data")
            
            # If we didn't find any data in the all_chat_data approach, try the direct approach
            if new_chats_count == 0:
                logger.info("No chats added from all_chat_data, trying direct approach")
                
                # Get prompts and generations directly
                prompts, generations = self.get_cursor_chat_data(db_path)
                logger.info(f"Retrieved {len(prompts)} prompts and {len(generations)} generations directly")
                
                if prompts:
                    # Process the prompts and generations
                    count = self._process_prompts_and_generations(project_id, prompts, generations)
                    new_chats_count += count
                    logger.info(f"Added {count} chats from direct approach")
            
            logger.info(f"Added a total of {new_chats_count} new cursor chats")
            return new_chats_count
            
        except Exception as e:
            logger.error(f"Error processing all cursor chat data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0
            
    def _process_prompts_and_generations(self, project_id: Optional[int], prompts: List[Dict[str, Any]], generations: List[Dict[str, Any]]) -> int:
        """Process prompts and generations and store in the database.
        
        Args:
            project_id: ID of the cursor project (optional)
            prompts: List of prompts
            generations: List of generations
            
        Returns:
            int: Number of new chats added
        """
        # Match prompts with generations
        matched_data = self.match_prompts_with_generations(prompts, generations)
        logger.info(f"Matched {len(matched_data)} prompts with generations")
        
        # If we couldn't match any prompts with generations, but we have prompts,
        # store the prompts without responses
        if not matched_data and prompts:
            logger.info("No matched data found, but we have prompts. Storing prompts without responses.")
            
            # Process each prompt
            new_chats_count = 0
            for index, prompt in enumerate(prompts):
                try:
                    # Try different possible key names for the prompt ID
                    prompt_id = None
                    for key in ['id', 'promptId', 'prompt_id']:
                        if key in prompt:
                            prompt_id = prompt.get(key)
                            break
                    
                    # If no ID found, generate one based on the prompt text and index
                    if not prompt_id:
                        # Try to get text from the prompt
                        prompt_text = None
                        for key in ['prompt', 'text', 'content', 'message']:
                            if key in prompt:
                                prompt_text = prompt.get(key)
                                if prompt_text:
                                    break
                        
                        if prompt_text:
                            # Generate a hash from the prompt text and index
                            import hashlib
                            prompt_hash = hashlib.md5(f"{prompt_text}_{index}".encode()).hexdigest()
                            prompt_id = f"generated_{prompt_hash[:10]}"
                            logger.info(f"Generated ID for prompt: {prompt_id}")
                        else:
                            logger.warning("Prompt missing both ID and text, skipping")
                            continue
                    
                    # Try different possible key names for the prompt text
                    prompt_text = None
                    for key in ['prompt', 'text', 'content', 'message']:
                        if key in prompt:
                            prompt_text = prompt.get(key)
                            if prompt_text:
                                break
                    
                    if not prompt_text:
                        logger.warning(f"Prompt {prompt_id} missing text, skipping")
                        continue
                    
                    # Convert timestamp from milliseconds to datetime
                    timestamp = datetime.fromtimestamp(prompt.get('timestamp', 0) / 1000) if prompt.get('timestamp', 0) else datetime.now()
                    
                    # Add to database
                    chat_id = self.db_ops.add_cursor_chat(
                        cursor_project_id=project_id,
                        prompt_key='aiService.prompts',
                        prompt_id=prompt_id,
                        prompt_text=prompt_text,
                        response_text=None,  # No response
                        files=prompt.get('files', []),
                        timestamp=timestamp,
                        model_name=None  # No model
                    )
                    
                    if chat_id is not None:
                        logger.info(f"Added prompt-only chat with ID: {chat_id}")
                        new_chats_count += 1
                    else:
                        logger.warning(f"Failed to add prompt-only chat with prompt_id: {prompt_id}")
                        
                except Exception as e:
                    logger.error(f"Error processing prompt: {e}")
                    logger.error(f"Prompt: {prompt}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Continue with the next prompt
                    continue
            
            logger.info(f"Added {new_chats_count} prompt-only chats")
            return new_chats_count
        
        if not matched_data:
            logger.info("No matched data found")
            return 0
            
        # Store in database
        new_chats_count = 0
        for item in matched_data:
            try:
                # Convert timestamp from milliseconds to datetime
                timestamp = datetime.fromtimestamp(item['timestamp'] / 1000) if item['timestamp'] else datetime.now()
                
                # Log the item being processed
                logger.info(f"Processing chat item: prompt_id={item['prompt_id']}, timestamp={timestamp}")
                
                # Add to database
                chat_id = self.db_ops.add_cursor_chat(
                    cursor_project_id=project_id,
                    prompt_key='aiService.prompts',
                    prompt_id=item['prompt_id'],
                    prompt_text=item['prompt_text'],
                    response_text=item['response_text'],
                    files=item['files'],
                    timestamp=timestamp,
                    model_name=item['model_name']
                )
                
                if chat_id is not None:
                    logger.info(f"Added chat with ID: {chat_id}")
                    new_chats_count += 1
                else:
                    logger.warning(f"Failed to add chat with prompt_id: {item['prompt_id']}")
                    
            except Exception as e:
                logger.error(f"Error processing chat item: {e}")
                logger.error(f"Item: {item}")
                # Continue with the next item
                continue
                
        logger.info(f"Added {new_chats_count} new cursor chats")
        return new_chats_count
        
    def _process_chat_data(self, project_id: Optional[int], chat_data: Dict[str, Any]) -> int:
        """Process chat data and store in the database.
        
        Args:
            project_id: ID of the cursor project (optional)
            chat_data: Chat data
            
        Returns:
            int: Number of new chats added
        """
        try:
            # Extract tabs from chat data
            tabs = chat_data.get('tabs', [])
            if not tabs:
                logger.info("No tabs found in chat data")
                return 0
                
            logger.info(f"Found {len(tabs)} tabs in chat data")
            
            # Process each tab
            new_chats_count = 0
            for tab in tabs:
                try:
                    # Extract messages from tab
                    messages = tab.get('messages', [])
                    if not messages:
                        logger.debug(f"No messages found in tab {tab.get('id', 'unknown')}")
                        continue
                        
                    logger.debug(f"Found {len(messages)} messages in tab {tab.get('id', 'unknown')}")
                    
                    # Process each message
                    for i in range(0, len(messages), 2):
                        try:
                            # Get the user message (prompt)
                            user_message = messages[i] if i < len(messages) else None
                            if not user_message or user_message.get('role') != 'user':
                                continue
                                
                            # Get the assistant message (response)
                            assistant_message = messages[i+1] if i+1 < len(messages) else None
                            if not assistant_message or assistant_message.get('role') != 'assistant':
                                assistant_message = None
                                
                            # Extract prompt ID and text
                            prompt_id = f"{tab.get('id', 'unknown')}_{i}"
                            prompt_text = user_message.get('content', '')
                            if not prompt_text:
                                continue
                                
                            # Extract response text
                            response_text = assistant_message.get('content', '') if assistant_message else None
                            
                            # Extract timestamp
                            timestamp = user_message.get('timestamp', 0)
                            if not timestamp:
                                timestamp = tab.get('timestamp', 0)
                                
                            # Convert timestamp from milliseconds to datetime
                            timestamp = datetime.fromtimestamp(timestamp / 1000) if timestamp else datetime.now()
                            
                            # Extract model name
                            model_name = assistant_message.get('model', '') if assistant_message else None
                            
                            # Add to database
                            chat_id = self.db_ops.add_cursor_chat(
                                cursor_project_id=project_id,
                                prompt_key='workbench.panel.aichat.view.aichat.chatdata',
                                prompt_id=prompt_id,
                                prompt_text=prompt_text,
                                response_text=response_text,
                                files=None,  # No files
                                timestamp=timestamp,
                                model_name=model_name
                            )
                            
                            if chat_id is not None:
                                logger.debug(f"Added chat with ID: {chat_id}")
                                new_chats_count += 1
                            else:
                                logger.warning(f"Failed to add chat with prompt_id: {prompt_id}")
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            # Continue with the next message
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing tab: {e}")
                    # Continue with the next tab
                    continue
                    
            logger.info(f"Added {new_chats_count} new cursor chats from chat data")
            return new_chats_count
            
        except Exception as e:
            logger.error(f"Error processing chat data: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 0 