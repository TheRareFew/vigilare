#!/usr/bin/env python3
"""
Main entry point for the Vigilare application.
"""

import argparse
import logging
import logging.config
import os
import sys
import yaml
import subprocess
import time
import threading
from pathlib import Path
from datetime import datetime

# Add the parent directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.logger import setup_logging
from src.core.daemon import Daemon
from src.core.aw_client import ActivityWatchClient
from src.core.database import init_database, get_database, close_database
from src.core.config import setup_openai_key
from src.api.server import start_server
from src.cli.report_cmd import report

logger = logging.getLogger('vigilare')
aw_modules = ["aw-watcher-afk", "aw-watcher-window", "aw-watcher-input"]  # Removed aw-core as it's not an executable module

# Global list to track all module processes
module_processes = []

# Global thread for API server
api_server_thread = None

def create_output_reader(process, module_name):
    """Create an output reader thread for a process."""
    def output_reader():
        try:
            while process and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    print(f"[{module_name}] {line.strip()}")
        except (ValueError, IOError):
            # Handle case where stdout is closed
            pass
    return output_reader

def start_aw_modules():
    """Start ActivityWatch modules."""
    global module_processes
    max_retries = 3
    retry_delay = 2  # seconds
    
    for module in aw_modules:
        logger.info(f"Starting {module}...")
        for attempt in range(max_retries):
            try:
                # Start module with output capture
                process = subprocess.Popen(
                    f"{module} --testing",  # Add testing flag to match server mode
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Create and start output thread only for non-input watchers
                if module != "aw-watcher-input":
                    import threading
                    output_thread = threading.Thread(
                        target=create_output_reader(process, module),
                        name=f"{module}-output-reader"
                    )
                    output_thread.start()
                
                # Wait a bit to check if process dies immediately
                time.sleep(1)
                if process.poll() is not None:
                    exit_code = process.poll()
                    logger.error(f"{module} failed to start (exit code: {exit_code})")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying {module} in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    else:
                        logger.error(f"Failed to start {module} after {max_retries} attempts")
                        break
                
                # If we get here, process started successfully
                logger.info(f"Successfully started {module}")
                module_processes.append(process)
                break
                
            except Exception as e:
                logger.error(f"Failed to start {module}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying {module} in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to start {module} after {max_retries} attempts")

def start_aw_server():
    """Start ActivityWatch server in testing mode."""
    global server_process  # Keep track of server process globally for cleanup
    logger = logging.getLogger('vigilare')
    server_ready = False
    
    try:
        # Check if aw-server is already running
        if sys.platform == "win32":
            check_cmd = "tasklist | findstr aw-server"
        else:
            check_cmd = "pgrep -f aw-server"
            
        result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout.strip():
            logger.info("ActivityWatch server is already running")
            return True
            
        # Start aw-server in testing mode
        logger.info("Starting ActivityWatch server in testing mode...")
        if sys.platform == "win32":
            # Start server and capture output
            server_process = subprocess.Popen(
                "aw-server --testing",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Instead of a daemon thread, use a regular thread we can control
            def output_reader():
                nonlocal server_ready
                try:
                    while server_process and server_process.poll() is None:
                        line = server_process.stdout.readline()
                        if line:
                            print(f"[aw-server] {line.strip()}")
                            # Check for server ready message
                            if "Running on http://localhost:5666" in line:
                                server_ready = True
                except (ValueError, IOError):
                    # Handle case where stdout is closed
                    pass
            
            import threading
            output_thread = threading.Thread(target=output_reader)
            output_thread.start()
        else:
            # Use nohup to run in background on Unix
            server_process = subprocess.Popen(["nohup", "aw-server", "--testing"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL,
                           start_new_session=True)
            server_ready = True  # We can't check output in this case
            
        # Wait for server to be ready
        max_wait = 30  # Maximum seconds to wait
        start_time = time.time()
        while not server_ready and time.time() - start_time < max_wait:
            time.sleep(0.1)
            
        if not server_ready and sys.platform == "win32":
            logger.error("Server failed to start within timeout period")
            return False
            
        logger.info("ActivityWatch server started in testing mode")
        return True
        
    except Exception as e:
        logger.error(f"Error starting ActivityWatch server: {e}")
        return False

def start_api_server():
    """Start the API server in a separate thread."""
    global api_server_thread
    
    def run_server():
        logger.info("Starting Vigilare API server...")
        try:
            start_server(host='localhost', port=5667)
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
    
    api_server_thread = threading.Thread(target=run_server, daemon=True)
    api_server_thread.start()
    
    # Give it a moment to start
    time.sleep(2)
    logger.info("Vigilare API server started on port 5667")

def cleanup():
    """Cleanup function to handle graceful shutdown."""
    logger = logging.getLogger('vigilare')
    
    # Clean up module processes
    for process in module_processes:
        if process and process.poll() is None:
            logger.info(f"Shutting down process {process.pid}...")
            if sys.platform == "win32":
                # On Windows, we need to kill the process tree
                subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True)
            else:
                process.terminate()
                process.wait(timeout=5)  # Give it 5 seconds to terminate gracefully
            
            # Force kill if still running
            if process.poll() is None:
                process.kill()
                process.wait()
    
    # Clean up server process
    if 'server_process' in globals() and server_process:
        logger.info("Shutting down ActivityWatch server...")
        if sys.platform == "win32":
            # On Windows, we need to kill the process tree
            subprocess.run(f"taskkill /F /T /PID {server_process.pid}", shell=True)
        else:
            server_process.terminate()
            server_process.wait(timeout=5)  # Give it 5 seconds to terminate gracefully
        
        # Force kill if still running
        if server_process.poll() is None:
            server_process.kill()
            server_process.wait()
    
    # Close any database connections
    try:
        close_database()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

# Register cleanup handler
import atexit
atexit.register(cleanup)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Vigilare - Intelligent Productivity Tracking')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Main command (default)
    parser.add_argument('--config', default=os.path.join(PROJECT_ROOT, 'config', 'config.yaml'),
                      help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--hours', type=int, default=1, 
                             help='Number of hours to look back for the report')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        # Setup OpenAI key from .env before loading config
        setup_openai_key()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Handle environment variables and resolve paths
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                # Special handling for known path fields
                if 'path' in obj and 'database' not in obj:  # Skip database paths
                    obj['path'] = os.path.normpath(obj['path'])
                if 'logging_config' in obj:
                    obj['logging_config'] = os.path.normpath(obj['logging_config'])
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(elem) for elem in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                if env_var not in os.environ:
                    raise ValueError(f"Required environment variable {env_var} not set")
                return os.environ[env_var]
            return obj
            
        return replace_env_vars(config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def ensure_log_directory(log_path):
    """Ensure log directory exists."""
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        log_config_path = os.path.abspath(os.path.join(PROJECT_ROOT, config['logging_config']))
        # Ensure the log directory exists
        os.makedirs(os.path.dirname(log_config_path), exist_ok=True)
        setup_logging(log_config_path, args.debug)
        
        # Handle different commands
        if hasattr(args, 'command') and args.command == 'report':
            # Start ActivityWatch server in testing mode first
            if not start_aw_server():
                logger.error("Failed to start ActivityWatch server")
                sys.exit(1)
                
            # Initialize database using ActivityWatch's database
            logger.info("Initializing database using ActivityWatch's database...")
            if not init_database(testing=True):  # Use testing mode to match AW server
                logger.error("Failed to initialize database")
                sys.exit(1)
            
            from src.analysis.report_generator import ReportGenerator
            
            logger.info(f"Generating report for the last {args.hours} hour(s)")
            
            # Calculate time range
            end_time = datetime.now()
            
            # Initialize report generator
            generator = ReportGenerator(testing=True)
            
            # Generate report
            report = generator.generate_hourly_report(end_time)
            
            if report:
                logger.info(f"Report generated successfully with ID: {report.report_id}")
                print(f"Report generated successfully with ID: {report.report_id}")
                
                # Print a preview of the report
                preview_length = min(500, len(report.report_text))
                preview = report.report_text[:preview_length] + "..." if len(report.report_text) > preview_length else report.report_text
                print("\nReport Preview:")
                print("=" * 80)
                print(preview)
                print("=" * 80)
            else:
                logger.error("Failed to generate report")
                print("Failed to generate report")
            
            # Close the generator
            generator.close()
            return
        
        # Default command - start the daemon
        
        # Start ActivityWatch server in testing mode first
        if not start_aw_server():
            logger.error("Failed to start ActivityWatch server")
            sys.exit(1)
        
        # Start ActivityWatch modules
        start_aw_modules()
        
        # Initialize ActivityWatch client
        aw_client = ActivityWatchClient(
            client_id=config['aw_client_id'],
            testing=True,  # Always use testing mode to match server
            server_url=config['aw_server_url']
        )
        
        # Now initialize our database using ActivityWatch's database
        logger.info("Initializing database using ActivityWatch's database...")
        if not init_database(testing=True):  # Use testing mode to match AW server
            logger.error("Failed to initialize database")
            sys.exit(1)
        
        # Start our API server
        start_api_server()
        
        # Initialize and start daemon
        daemon = Daemon(config, aw_client)
        daemon.start()
        
    except KeyboardInterrupt:
        logger.info("Shutting down Vigilare...")
        cleanup()
    except Exception as e:
        logger.error(f"Error running Vigilare: {e}")
        cleanup()
        sys.exit(1)

if __name__ == '__main__':
    main() 