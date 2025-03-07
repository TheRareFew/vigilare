#!/usr/bin/env python3
"""
Test script for the report generator.
"""

import os
import sys
import subprocess
import time
from datetime import datetime, timedelta

# Add the parent directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.analysis.report_generator import ReportGenerator
from src.core.database import init_database
from src.core.config import setup_openai_key

def start_aw_server():
    """Start the ActivityWatch server in testing mode."""
    try:
        print("Starting ActivityWatch server in testing mode...")
        
        # Check if aw-server is already running
        try:
            import requests
            response = requests.get("http://127.0.0.1:5666/api/0/info")
            if response.status_code == 200:
                print("ActivityWatch server is already running")
                return True
        except:
            pass  # Server not running, continue to start it
        
        # Start aw-server with testing flag
        process = subprocess.Popen(
            ["aw-server", "--testing"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        for _ in range(10):  # Try for 10 seconds
            try:
                import requests
                response = requests.get("http://127.0.0.1:5666/api/0/info")
                if response.status_code == 200:
                    print("ActivityWatch server started successfully")
                    return True
            except:
                pass
            time.sleep(1)
        
        print("Failed to start ActivityWatch server")
        return False
        
    except Exception as e:
        print(f"Error starting ActivityWatch server: {e}")
        return False

def main():
    """Main entry point."""
    try:
        print("Initializing report generator test...")
        
        # Setup OpenAI key
        setup_openai_key()
        
        # Start ActivityWatch server
        if not start_aw_server():
            print("Failed to start ActivityWatch server")
            sys.exit(1)
        
        # Initialize database using ActivityWatch's database
        print("Initializing database using ActivityWatch's database...")
        if not init_database(testing=True):  # Use testing mode to match AW server
            print("Failed to initialize database")
            sys.exit(1)
        
        # Calculate time range
        end_time = datetime.now()
        
        # Initialize report generator with testing mode
        generator = ReportGenerator(testing=True)
        
        print(f"Generating hourly report ending at {end_time}...")
        
        # Generate report
        report = generator.generate_hourly_report(end_time)
        
        if report:
            print(f"Report generated successfully with ID: {report.report_id}")
            
            # Print a preview of the report
            preview_length = min(500, len(report.report_text))
            preview = report.report_text[:preview_length] + "..." if len(report.report_text) > preview_length else report.report_text
            print("\nReport Preview:")
            print("=" * 80)
            print(preview)
            print("=" * 80)
        else:
            print("Failed to generate report")
        
        # Close the generator
        generator.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 