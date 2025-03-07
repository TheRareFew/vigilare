#!/usr/bin/env python3
"""
Script to list all saved reports.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add the parent directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.core.database import init_database
from src.storage.models import ReportModel

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
        print("Listing all saved reports...")
        
        # Start ActivityWatch server
        if not start_aw_server():
            print("Failed to start ActivityWatch server")
            sys.exit(1)
        
        # Initialize database using ActivityWatch's database
        print("Initializing database using ActivityWatch's database...")
        if not init_database(testing=True):  # Use testing mode to match AW server
            print("Failed to initialize database")
            sys.exit(1)
        
        # Query all reports
        reports = ReportModel.select().order_by(ReportModel.timestamp.desc())
        
        if not reports:
            print("No reports found.")
            return
            
        print(f"Found {len(list(reports))} reports:")
        print("=" * 80)
        
        for report in reports:
            start_time = report.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            end_time = report.period_end.strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"Report ID: {report.report_id}")
            print(f"Time Period: {start_time} to {end_time}")
            print(f"Interval Type: {report.interval_type.interval_name}")
            
            # Print a preview of the report
            preview_length = min(200, len(report.report_text))
            preview = report.report_text[:preview_length] + "..." if len(report.report_text) > preview_length else report.report_text
            print(f"Preview: {preview}")
            print("-" * 80)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 