#!/usr/bin/env python3
"""
Database setup script for Vigilare.
Creates all necessary tables and initializes reference data.
"""

import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.storage.models import (
    IntervalTypeModel, PromptTypeModel, ScreenshotModel,
    PromptModel, PromptEmbeddingModel, ReportModel,
    ReportEmbeddingModel, AppClassificationModel
)
from src.core.database import init_database, close_database

def create_tables():
    """Create all database tables."""
    models = [
        IntervalTypeModel,
        PromptTypeModel,
        ScreenshotModel,
        PromptModel,
        PromptEmbeddingModel,
        ReportModel,
        ReportEmbeddingModel,
        AppClassificationModel
    ]
    
    for model in models:
        model.create_table()
        print(f"Created table: {model._meta.table_name}")

def initialize_reference_data():
    """Initialize reference data in the database."""
    # Initialize interval types
    interval_types = ['hourly', 'daily', 'weekly', 'monthly', 'yearly']
    for interval in interval_types:
        IntervalTypeModel.get_or_create(interval_name=interval)
        print(f"Created interval type: {interval}")

    # Initialize prompt types
    prompt_types = ['programming', 'research', 'documentation', 'other']
    for p_type in prompt_types:
        PromptTypeModel.get_or_create(prompt_type_name=p_type)
        print(f"Created prompt type: {p_type}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Setup Vigilare database')
    parser.add_argument('--testing', action='store_true', help='Use testing database')
    return parser.parse_args()

def main():
    """Main function to set up the database."""
    try:
        args = parse_args()
        
        print("Initializing database...")
        if not init_database(testing=args.testing):
            print("Failed to initialize database")
            sys.exit(1)
        
        print("\nCreating database tables...")
        create_tables()
        
        print("\nInitializing reference data...")
        initialize_reference_data()
        
        print("\nDatabase setup completed successfully!")
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        sys.exit(1)
    finally:
        close_database()

if __name__ == "__main__":
    main() 