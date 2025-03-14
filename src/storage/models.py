"""Database models for Vigilare."""

import json
from datetime import datetime
from typing import Dict, Any, Optional

from peewee import (
    Model, AutoField, CharField, DateTimeField,
    DecimalField, ForeignKeyField, BlobField,
    BooleanField, TextField
)

from src.core.database import database_proxy

class BaseModel(Model):
    """Base model class."""
    class Meta:
        database = database_proxy

class IntervalTypeModel(BaseModel):
    """Interval types for reports."""
    interval_type_id = AutoField()
    interval_name = CharField(unique=True)

    class Meta:
        table_name = 'interval_types'

class PromptTypeModel(BaseModel):
    """Types of prompts."""
    prompt_type_id = AutoField()
    prompt_type_name = CharField(unique=True)

    class Meta:
        table_name = 'prompt_types'

class ScreenshotModel(BaseModel):
    """Screenshots with metadata."""
    image_id = AutoField()
    image_path = CharField()
    ocr_text = TextField(null=True)
    image_summary = TextField(null=True)
    context = TextField()  # JSON string containing window_title and application_name
    code_insights = TextField(null=True)  # JSON string containing code analysis insights
    timestamp = DateTimeField(default=datetime.now)

    def set_context(self, context_dict: Dict[str, Any]):
        """Set context as JSON string."""
        self.context = json.dumps(context_dict)

    def get_context(self) -> Dict[str, Any]:
        """Get context as dictionary."""
        return json.loads(self.context)

    def set_code_insights(self, insights_dict: Dict[str, Any]):
        """Set code insights as JSON string."""
        self.code_insights = json.dumps(insights_dict)

    def get_code_insights(self) -> Optional[Dict[str, Any]]:
        """Get code insights as dictionary."""
        if self.code_insights:
            return json.loads(self.code_insights)
        return None

    class Meta:
        table_name = 'screenshots'

class PromptModel(BaseModel):
    """LLM prompts with metadata."""
    prompt_id = AutoField()
    prompt_text = TextField()
    prompt_type = ForeignKeyField(PromptTypeModel, backref='prompts')
    model_name = CharField(null=True)
    llm_tool_used = CharField(null=True)
    timestamp = DateTimeField(default=datetime.now)
    quality_score = DecimalField(null=True)
    confidence = DecimalField(null=True)  # Confidence score for prompt detection (0-1)
    context = TextField()  # Application or environment context

    class Meta:
        table_name = 'prompts'

class PromptEmbeddingModel(BaseModel):
    """Vector embeddings for prompts."""
    prompt = ForeignKeyField(PromptModel, primary_key=True, backref='embedding')
    embedding = BlobField()

    class Meta:
        table_name = 'prompt_embeddings'

class ReportModel(BaseModel):
    """Activity reports."""
    report_id = AutoField()
    report_text = TextField()
    interval_type = ForeignKeyField(IntervalTypeModel, backref='reports')
    timestamp = DateTimeField()  # Start time of the report interval
    period_end = DateTimeField()  # End time of the report interval

    class Meta:
        table_name = 'reports'

class ReportEmbeddingModel(BaseModel):
    """Vector embeddings for reports."""
    report = ForeignKeyField(ReportModel, primary_key=True, backref='embedding')
    embedding = BlobField()

    class Meta:
        table_name = 'report_embeddings'

class AppClassificationModel(BaseModel):
    """Application classifications."""
    app_classification_id = AutoField()
    application_name = CharField()
    window_title_regex = CharField()
    category = CharField()
    confidence = DecimalField()
    manual_override = BooleanField(default=False)

    class Meta:
        table_name = 'app_classifications'

class CursorProjectModel(BaseModel):
    """Cursor IDE project information."""
    project_id = AutoField()
    project_name = CharField(unique=True)
    project_path = CharField()
    cursor_data_path = CharField(null=True)  # Path to Cursor data for the project
    is_active = BooleanField(default=False)
    last_accessed = DateTimeField(default=datetime.now)

    class Meta:
        table_name = 'cursor_projects'

class CursorChatModel(BaseModel):
    """Cursor chat data."""
    chat_id = AutoField()
    cursor_project = ForeignKeyField(CursorProjectModel, backref='chats', null=True)
    prompt_key = CharField()  # Key from the Cursor database (e.g., 'aiService.prompts')
    prompt_id = CharField()  # Unique identifier for the prompt within Cursor
    prompt_text = TextField()  # The actual prompt text
    response_text = TextField(null=True)  # The response text
    files = TextField(null=True)  # JSON string containing associated files
    timestamp = DateTimeField()  # When the prompt was created
    model_name = CharField(null=True)  # The model used for the response
    processed_at = DateTimeField(default=datetime.now)  # When this record was processed

    class Meta:
        table_name = 'cursor_chats'
        indexes = (
            # Create a unique index on prompt_key and prompt_id
            (('prompt_key', 'prompt_id'), True),
        ) 