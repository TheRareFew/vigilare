"""Report generation for productivity, LLM usage, and coding practices."""

import logging
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field

from src.core.aw_client import ActivityWatchClient
from src.storage.models import (
    ReportModel, IntervalTypeModel, PromptModel, ScreenshotModel,
    CursorChatModel, CursorProjectModel
)
from src.storage.operations import DatabaseOperations
from src.utils.logger import get_logger

# Get the application logger
logger = get_logger(__name__)

class AppUsageData(BaseModel):
    """Schema for application usage data."""
    app_name: str = Field(description="Name of the application")
    window_titles: List[str] = Field(description="List of window titles")
    duration_seconds: float = Field(description="Time spent in seconds")
    percentage: float = Field(description="Percentage of total time")

class ReportContext(BaseModel):
    """Schema for report context data."""
    total_active_time: float = Field(description="Total active time in seconds")
    total_afk_time: float = Field(description="Total AFK time in seconds")
    productivity_score: float = Field(description="Productivity score (0-1)")
    app_usage: List[AppUsageData] = Field(description="Application usage data")
    prompts_count: int = Field(description="Number of LLM prompts")
    prompts_by_tool: Dict[str, int] = Field(description="Prompts count by LLM tool")
    prompts_by_type: Dict[str, int] = Field(description="Prompts count by type")
    coding_time: float = Field(description="Time spent coding in seconds")
    coding_languages: Dict[str, int] = Field(description="Time spent in each coding language")
    best_practices_followed: List[str] = Field(description="Best practices followed")
    best_practices_violations: List[str] = Field(description="Best practices violations")
    screenshots_count: int = Field(description="Number of screenshots taken")

class ReportGenerator:
    """Generator for productivity, LLM usage, and coding practice reports."""
    
    def __init__(self, client_id: str = "vigilare_report_generator", 
                 model_name: str = "gpt-4o-mini",
                 api_key: Optional[str] = None,
                 testing: bool = True):
        """Initialize the report generator.
        
        Args:
            client_id: Client identifier for ActivityWatch
            model_name: OpenAI model to use for report generation
            api_key: OpenAI API key (defaults to environment variable)
            testing: Whether to use testing mode (port 5666)
        """
        self.aw_client = ActivityWatchClient(client_id, testing=testing)
        self.model_name = model_name
        self.db_ops = DatabaseOperations()
        
        # Use the configured OpenAI key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
            
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.api_key)
        
        logger.info(f"Initialized ReportGenerator with model: {model_name}")
    
    def _get_interval_type(self, interval_name: str = "hourly") -> IntervalTypeModel:
        """Get or create an interval type.
        
        Args:
            interval_name: Name of the interval type
            
        Returns:
            IntervalTypeModel: The interval type model
        """
        interval_type, created = IntervalTypeModel.get_or_create(
            interval_name=interval_name
        )
        
        if created:
            logger.info(f"Created new interval type: {interval_name}")
        
        return interval_type
    
    def _get_app_usage(self, start_time: datetime, end_time: datetime) -> List[AppUsageData]:
        """Get application usage data for the specified time period.
        
        Args:
            start_time: Start time of the period
            end_time: End time of the period
            
        Returns:
            List[AppUsageData]: List of application usage data
        """
        try:
            # Convert to UTC timezone for AW queries
            start_time_utc = start_time.astimezone(datetime.now().tzinfo)
            end_time_utc = end_time.astimezone(datetime.now().tzinfo)
            
            # Query for window events
            query = """
                window_events = flood(query_bucket(find_bucket("aw-watcher-window_")));
                window_events = sort_by_timestamp(window_events);
                RETURN = window_events;
            """
            
            timeperiods = [(start_time_utc, end_time_utc)]
            results = self.aw_client.client.query(query, timeperiods)
            
            if not results or not results[0]:
                logger.warning(f"No window events found from {start_time} to {end_time}")
                return []
            
            # Process window events
            app_usage = {}
            for event in results[0]:
                app = event["data"].get("app", "Unknown")
                title = event["data"].get("title", "Unknown")
                duration = float(event.get("duration", 0))
                
                if app not in app_usage:
                    app_usage[app] = {
                        "duration": 0.0,
                        "window_titles": set()
                    }
                
                app_usage[app]["duration"] += duration
                app_usage[app]["window_titles"].add(title)
            
            # Calculate total duration
            total_duration = sum(app["duration"] for app in app_usage.values())
            
            # Convert to AppUsageData objects
            app_usage_data = []
            for app_name, data in app_usage.items():
                percentage = (data["duration"] / total_duration) * 100 if total_duration > 0 else 0
                app_usage_data.append(AppUsageData(
                    app_name=app_name,
                    window_titles=list(data["window_titles"]),
                    duration_seconds=data["duration"],
                    percentage=percentage
                ))
            
            # Sort by duration (descending)
            app_usage_data.sort(key=lambda x: x.duration_seconds, reverse=True)
            
            return app_usage_data
            
        except Exception as e:
            logger.error(f"Error getting app usage: {e}")
            return []
    
    def _get_active_afk_time(self, start_time: datetime, end_time: datetime) -> Tuple[float, float]:
        """Get active and AFK time for the specified time period.
        
        Args:
            start_time: Start time of the period
            end_time: End time of the period
            
        Returns:
            Tuple[float, float]: (active_time_seconds, afk_time_seconds)
        """
        try:
            # Convert to UTC timezone for AW queries
            start_time_utc = start_time.astimezone(datetime.now().tzinfo)
            end_time_utc = end_time.astimezone(datetime.now().tzinfo)
            
            # Query for AFK events
            query = """
                afk_events = flood(query_bucket(find_bucket("aw-watcher-afk_")));
                RETURN = afk_events;
            """
            
            timeperiods = [(start_time_utc, end_time_utc)]
            results = self.aw_client.client.query(query, timeperiods)
            
            if not results or not results[0]:
                logger.warning(f"No AFK events found from {start_time} to {end_time}")
                return 0.0, 0.0
            
            # Calculate active and AFK time
            active_time = 0.0
            afk_time = 0.0
            
            for event in results[0]:
                status = event["data"].get("status", "")
                duration = float(event.get("duration", 0))
                
                if status == "not-afk":
                    active_time += duration
                elif status == "afk":
                    afk_time += duration
            
            return active_time, afk_time
            
        except Exception as e:
            logger.error(f"Error getting active/AFK time: {e}")
            return 0.0, 0.0
    
    def _get_coding_time_and_languages(self, app_usage: List[AppUsageData]) -> Tuple[float, Dict[str, int]]:
        """Calculate coding time and languages from app usage data.
        
        Args:
            app_usage: List of application usage data
            
        Returns:
            Tuple[float, Dict[str, int]]: (coding_time_seconds, language_time_dict)
        """
        coding_apps = ["code", "visual studio code", "cursor", "intellij", "pycharm", "webstorm", "vscode"]
        coding_time = 0.0
        languages = {}
        
        for app in app_usage:
            app_lower = app.app_name.lower()
            if any(coding_app in app_lower for coding_app in coding_apps):
                coding_time += app.duration_seconds
                
                # Try to extract language from window titles
                for title in app.window_titles:
                    # Extract file extension from window title
                    file_match = re.search(r'\.([a-zA-Z0-9]+)(?:\s|$)', title)
                    if file_match:
                        ext = file_match.group(1).lower()
                        # Map extensions to languages
                        lang_map = {
                            "py": "Python",
                            "js": "JavaScript",
                            "ts": "TypeScript",
                            "html": "HTML",
                            "css": "CSS",
                            "java": "Java",
                            "c": "C",
                            "cpp": "C++",
                            "cs": "C#",
                            "go": "Go",
                            "rs": "Rust",
                            "rb": "Ruby",
                            "php": "PHP",
                            "swift": "Swift",
                            "kt": "Kotlin",
                            "sql": "SQL",
                            "md": "Markdown",
                            "json": "JSON",
                            "yaml": "YAML",
                            "yml": "YAML",
                            "sh": "Shell",
                            "bat": "Batch",
                            "ps1": "PowerShell"
                        }
                        
                        language = lang_map.get(ext, ext.capitalize())
                        languages[language] = languages.get(language, 0) + 1
        
        return coding_time, languages
    
    def _get_best_practices(self, screenshots: List[ScreenshotModel]) -> Tuple[List[str], List[str]]:
        """Extract best practices followed and violations from screenshots.
        
        Args:
            screenshots: List of screenshot models
            
        Returns:
            Tuple[List[str], List[str]]: (best_practices_followed, best_practices_violations)
        """
        followed = []
        violations = []
        
        for screenshot in screenshots:
            insights = screenshot.get_code_insights()
            if not insights:
                continue
                
            if "best_practices" in insights:
                best_practices = insights["best_practices"]
                
                if "followed" in best_practices and isinstance(best_practices["followed"], list):
                    followed.extend(best_practices["followed"])
                    
                if "violations" in best_practices and isinstance(best_practices["violations"], list):
                    violations.extend(best_practices["violations"])
        
        # Remove duplicates
        followed = list(set(followed))
        violations = list(set(violations))
        
        return followed, violations
    
    def _get_prompts_data(self, prompts: List[PromptModel]) -> Tuple[int, Dict[str, int], Dict[str, int]]:
        """Extract prompt statistics from prompt models.
        
        Args:
            prompts: List of prompt models
            
        Returns:
            Tuple[int, Dict[str, int], Dict[str, int]]: (count, by_tool, by_type)
        """
        count = len(prompts)
        by_tool = {}
        by_type = {}
        
        for prompt in prompts:
            # Count by tool
            tool = prompt.llm_tool_used or "Unknown"
            by_tool[tool] = by_tool.get(tool, 0) + 1
            
            # Count by type
            type_name = prompt.prompt_type.prompt_type_name
            by_type[type_name] = by_type.get(type_name, 0) + 1
        
        return count, by_tool, by_type
    
    def _prepare_report_context(self, start_time: datetime, end_time: datetime) -> ReportContext:
        """Prepare context data for the report.
        
        Args:
            start_time: Start time of the period
            end_time: End time of the period
            
        Returns:
            ReportContext: Context data for the report
        """
        # Get app usage data
        app_usage = self._get_app_usage(start_time, end_time)
        
        # Get active and AFK time
        active_time, afk_time = self._get_active_afk_time(start_time, end_time)
        
        # Get coding time and languages
        coding_time, coding_languages = self._get_coding_time_and_languages(app_usage)
        
        # Get prompts in timeframe
        prompts = self.db_ops.get_prompts_in_timeframe(start_time, end_time)
        prompts_count, prompts_by_tool, prompts_by_type = self._get_prompts_data(prompts)
        
        # Get screenshots in timeframe
        screenshots = self.db_ops.get_screenshots_in_timeframe(start_time, end_time)
        best_practices_followed, best_practices_violations = self._get_best_practices(screenshots)
        
        # Calculate productivity score (simple heuristic)
        total_time = active_time + afk_time
        productivity_score = active_time / total_time if total_time > 0 else 0
        
        # Adjust productivity score based on coding time
        if active_time > 0:
            coding_ratio = coding_time / active_time
            productivity_score = (productivity_score + coding_ratio) / 2
        
        # Cap productivity score at 1.0
        productivity_score = min(productivity_score, 1.0)
        
        # Create report context
        context = ReportContext(
            total_active_time=active_time,
            total_afk_time=afk_time,
            productivity_score=productivity_score,
            app_usage=app_usage,
            prompts_count=prompts_count,
            prompts_by_tool=prompts_by_tool,
            prompts_by_type=prompts_by_type,
            coding_time=coding_time,
            coding_languages=coding_languages,
            best_practices_followed=best_practices_followed,
            best_practices_violations=best_practices_violations,
            screenshots_count=len(screenshots)
        )
        
        return context
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into a human-readable time string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            str: Formatted time string
        """
        # Convert to integer for formatting
        seconds_int = int(seconds)
        hours, remainder = divmod(seconds_int, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _generate_report_with_llm(self, context: ReportContext, start_time: datetime, end_time: datetime) -> str:
        """Generate a report using the OpenAI LLM.
        
        Args:
            context: Report context data
            start_time: Start time of the period
            end_time: End time of the period
            
        Returns:
            str: Generated report text
        """
        try:
            # Format time periods for the prompt
            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format app usage for the prompt
            app_usage_str = ""
            for app in context.app_usage[:10]:  # Limit to top 10 apps
                duration_str = self._format_time(app.duration_seconds)
                app_usage_str += f"- {app.app_name}: {duration_str} ({app.percentage:.1f}%)\n"
            
            # Format coding languages for the prompt
            coding_languages_str = ""
            for lang, count in context.coding_languages.items():
                coding_languages_str += f"- {lang}: {count} occurrences\n"
            
            # Format prompts by tool for the prompt
            prompts_by_tool_str = ""
            for tool, count in context.prompts_by_tool.items():
                prompts_by_tool_str += f"- {tool}: {count}\n"
            
            # Format prompts by type for the prompt
            prompts_by_type_str = ""
            for type_name, count in context.prompts_by_type.items():
                prompts_by_type_str += f"- {type_name}: {count}\n"
            
            # Format best practices for the prompt
            best_practices_followed_str = "\n".join([f"- {practice}" for practice in context.best_practices_followed[:10]])
            best_practices_violations_str = "\n".join([f"- {violation}" for violation in context.best_practices_violations[:10]])
            
            # Check if Cursor is being used
            cursor_usage = context.prompts_by_tool.get("Cursor", 0)
            cursor_usage_str = ""
            if cursor_usage > 0:
                cursor_usage_str = f"\n\n## Cursor Usage:\n- Total Cursor prompts: {cursor_usage}"
                if "cursor_chat" in context.prompts_by_type:
                    cursor_usage_str += f"\n- Cursor chat prompts: {context.prompts_by_type['cursor_chat']}"
            
            # Create the prompt
            prompt = f"""
You are an expert productivity analyst and coding mentor. Generate a detailed hourly report for the user's activity from {start_time_str} to {end_time_str}.

## Activity Data:
- Total active time: {self._format_time(context.total_active_time)}
- Total AFK time: {self._format_time(context.total_afk_time)}
- Productivity score: {context.productivity_score:.2f} (0-1 scale)
- Screenshots captured: {context.screenshots_count}

## Application Usage (Top Apps):
{app_usage_str}

## Coding Activity:
- Total coding time: {self._format_time(context.coding_time)}
- Languages detected:
{coding_languages_str}

## LLM Usage:
- Total prompts: {context.prompts_count}
- Prompts by tool:
{prompts_by_tool_str}
- Prompts by type:
{prompts_by_type_str}{cursor_usage_str}

## Coding Best Practices:
### Followed:
{best_practices_followed_str if context.best_practices_followed else "- No best practices detected"}

### Violations:
{best_practices_violations_str if context.best_practices_violations else "- No violations detected"}

Based on this data, generate a thorough, insightful report with the following sections:
1. Summary of Activity - Overview of the hour's activity
2. Productivity Analysis - Detailed analysis of productivity, including suggestions for improvement
3. Coding Practices - Analysis of coding practices, highlighting strengths and areas for improvement
4. LLM Usage Analysis - How effectively the user is leveraging LLMs, with suggestions for better prompting
5. Recommendations - Specific, actionable recommendations for improving productivity and coding practices

The report should be detailed, insightful, and provide specific, actionable advice. Reference specific data from the context in the report. Use markdown formatting.
"""
            
            # Call the OpenAI API
            logger.info(f"Generating report with {self.model_name}")
            logger.info(f"Prompt: {prompt}")
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert productivity analyst and coding mentor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Extract the report text
            report_text = response.choices[0].message.content
            logger.info(f"Generated report with {len(report_text)} characters")
            
            return report_text
            
        except Exception as e:
            logger.error(f"Error generating report with LLM: {e}")
            return f"Error generating report: {str(e)}"
    
    def generate_hourly_report(self, end_time: Optional[datetime] = None) -> Optional[ReportModel]:
        """Generate an hourly report.
        
        Args:
            end_time: End time of the report period (defaults to now)
            
        Returns:
            Optional[ReportModel]: The generated report model or None if failed
        """
        try:
            # Set end time to now if not provided
            if end_time is None:
                end_time = datetime.now()
            
            # Set start time to one hour before end time
            start_time = end_time - timedelta(hours=1)
            
            logger.info(f"Generating hourly report from {start_time} to {end_time}")
            
            # Get the hourly interval type
            interval_type = self._get_interval_type("hourly")
            
            # Prepare report context
            context = self._prepare_report_context(start_time, end_time)
            
            # Generate report text
            report_text = self._generate_report_with_llm(context, start_time, end_time)
            
            # Create report model
            report = ReportModel.create(
                report_text=report_text,
                interval_type=interval_type,
                timestamp=start_time,
                period_end=end_time
            )
            
            logger.info(f"Created hourly report with ID: {report.report_id}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating hourly report: {e}")
            return None
    
    def close(self):
        """Close the ActivityWatch client connection."""
        try:
            self.aw_client.close()
        except Exception as e:
            logger.error(f"Error closing ActivityWatch client: {e}")
