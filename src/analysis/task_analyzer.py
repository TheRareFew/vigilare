"""Task and activity analysis."""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from src.storage.models import (
    ScreenshotModel, PromptModel, ReportModel,
    AppClassificationModel, IntervalTypeModel
)

logger = logging.getLogger(__name__)

class TaskAnalyzer:
    """Analyzes user tasks and activities."""
    
    def __init__(self):
        """Initialize task analyzer."""
        logger.info("Initialized task analyzer")
    
    def analyze_time_period(self, start_time: datetime,
                          end_time: datetime) -> Dict[str, Any]:
        """Analyze activities in a time period.
        
        Args:
            start_time: Start of the period
            end_time: End of the period
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Get data for the period
            screenshots = (ScreenshotModel
                .select()
                .where(
                    (ScreenshotModel.timestamp >= start_time) &
                    (ScreenshotModel.timestamp <= end_time)
                )
                .order_by(ScreenshotModel.timestamp))
            
            prompts = (PromptModel
                .select()
                .where(
                    (PromptModel.timestamp >= start_time) &
                    (PromptModel.timestamp <= end_time)
                )
                .order_by(PromptModel.timestamp))
            
            # Analyze application usage
            app_usage = self._analyze_app_usage(screenshots)
            
            # Analyze task sequences
            task_sequences = self._analyze_task_sequences(screenshots)
            
            # Analyze prompt patterns
            prompt_patterns = self._analyze_prompt_patterns(prompts)
            
            # Generate summary
            summary = self._generate_period_summary(
                screenshots, prompts, app_usage,
                task_sequences, prompt_patterns
            )
            
            return {
                'period_start': start_time,
                'period_end': end_time,
                'app_usage': app_usage,
                'task_sequences': task_sequences,
                'prompt_patterns': prompt_patterns,
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error analyzing time period: {e}")
            return {
                'error': str(e)
            }
    
    def _analyze_app_usage(self,
                          screenshots: List[ScreenshotModel]) -> List[Dict[str, Any]]:
        """Analyze application usage patterns.
        
        Args:
            screenshots: List of screenshots
            
        Returns:
            List[Dict[str, Any]]: Application usage statistics
        """
        try:
            app_stats = {}
            
            for screenshot in screenshots:
                context = screenshot.get_context()
                app_name = context.get('application_name', 'Unknown')
                
                if app_name not in app_stats:
                    app_stats[app_name] = {
                        'total_time': 0,
                        'session_count': 0,
                        'last_seen': None,
                        'categories': set()
                    }
                
                # Get app classification
                classifications = (AppClassificationModel
                    .select()
                    .where(AppClassificationModel.application_name == app_name)
                    .order_by(AppClassificationModel.confidence.desc()))
                
                for classification in classifications:
                    app_stats[app_name]['categories'].add(classification.category)
                
                # Update stats
                current_time = screenshot.timestamp
                last_seen = app_stats[app_name]['last_seen']
                
                if last_seen:
                    time_diff = (current_time - last_seen).total_seconds()
                    if time_diff <= 300:  # 5 minutes threshold
                        app_stats[app_name]['total_time'] += time_diff
                    else:
                        app_stats[app_name]['session_count'] += 1
                else:
                    app_stats[app_name]['session_count'] = 1
                
                app_stats[app_name]['last_seen'] = current_time
            
            # Format results
            results = []
            for app_name, stats in app_stats.items():
                results.append({
                    'application_name': app_name,
                    'total_time_seconds': stats['total_time'],
                    'session_count': stats['session_count'],
                    'categories': list(stats['categories'])
                })
            
            return sorted(results,
                         key=lambda x: x['total_time_seconds'],
                         reverse=True)
            
        except Exception as e:
            logger.error(f"Error analyzing app usage: {e}")
            return []
    
    def _analyze_task_sequences(self,
                              screenshots: List[ScreenshotModel]) -> List[Dict[str, Any]]:
        """Analyze task sequences from screenshots.
        
        Args:
            screenshots: List of screenshots
            
        Returns:
            List[Dict[str, Any]]: Task sequence analysis
        """
        try:
            sequences = []
            current_sequence = None
            
            for screenshot in screenshots:
                context = screenshot.get_context()
                current_app = context.get('application_name')
                current_title = context.get('window_title', '')
                
                if not current_sequence:
                    current_sequence = {
                        'start_time': screenshot.timestamp,
                        'end_time': screenshot.timestamp,
                        'applications': [current_app],
                        'window_titles': [current_title],
                        'screenshots': [screenshot.image_id]
                    }
                    continue
                
                # Check if this is part of the same sequence
                time_diff = (screenshot.timestamp - current_sequence['end_time']).total_seconds()
                
                if time_diff <= 300:  # 5 minutes threshold
                    # Update current sequence
                    current_sequence['end_time'] = screenshot.timestamp
                    if current_app not in current_sequence['applications']:
                        current_sequence['applications'].append(current_app)
                    current_sequence['window_titles'].append(current_title)
                    current_sequence['screenshots'].append(screenshot.image_id)
                else:
                    # Start new sequence
                    sequences.append(current_sequence)
                    current_sequence = {
                        'start_time': screenshot.timestamp,
                        'end_time': screenshot.timestamp,
                        'applications': [current_app],
                        'window_titles': [current_title],
                        'screenshots': [screenshot.image_id]
                    }
            
            # Add last sequence
            if current_sequence:
                sequences.append(current_sequence)
            
            # Add duration and clean up window titles
            for sequence in sequences:
                sequence['duration_seconds'] = (
                    sequence['end_time'] - sequence['start_time']
                ).total_seconds()
                
                # Keep only unique window titles
                sequence['window_titles'] = list(set(
                    title for title in sequence['window_titles']
                    if title  # Filter out empty titles
                ))
            
            return sequences
            
        except Exception as e:
            logger.error(f"Error analyzing task sequences: {e}")
            return []
    
    def _analyze_prompt_patterns(self,
                               prompts: List[PromptModel]) -> Dict[str, Any]:
        """Analyze patterns in prompt usage.
        
        Args:
            prompts: List of prompts
            
        Returns:
            Dict[str, Any]: Prompt pattern analysis
        """
        try:
            patterns = {
                'type_distribution': {},
                'quality_distribution': {
                    'high': 0,    # > 0.7
                    'medium': 0,  # 0.3 - 0.7
                    'low': 0      # < 0.3
                },
                'model_usage': {},
                'tool_usage': {},
                'hourly_distribution': {str(i): 0 for i in range(24)}
            }
            
            for prompt in prompts:
                # Type distribution
                prompt_type = prompt.prompt_type.prompt_type_name
                patterns['type_distribution'][prompt_type] = (
                    patterns['type_distribution'].get(prompt_type, 0) + 1
                )
                
                # Quality distribution
                if prompt.quality_score:
                    score = float(prompt.quality_score)
                    if score > 0.7:
                        patterns['quality_distribution']['high'] += 1
                    elif score > 0.3:
                        patterns['quality_distribution']['medium'] += 1
                    else:
                        patterns['quality_distribution']['low'] += 1
                
                # Model usage
                if prompt.model_name:
                    patterns['model_usage'][prompt.model_name] = (
                        patterns['model_usage'].get(prompt.model_name, 0) + 1
                    )
                
                # Tool usage
                try:
                    context = json.loads(prompt.context)
                    tool = context.get('llm_tool', 'Unknown')
                    patterns['tool_usage'][tool] = (
                        patterns['tool_usage'].get(tool, 0) + 1
                    )
                except:
                    pass
                
                # Hourly distribution
                hour = str(prompt.timestamp.hour)
                patterns['hourly_distribution'][hour] += 1
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing prompt patterns: {e}")
            return {}
    
    def _generate_period_summary(self, screenshots: List[ScreenshotModel],
                               prompts: List[PromptModel],
                               app_usage: List[Dict[str, Any]],
                               task_sequences: List[Dict[str, Any]],
                               prompt_patterns: Dict[str, Any]) -> str:
        """Generate a summary for the period.
        
        Args:
            screenshots: List of screenshots
            prompts: List of prompts
            app_usage: Application usage statistics
            task_sequences: Task sequence analysis
            prompt_patterns: Prompt pattern analysis
            
        Returns:
            str: Generated summary
        """
        try:
            # Calculate basic stats
            total_time = sum(app['total_time_seconds'] for app in app_usage)
            total_screenshots = len(screenshots)
            total_prompts = len(prompts)
            total_sequences = len(task_sequences)
            
            # Get top applications
            top_apps = sorted(
                app_usage,
                key=lambda x: x['total_time_seconds'],
                reverse=True
            )[:3]
            
            # Get most used prompt types
            top_types = sorted(
                prompt_patterns.get('type_distribution', {}).items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            # Generate summary
            summary = (
                f"During this period, {total_screenshots} screenshots and "
                f"{total_prompts} prompts were captured across {total_sequences} "
                f"task sequences, totaling {total_time/3600:.1f} hours of activity.\n\n"
            )
            
            if top_apps:
                summary += "Top applications used:\n"
                for app in top_apps:
                    hours = app['total_time_seconds'] / 3600
                    summary += (
                        f"- {app['application_name']}: {hours:.1f} hours "
                        f"({app['session_count']} sessions)\n"
                    )
                summary += "\n"
            
            if top_types:
                summary += "Most common prompt types:\n"
                for type_name, count in top_types:
                    summary += f"- {type_name}: {count} prompts\n"
                summary += "\n"
            
            if task_sequences:
                avg_sequence_duration = (
                    sum(s['duration_seconds'] for s in task_sequences) /
                    len(task_sequences)
                )
                summary += (
                    f"Average task sequence duration: "
                    f"{avg_sequence_duration/60:.1f} minutes\n"
                )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating period summary: {e}")
            return "Error generating summary"
    
    def generate_report(self, start_time: datetime,
                       end_time: datetime,
        if interval_type == 'hourly':
            start_time = end_time - timedelta(hours=1)
        elif interval_type == 'daily':
            start_time = end_time - timedelta(days=1)
        elif interval_type == 'weekly':
            start_time = end_time - timedelta(weeks=1)
        elif interval_type == 'monthly':
            start_time = end_time - timedelta(days=30)
        else:  # yearly
            start_time = end_time - timedelta(days=365)
        
        return start_time, end_time

    def analyze_interval(self, interval_type: str,
                        end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Analyze user activity for a time interval.
        
        Args:
            interval_type: Type of interval (hourly, daily, etc.)
            end_time: End time (defaults to now)
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            start_time, end_time = self._get_time_range(interval_type, end_time)
            
            # Get screenshots in interval
            screenshots = (ScreenshotModel
                .select()
                .where(
                    (ScreenshotModel.timestamp >= start_time) &
                    (ScreenshotModel.timestamp <= end_time)
                )
                .order_by(ScreenshotModel.timestamp))
            
            # Get prompts in interval
            prompts = (PromptModel
                .select()
                .where(
                    (PromptModel.timestamp >= start_time) &
                    (PromptModel.timestamp <= end_time)
                )
                .order_by(PromptModel.timestamp))
            
            # Analyze application usage
            app_usage = {}
            for screenshot in screenshots:
                context = screenshot.get_context()
                app_name = context.get('app', 'Unknown')
                app_usage[app_name] = app_usage.get(app_name, 0) + 1
            
            # Analyze prompt usage
            prompt_stats = {
                'total': prompts.count(),
                'by_type': {},
                'by_model': {},
                'average_quality': 0
            }
            
            for prompt in prompts:
                # Count by type
                p_type = prompt.prompt_type.prompt_type_name
                prompt_stats['by_type'][p_type] = prompt_stats['by_type'].get(p_type, 0) + 1
                
                # Count by model
                if prompt.model_name:
                    prompt_stats['by_model'][prompt.model_name] = (
                        prompt_stats['by_model'].get(prompt.model_name, 0) + 1
                    )
                
                # Track quality scores
                if prompt.quality_score is not None:
                    prompt_stats['average_quality'] += float(prompt.quality_score)
            
            if prompt_stats['total'] > 0:
                prompt_stats['average_quality'] /= prompt_stats['total']
            
            return {
                'interval_type': interval_type,
                'start_time': start_time,
                'end_time': end_time,
                'total_screenshots': screenshots.count(),
                'application_usage': app_usage,
                'prompt_statistics': prompt_stats
            }
            
        except Exception as e:
            logger.error(f"Error analyzing interval: {e}")
            return {
                'interval_type': interval_type,
                'start_time': start_time,
                'end_time': end_time,
                'error': str(e)
            }

    def generate_report(self, interval_type: str,
                       end_time: Optional[datetime] = None) -> Optional[int]:
        """Generate a report for a time interval.
        
        Args:
            interval_type: Type of interval (hourly, daily, etc.)
            end_time: End time (defaults to now)
            
        Returns:
            Optional[int]: Report ID if successful, None otherwise
        """
        try:
            # Get analysis
            analysis = self.analyze_interval(interval_type, end_time)
            
            # Format report text
            report_text = f"Activity Report ({interval_type})\n\n"
            report_text += f"Period: {analysis['start_time']} to {analysis['end_time']}\n\n"
            
            # Add application usage
            report_text += "Application Usage:\n"
            for app, count in sorted(
                analysis['application_usage'].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                report_text += f"- {app}: {count} captures\n"
            
            # Add prompt statistics
            stats = analysis['prompt_statistics']
            report_text += f"\nPrompt Usage:\n"
            report_text += f"Total Prompts: {stats['total']}\n"
            report_text += f"Average Quality: {stats['average_quality']:.2f}\n\n"
            
            report_text += "By Type:\n"
            for p_type, count in stats['by_type'].items():
                report_text += f"- {p_type}: {count}\n"
            
            report_text += "\nBy Model:\n"
            for model, count in stats['by_model'].items():
                report_text += f"- {model}: {count}\n"
            
            # Store report
            interval_type_model = self._get_interval_type(interval_type)
            report = ReportModel.create(
                report_text=report_text,
                interval_type=interval_type_model,
                timestamp=analysis['start_time'],
                period_end=analysis['end_time']
            )
            
            logger.info(f"Generated {interval_type} report: {report.report_id}")
            return report.report_id
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None

    def get_productivity_score(self, interval_type: str,
                             end_time: Optional[datetime] = None) -> float:
        """Calculate productivity score for a time interval.
        
        Args:
            interval_type: Type of interval (hourly, daily, etc.)
            end_time: End time (defaults to now)
            
        Returns:
            float: Productivity score (0-1)
        """
        try:
            analysis = self.analyze_interval(interval_type, end_time)
            
            # Factors to consider:
            # 1. Prompt quality scores
            # 2. Number of productive apps used
            # 3. Activity frequency
            
            score = 0.0
            factors = 0
            
            # Factor 1: Prompt quality
            if analysis['prompt_statistics']['total'] > 0:
                score += analysis['prompt_statistics']['average_quality']
                factors += 1
            
            # Factor 2: Productive app usage
            total_captures = analysis['total_screenshots']
            if total_captures > 0:
                productive_apps = 0
                for app, count in analysis['application_usage'].items():
                    try:
                        classification = (AppClassificationModel
                            .select()
                            .where(AppClassificationModel.application_name == app)
                            .get())
                        if classification.category in ['Development', 'Research', 'Documentation']:
                            productive_apps += count
                    except:
                        continue
                
                productivity_ratio = productive_apps / total_captures
                score += productivity_ratio
                factors += 1
            
            # Factor 3: Activity frequency
            expected_captures = {
                'hourly': 30,  # Assuming 2-minute intervals
                'daily': 480,  # Assuming 16 active hours
                'weekly': 2400,  # 5 work days
                'monthly': 9600  # 4 weeks
            }
            
            if interval_type in expected_captures:
                activity_ratio = min(
                    total_captures / expected_captures[interval_type],
                    1.0
                )
                score += activity_ratio
                factors += 1
            
            return score / max(factors, 1)
            
        except Exception as e:
            logger.error(f"Error calculating productivity score: {e}")
            return 0.0 