"""CLI command for generating reports."""

import logging
import click
from datetime import datetime, timedelta

from src.analysis.report_generator import ReportGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)

@click.group()
def report():
    """Report generation commands."""
    pass

@report.command()
@click.option('--hours', default=1, help='Number of hours to look back')
def generate(hours):
    """Generate a productivity report."""
    try:
        logger.info(f"Generating report for the last {hours} hour(s)")
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Initialize report generator
        generator = ReportGenerator()
        
        # Generate report
        report = generator.generate_hourly_report(end_time)
        
        if report:
            logger.info(f"Report generated successfully with ID: {report.report_id}")
            click.echo(f"Report generated successfully with ID: {report.report_id}")
            
            # Print a preview of the report
            preview_length = min(500, len(report.report_text))
            preview = report.report_text[:preview_length] + "..." if len(report.report_text) > preview_length else report.report_text
            click.echo("\nReport Preview:")
            click.echo("=" * 80)
            click.echo(preview)
            click.echo("=" * 80)
        else:
            logger.error("Failed to generate report")
            click.echo("Failed to generate report")
        
        # Close the generator
        generator.close()
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        click.echo(f"Error generating report: {e}")

if __name__ == '__main__':
    report() 