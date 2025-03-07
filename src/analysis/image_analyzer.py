"""Image analysis using gpt-3.5-turbo for code and gpt-4o-mini for screenshots."""

import base64
import io
import json
import os
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from openai import OpenAI
from PIL import Image
from src.storage.models import ScreenshotModel
from src.core.config import setup_environment
from src.core.aw_client import ActivityWatchClient
from src.utils.logger import get_logger

# Get the application logger
logger = get_logger(__name__)

class ComplexityOutput(BaseModel):
    """Schema for complexity assessment."""
    level: str = Field(description="Complexity level (low/medium/high)")
    explanation: str = Field(description="Brief explanation of complexity assessment")

class BestPracticesOutput(BaseModel):
    """Schema for best practices assessment."""
    followed: List[str] = Field(description="List of followed best practices")
    violations: List[str] = Field(description="List of best practice violations")

class CodeAnalysisOutput(BaseModel):
    """Schema for code analysis output."""
    languages: List[str] = Field(description="List of programming languages")
    key_components: List[str] = Field(description="List of important functions/classes")
    complexity: ComplexityOutput = Field(description="Complexity assessment with level and explanation")
    potential_issues: List[str] = Field(description="List of potential problems")
    best_practices: BestPracticesOutput = Field(description="Detailed Best practices followed and violations")
    dependencies: List[str] = Field(description="List of identified dependencies")
    purpose: str = Field(description="Brief description of code purpose")
    confidence: float = Field(description="Confidence in analysis (0-1)", ge=0.0, le=1.0)

    class Config:
        extra = 'allow'

    def __init__(self, **data):
        logger.debug(f"Initializing CodeAnalysisOutput with data: {json.dumps(data, indent=2)}")
        try:
            super().__init__(**data)
            logger.debug("Successfully initialized CodeAnalysisOutput")
        except Exception as e:
            logger.error(f"Error initializing CodeAnalysisOutput: {str(e)}")
            raise

class PromptOutput(BaseModel):
    """Schema for individual prompt output."""
    prompt_text: str = Field(description="The extracted LLM prompt", default="Unknown prompt")
    prompt_type: str = Field(description="Type of prompt (programming/research/documentation/other)", default="other")
    model_name: str = Field(description="The LLM model used (e.g. GPT-o1, Claude-3.5-sonnet, Claude-3.7-sonnet)", default="unknown")
    llm_tool_used: str = Field(description="The LLM tool used (e.g. Cursor, Windsurf, ChatGPT, Perplexity, Gemini, etc.)", default="unknown")
    confidence: float = Field(description="Confidence in prompt detection (0-1)", ge=0.0, le=1.0, default=0.0)
    
    class Config:
        extra = 'allow'

class ImageAnalysisOutput(BaseModel):
    """Schema for image analysis output."""
    prompts: List[PromptOutput] = Field(description="List of detected LLM prompts", default_factory=list)
    full_analysis: str = Field(description="Detailed summary of user context, activities, and overall analysis", default="No analysis available")
    code_insights: Optional[CodeAnalysisOutput] = Field(description="Code analysis results if code is detected", default=None)

    class Config:
        extra = 'allow'

    def __init__(self, **data):
        logger.debug(f"Initializing ImageAnalysisOutput with data: {json.dumps(data, indent=2)}")
        try:
            # Validate prompt structure
            if "prompts" in data:
                # Filter out non-dictionary entries
                filtered_prompts = []
                for prompt in data["prompts"]:
                    if isinstance(prompt, dict):
                        # Ensure all required fields are present
                        if "prompt_text" not in prompt:
                            prompt["prompt_text"] = "Unknown prompt"
                        if "prompt_type" not in prompt:
                            prompt["prompt_type"] = "other"
                        if "model_name" not in prompt:
                            prompt["model_name"] = "unknown"
                        if "llm_tool_used" not in prompt:
                            prompt["llm_tool_used"] = "unknown"
                        if "confidence" not in prompt:
                            prompt["confidence"] = 0.0
                        filtered_prompts.append(prompt)
                data["prompts"] = filtered_prompts
            
            # Ensure full_analysis is present
            if "full_analysis" not in data:
                data["full_analysis"] = "No analysis available"
                
            super().__init__(**data)
        except Exception as e:
            logger.error(f"Error initializing ImageAnalysisOutput: {e}")
            # Create a minimal valid instance
            super().__init__(
                prompts=[],
                full_analysis=f"Error initializing output: {str(e)}",
                code_insights=None
            )

class ImageAnalyzer:
    """Analyzes screenshots using GPT-4o-mini."""
    
    def __init__(self, model_name: str = "gpt-4o-mini",
                 temperature: float = 0, max_tokens: int = 6900,
                 api_key: Optional[str] = None):
        """Initialize image analyzer."""
        logger.debug("Initializing ImageAnalyzer")
        try:
            # Setup environment variables
            setup_environment()
            
            # Use the configured OpenAI key
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key not found")
            
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=self.api_key)
            
            # Initialize ActivityWatch client with testing=True to use port 5666
            self.aw_client = ActivityWatchClient("vigilare-analyzer", testing=True)
            
            # Store model parameters
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            
            # Set up output parsers
            self._setup_chains()
            
            logger.info(f"ImageAnalyzer initialized with model {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {str(e)}")
            raise

    def _setup_chains(self):
        """Set up LangChain chains for analysis."""
        try:
            logger.debug("Setting up LangChain parsers")
            
            # Set up parsers
            self.image_parser = PydanticOutputParser(pydantic_object=ImageAnalysisOutput)
            self.code_parser = PydanticOutputParser(pydantic_object=CodeAnalysisOutput)
            
            logger.debug("LangChain parsers set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up LangChain chains: {e}")
            raise

    def _encode_image(self, image: Image.Image) -> str:
        """Encode image to base64 string.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            str: Base64 encoded image
        """
        buffered = io.BytesIO()
        # Create a copy of the image and convert to RGB if needed
        img_copy = image.copy()
        if img_copy.mode == 'RGBA':
            img_copy = img_copy.convert('RGB')
        img_copy.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _analyze_code(self, text: str) -> Dict[str, Any]:
        """Analyze code content using GPT-3.5-turbo.
        
        Args:
            text: Code content to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            logger.info("Analyzing code with GPT-3.5-turbo")
            
            # Prepare system message with format instructions
            system_message = {
                "role": "system",
                "content": "You are a code analysis system that examines code to identify key components, complexity, potential issues, and best practices. " +
                          "Provide a detailed analysis of the code following the specified output format. " +
                          "In addition to analyzing the code structure, also assess the user's programming competency based on the code quality, patterns, and techniques used. " +
                          "For the full_analysis field, provide a comprehensive assessment that includes: " +
                          "1. A detailed summary of the file's purpose and functionality " +
                          "2. An evaluation of the code quality, organization, and readability " +
                          "3. An assessment of the user's programming competency level (beginner, intermediate, advanced) with specific examples from the code " +
                          "4. Suggestions for improvement that match the user's apparent skill level " +
                          self.code_parser.get_format_instructions()
            }
            
            # Prepare user message with code content
            user_message = {
                "role": "user",
                "content": f"Analyze this code and provide a complete analysis including an assessment of the programmer's competency level based on the code quality, patterns, and techniques used:\n\n```\n{text}\n```"
            }
            
            # Run analysis through OpenAI client with GPT-3.5-turbo
            logger.debug("Sending code to GPT-3.5-turbo for analysis")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use GPT-3.5-turbo instead of GPT-4o-mini
                messages=[system_message, user_message],
                temperature=0,
                max_tokens=2048
            )
            
            logger.info("Received response from GPT-3.5-turbo")
            
            # Parse the response
            content = response.choices[0].message.content
            
            # Format the response to ensure it has the required fields
            formatted_content = self._format_code_analysis_response(content)
            
            # Parse the content
            try:
                result = self.code_parser.parse(formatted_content)
                logger.info("Successfully parsed code analysis response")
                
                # Convert to dict to avoid JSON serialization issues
                return result.dict()
            except Exception as parse_error:
                logger.error(f"Error parsing code analysis response: {parse_error}", exc_info=True)
                
                # Try to extract JSON directly
                try:
                    json_content = json.loads(formatted_content)
                    logger.info("Successfully extracted JSON content directly")
                    return json_content
                except Exception as json_error:
                    logger.error(f"Error extracting JSON content: {json_error}")
                    
                    # Create a fallback response
                    return {
                        "languages": ["unknown"],
                        "key_components": ["Error parsing response"],
                        "complexity": {"level": "unknown", "explanation": "Error parsing response"},
                        "potential_issues": ["Error parsing response"],
                        "best_practices": {"followed": [], "violations": ["Error parsing response"]},
                        "dependencies": [],
                        "purpose": f"Error parsing response: {str(parse_error)}",
                        "confidence": 0.0
                    }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}", exc_info=True)
            return {
                "languages": [],
                "key_components": [],
                "complexity": {"level": "unknown", "explanation": f"Error: {str(e)}"},
                "potential_issues": [],
                "best_practices": {"followed": [], "violations": []},
                "dependencies": [],
                "purpose": f"Error analyzing code: {str(e)}",
                "confidence": 0.0
            }

    def _get_current_file_content(self) -> Optional[Dict[str, Any]]:
        """Get content of the currently open file in Cursor."""
        try:
            # Get current file info from Cursor
            logger.debug("Attempting to get current Cursor file info")
            file_info = self.aw_client.get_current_cursor_file()
            
            if not file_info:
                logger.info("No current file detected in Cursor")
                return None
                
            return file_info
                
        except Exception as e:
            logger.error(f"Error getting current file content: {e}", exc_info=True)
            logger.info("No active file content available")
            return None

    def analyze_image(self, image: Image.Image,
                     context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze an image and extract information."""
        try:
            logger.info("Starting image analysis")
            
            # Try to get current file content
            file_data = self._get_current_file_content()
            
            # Analyze code if file content is available
            code_analysis = None
            
            # If we have file content, use that for code analysis
            if file_data:
                logger.info(f"Using content from file: {file_data['file_path']}")
                try:
                    code_analysis = self._analyze_code(file_data['content'])
                    logger.debug(f"Code analysis result from file: {json.dumps(code_analysis, indent=2)}")
                except Exception as e:
                    logger.error(f"Error analyzing file content: {e}", exc_info=True)
                    code_analysis = None
            else:
                logger.info("No file content available for code analysis")
            
            # Encode image to base64
            logger.debug("Encoding image to base64")
            base64_image = self._encode_image(image)
            logger.debug("Image successfully encoded to base64")
            
            # Prepare system message with format instructions
            logger.debug("Preparing messages for OpenAI API")
            system_message = {
                "role": "system",
                "content": "You are a system that analyzes screen content to identify LLM interactions and other relevant information. " + 
                          "IMPORTANT: When identifying prompts, include ONLY user prompts TO LLMs in the 'prompts' field. DO NOT include LLM-generated responses as prompts. " +
                          "However, you should still analyze the full image content for the 'full_analysis' field and other structured outputs. " +
                          self.image_parser.get_format_instructions()
            }
            
            # Prepare user message with image and context
            context_text = f"\nContext: {json.dumps(context)}" if context else ""
            file_context = ""
            if file_data:
                file_context = f"\nCurrent file: {file_data['file_path']} (Language: {file_data['language']})"
                
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this screenshot and provide a complete analysis. For the 'prompts' field, include ONLY user prompts sent TO LLMs, not LLM-generated responses.{context_text}{file_context}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
            
            # Run analysis through OpenAI client directly
            try:
                logger.debug("Sending request to OpenAI API")
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                logger.info("Received response from OpenAI")
                logger.debug(f"Raw API response: {response}")
                
                # Parse the response
                try:
                    logger.debug("Parsing API response")
                    content = response.choices[0].message.content
                    
                    # Try to extract JSON from the response if it's not properly formatted
                    try:
                        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                        if json_match:
                            logger.info("Found JSON in code block, extracting it")
                            content = json_match.group(1).strip()
                    except Exception as e:
                        logger.warning(f"Error extracting JSON from response: {e}")
                    
                    # Format the response to ensure it has the required fields
                    formatted_content = self._format_code_analysis_response(content)
                    
                    result = self.image_parser.parse(formatted_content)
                    logger.info("Successfully parsed API response")
                    
                    # Convert to dict to avoid JSON serialization issues
                    result_dict = result.dict()
                    
                    # Add code analysis if available
                    if code_analysis:
                        logger.debug("Adding code analysis to result")
                        result_dict["code_insights"] = code_analysis
                    
                    logger.debug(f"Final analysis result: {json.dumps(result_dict, indent=2)}")
                    return result_dict
                    
                except Exception as e:
                    logger.error(f"Error parsing image analysis response: {e}", exc_info=True)
                    # Create a safe fallback response
                    content = response.choices[0].message.content
                    
                    # Try to extract any potential prompts using regex
                    prompts = []
                    try:
                        # Look for patterns that might indicate prompts in the text
                        prompt_matches = re.findall(r'"prompt_text"\s*:\s*"([^"]+)"', content)
                        for i, match in enumerate(prompt_matches):
                            prompts.append({
                                "prompt_text": match,
                                "prompt_type": "other",
                                "model_name": "unknown",
                                "llm_tool_used": "unknown",
                                "confidence": 0.0
                            })
                    except Exception as extract_err:
                        logger.error(f"Error extracting prompts from response: {extract_err}")
                    
                    # Ensure we have a full_analysis field
                    full_analysis = content
                    if len(full_analysis) > 1000:
                        # Truncate very long content for the full_analysis field
                        full_analysis = full_analysis[:1000] + "... (truncated)"
                    
                    result_dict = {
                        "prompts": prompts,
                        "full_analysis": full_analysis,
                        "code_insights": code_analysis
                    }
                    return result_dict
                
            except Exception as e:
                logger.error(f"Error in OpenAI API call: {e}", exc_info=True)
                return {
                    "prompts": [],
                    "full_analysis": f"Error in analysis: {str(e)}. Unable to provide a detailed assessment of the code and user competency at this time.",
                    "code_insights": code_analysis
                }
                
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}", exc_info=True)
            return {
                "prompts": [],
                "full_analysis": f"Error analyzing image: {str(e)}. The system encountered an issue while attempting to analyze the code and assess user competency.",
                "code_insights": None
            }

    def analyze_screenshot(self, screenshot: ScreenshotModel) -> Dict[str, Any]:
        """Analyze a screenshot from the database.
        
        Args:
            screenshot: Screenshot model instance
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Load image
            image = Image.open(screenshot.image_path)
            
            # Get context from screenshot metadata
            context = {
                "timestamp": screenshot.timestamp.isoformat() if screenshot.timestamp else None,
                "window_title": screenshot.window_title or "",
                "app_name": screenshot.app_name or ""
            }
            
            # Analyze image with context
            analysis = self.analyze_image(image, context=context)
            
            # Update screenshot with analysis
            screenshot.image_summary = analysis.get('full_analysis', '')
            screenshot.save()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot: {e}", exc_info=True)
            return {
                "prompts": [],
                "full_analysis": f"Error: {str(e)}. The system was unable to analyze this screenshot to assess code quality and user competency.",
                "code_insights": None
            }

    def _format_code_analysis_response(self, content: str) -> str:
        """Format the code analysis response to ensure it has the required fields.
        
        Args:
            content: The response content from GPT-3.5-turbo
            
        Returns:
            str: Formatted response content
        """
        try:
            # Try to parse as JSON
            try:
                json_content = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from the response if it's not properly formatted
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if json_match:
                    content = json_match.group(1).strip()
                    json_content = json.loads(content)
                else:
                    # If we can't parse as JSON, return a default response with empty prompts
                    return json.dumps({
                        "prompts": [],
                        "full_analysis": "Unable to parse code analysis response. The file may be too complex or contain syntax errors that prevented proper analysis."
                    })
            
            # Ensure the response has the required fields
            if "prompts" not in json_content:
                json_content["prompts"] = []  # Empty array - no prompts detected
            
            if "full_analysis" not in json_content:
                json_content["full_analysis"] = "This file appears to contain code that requires analysis. A detailed assessment of the code structure, quality, and the user's programming competency would normally appear here. Please check the code_insights field for technical details about the code."
            
            return json.dumps(json_content)
            
        except Exception as e:
            logger.error(f"Error formatting code analysis response: {e}", exc_info=True)
            # Return a default response with empty prompts
            return json.dumps({
                "prompts": [],
                "full_analysis": "An error occurred during code analysis. The system was unable to properly analyze this file's content and assess the user's programming competency."
            }) 