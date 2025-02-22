"""Image analysis using gpt-4o-mini."""

import base64
import io
import json
import logging
import os
from typing import Dict, Any, Optional

from langchain_openai import ChatOpenAI
from openai import OpenAI
from PIL import Image
from src.storage.models import ScreenshotModel
from src.vision.ocr import OCRProcessor

logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Analyzes screenshots using GPT-4o-mini."""
    
    def __init__(self, model_name: str = "gpt-4o-mini",
                 temperature: float = 0, max_tokens: int = 6900,
                 api_key: Optional[str] = None):
        """Initialize image analyzer.
        
        Args:
            model_name: Name of the GPT-4o-mini model
            temperature: Model temperature (0-1). Set to 0 for consistent, deterministic responses
            max_tokens: Maximum tokens for response. Set to 4096 to allow for detailed analysis
            api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either through api_key parameter "
                "or OPENAI_API_KEY environment variable"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.openai_client = OpenAI(api_key=self.api_key)
        self.code_analyzer = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=self.api_key
        )
        self.ocr = OCRProcessor()
        logger.info(f"Initialized image analyzer with model: {model_name}")

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
            text: Text content from OCR
            
        Returns:
            Dict[str, Any]: Code analysis results
        """
        try:
            # Prepare system message for code analysis
            system_message = """You are an expert code analyzer. Analyze the provided code snippet and extract:
1. Programming languages used
2. Key functions and classes
3. Code complexity assessment
4. Potential bugs or issues
5. Best practices adherence
6. Dependencies and imports
7. Code purpose and functionality

Return a JSON object with these fields:
{
    "languages": ["list of programming languages"],
    "key_components": ["list of important functions/classes"],
    "complexity": {
        "level": "low/medium/high",
        "explanation": "brief explanation"
    },
    "potential_issues": ["list of potential problems"],
    "best_practices": {
        "followed": ["list of followed practices"],
        "violations": ["list of violations"]
    },
    "dependencies": ["list of identified dependencies"],
    "purpose": "brief description of code purpose",
    "confidence": float  # 0-1 indicating confidence in analysis
}"""

            # Get code analysis from GPT-3.5-turbo
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Analyze this code:\n\n{text}"}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0,
                max_tokens=4096
            )
            
            try:
                analysis = json.loads(response.choices[0].message.content)
                logger.debug("Successfully analyzed code content")
                return analysis
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing code analysis response: {e}")
                return {
                    "languages": [],
                    "key_components": [],
                    "complexity": {"level": "unknown", "explanation": "Analysis failed"},
                    "potential_issues": [],
                    "best_practices": {"followed": [], "violations": []},
                    "dependencies": [],
                    "purpose": "Analysis failed",
                    "confidence": 0.0
                }
                
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {
                "languages": [],
                "key_components": [],
                "complexity": {"level": "unknown", "explanation": "Analysis failed"},
                "potential_issues": [],
                "best_practices": {"followed": [], "violations": []},
                "dependencies": [],
                "purpose": f"Error: {str(e)}",
                "confidence": 0.0
            }

    def analyze_image(self, image: Image.Image,
                     context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze an image and extract information.
        
        Args:
            image: PIL Image to analyze
            context: Optional context information (app name, window title)
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Get OCR text to provide additional context
            ocr_text = self.ocr.get_text_only(image)
            logger.debug(f"OCR Text extracted:\n{ocr_text[:500]}...")  # Log first 500 chars of OCR text
            
            # Analyze code if code-like content is detected
            code_analysis = None
            code_indicators = [
                "def ", "class ", "import ", "function", "#include", "package",
                "public class", "const ", "var ", "let ", "fn ", "func "
            ]
            
            # Check for code-like content
            if ocr_text and any(indicator in ocr_text for indicator in code_indicators):
                logger.debug("Code-like content detected, initiating code analysis")
                code_analysis = self._analyze_code(ocr_text)
                logger.debug(f"Code analysis result: {json.dumps(code_analysis, indent=2)}")
            else:
                logger.debug("No code-like content detected")
            
            # Prepare the structured format for response
            structured_format = """
{
    "prompts": [
        {
            "prompt_text": "The extracted LLM prompt",
            "prompt_type": "programming, research, documentation, or other",
            "model_name": "The LLM Model used, if known (e.g., GPT-4, GPT-3.5, Claude-3, etc.)",
            "llm_tool_used": "Cursor, ChatGPT, Claude.ai, etc., or None",
            "confidence": "A float between 0 and 1 indicating confidence in prompt detection"
        }
    ],
    "image_summary": "Detailed summary of the user's work context and activities, including their current task or project focus, any challenges, tools and applications being used, interaction patterns, apparent workflow stages, and other relevant insights. Be extremely specific, detailed, and thorough.",
    "code_insights": {}
}
"""
            
            # Combine OCR text and code analysis into the context
            additional_context = ""
            if ocr_text:
                additional_context += f"OCR Text from image:\n{ocr_text}\n\n"
            if code_analysis:
                additional_context += f"Code Analysis:\n{json.dumps(code_analysis, indent=2)}\n\n"
            
            # Create the system prompt
            system_prompt = (
                "You are a JSON-only response system that analyzes screen content and OCR text to identify LLM interactions. "
                "YOU MUST ONLY RETURN VALID JSON matching this exact structure, with no additional text or explanation:\n\n"
                "{\n"
                '    "prompts": [\n'
                "        {\n"
                '            "prompt_text": "string - the extracted LLM prompt",\n'
                '            "prompt_type": "string - programming/research/documentation/other",\n'
                '            "model_name": "string - GPT-4/GPT-3.5/Claude/etc or null if unknown",\n'
                '            "llm_tool_used": "string - Cursor/ChatGPT/Claude.ai/etc or null if unknown",\n'
                '            "confidence": "number - float between 0 and 1"\n'
                "        }\n"
                "    ],\n"
                '    "image_summary": "string - detailed summary of user context and activities",\n'
                '    "code_insights": {}\n'
                "}\n\n"
                "Key points to analyze:\n"
                "- Text input areas and chat interfaces; 'Ask agent to do anything' is not a prompt\n"
                "- Code editors with LLM integrations\n"
                "- Web-based and desktop LLM interfaces\n"
                "- Common LLMs: GPT-4, GPT-3.5, Claude-3.5-sonnet, Claude-2, Gemini, Llama 2, Mixtral, DeepSeek, Grok\n"
                "- Common interfaces: ChatGPT, Claude.ai, Perplexity, Poe, Cursor, Gemini, DeepSeek Chat, Grok.x\n\n"
                "If no prompts are found, return an empty prompts array. Never include explanatory text outside the JSON structure."
            )
            
            # Construct the messages for the OpenAI API - single message with all context
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this content and return ONLY valid JSON:\n\n{additional_context}"}
            ]
            
            # Get OpenAI response
            logger.info("Sending request to OpenAI API...")
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                logger.info(f"Received response from OpenAI API. Model: {self.model_name}")
                logger.info(f"Raw response content: {response.choices[0].message.content}")
            except Exception as e:
                logger.error(f"Failed to get response from OpenAI API: {str(e)}")
                return {
                    "prompts": [],
                    "image_summary": f"OpenAI API error: {str(e)}",
                    "code_insights": code_analysis if code_analysis else {}
                }
            
            # Parse response
            try:
                # First try to parse as JSON
                response_text = response.choices[0].message.content.strip()
                logger.info("Attempting to parse response as JSON...")
                try:
                    analysis = json.loads(response_text)
                    logger.info("Successfully parsed response as JSON")
                except json.JSONDecodeError:
                    # If not valid JSON, wrap the text response in our expected structure
                    logger.warning("Response was not JSON, wrapping in standard format")
                    analysis = {
                        "prompts": [],
                        "image_summary": response_text,
                        "code_insights": {}
                    }
                
                # Add code analysis if available
                if code_analysis:
                    logger.debug("Adding code analysis to final result")
                    analysis["code_insights"] = code_analysis
                else:
                    logger.debug("No code analysis available to add to result")
                
                # Validate and ensure required fields
                if "prompts" not in analysis:
                    analysis["prompts"] = []
                if "image_summary" not in analysis:
                    analysis["image_summary"] = "No summary available"
                if "code_insights" not in analysis:
                    analysis["code_insights"] = {}
                
                logger.debug("Successfully analyzed image")
                logger.debug(f"Final analysis result: {json.dumps(analysis, indent=2)}")
                return analysis
                
            except Exception as e:
                logger.error(f"Error processing OpenAI response: {e}")
                return {
                    "prompts": [],
                    "image_summary": f"Error processing analysis: {str(e)}",
                    "code_insights": code_analysis if code_analysis else {}
                }
                
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return {
                "prompts": [],
                "image_summary": f"Error analyzing image: {str(e)}",
                "code_insights": code_analysis if code_analysis else {}
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
            
            # Analyze image
            analysis = self.analyze_image(image)
            
            # Update screenshot with analysis
            screenshot.image_summary = analysis.get('image_summary', '')
            screenshot.save()
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing screenshot: {e}")
            return {
                "prompts": [],
                "image_summary": f"Error: {str(e)}"
            } 