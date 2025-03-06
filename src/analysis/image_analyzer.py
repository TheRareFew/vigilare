"""Image analysis using gpt-4o-mini."""

import base64
import io
import json
import logging
import os
import atexit
from typing import Dict, Any, Optional, List

from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from pydantic import BaseModel, Field
from openai import OpenAI
from PIL import Image
from src.storage.models import ScreenshotModel
from src.vision.ocr import OCRProcessor
from src.core.config import setup_environment
from src.core.aw_client import ActivityWatchClient

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Enable debug logging

# Create a file handler
log_file = 'image_analyzer.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
    best_practices: BestPracticesOutput = Field(description="Best practices followed and violations")
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
    prompt_text: str = Field(description="The extracted LLM prompt")
    prompt_type: str = Field(description="Type of prompt (programming/research/documentation/other)")
    model_name: str = Field(description="The LLM model used (e.g. GPT-o1, Claude-3.5-sonnet, Claude-3.7-sonnet)")
    llm_tool_used: str = Field(description="The LLM tool used (e.g. Cursor, Windsurf, ChatGPT, Perplexity, Gemini, etc.)")
    confidence: float = Field(description="Confidence in prompt detection (0-1)", ge=0.0, le=1.0)

class ImageAnalysisOutput(BaseModel):
    """Schema for image analysis output."""
    prompts: List[PromptOutput] = Field(description="List of detected LLM prompts")
    full_analysis: str = Field(description="Detailed summary of user context, activities, and overall analysis")
    code_insights: Optional[CodeAnalysisOutput] = Field(description="Code analysis results if code is detected", default=None)

    class Config:
        extra = 'allow'

    def __init__(self, **data):
        logger.debug(f"Initializing ImageAnalysisOutput with data: {json.dumps(data, indent=2)}")
        try:
            # Validate prompt structure
            if "prompts" in data:
                for prompt in data["prompts"]:
                    if not isinstance(prompt, dict):
                        continue
                    # Ensure required fields with correct types
                    prompt["prompt_text"] = str(prompt.get("prompt_text", ""))
                    prompt["prompt_type"] = str(prompt.get("prompt_type", "other"))
                    prompt["model_name"] = str(prompt.get("model_name", "unknown"))
                    prompt["llm_tool_used"] = str(prompt.get("llm_tool_used", "unknown"))
                    prompt["confidence"] = float(prompt.get("confidence", 0.0))
            super().__init__(**data)
            logger.debug("Successfully initialized ImageAnalysisOutput")
        except Exception as e:
            logger.error(f"Error initializing ImageAnalysisOutput: {str(e)}")
            raise

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
            
            # Initialize OCR processor
            self.ocr = OCRProcessor()
            
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=self.api_key)
            
            # Initialize ActivityWatch client with testing=True to use port 5666
            self.aw_client = ActivityWatchClient("vigilare-analyzer", testing=True)
            
            # Store model parameters
            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            
            # Set up output parser
            self.image_parser = PydanticOutputParser(pydantic_object=ImageAnalysisOutput)
            self.code_parser = PydanticOutputParser(pydantic_object=CodeAnalysisOutput)
            
            # Set up chains
            self._setup_chains()
            
            logger.info(f"ImageAnalyzer initialized with model {model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing ImageAnalyzer: {str(e)}")
            raise

    def _setup_chains(self):
        """Set up LangChain chains for analysis."""
        # Code analysis chain
        code_system_template = """You are a system that analyzes code to identify languages, complexity, and best practices.
{format_instructions}"""
        
        code_prompt = ChatPromptTemplate.from_messages([
            ("system", code_system_template),
            ("human", "Analyze this code:\n\n{code}")
        ])
        
        self.code_chain = LLMChain(
            llm=ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=self.api_key,
                callbacks=[LangChainTracer()]
            ),
            prompt=code_prompt,
            output_parser=self.code_parser,
            verbose=True,
            tags=["code-analysis"]
        )
        
        # Image analysis chain
        image_system_template = """You are a system that analyzes screen content and OCR text to identify LLM interactions and other relevant information.
IMPORTANT: When identifying prompts, include ONLY user prompts TO LLMs in the 'prompts' field. DO NOT include LLM-generated responses as prompts.
However, you should still analyze the full image content for the 'full_analysis' field and other structured outputs.
{format_instructions}"""
        
        image_prompt = ChatPromptTemplate.from_messages([
            ("system", image_system_template),
            ("human", "Analyze this content and provide a complete analysis. For the 'prompts' field, include ONLY user prompts sent TO LLMs, not LLM-generated responses:\n\nImage: {image_data}\n\nOCR Text:\n{content}")
        ])
        
        self.image_chain = LLMChain(
            llm=ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self.api_key,
                callbacks=[LangChainTracer()]
            ),
            prompt=image_prompt,
            output_parser=self.image_parser,
            verbose=True,
            tags=["image-analysis"]
        )

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
        """Analyze code content."""
        try:
            logger.info("Analyzing code content")
            
            # Run the code analysis chain with format instructions
            result = self.code_chain.invoke({
                "code": text,
                "format_instructions": self.code_parser.get_format_instructions()
            })
            
            # Parse the result
            try:
                parsed_result = self.code_parser.parse(result)
                logger.info("Successfully parsed code analysis result")
                
                # Convert to dict with proper handling of nested Pydantic models
                result_dict = {
                    "languages": parsed_result.languages,
                    "key_components": parsed_result.key_components,
                    "complexity": {
                        "level": parsed_result.complexity.level,
                        "explanation": parsed_result.complexity.explanation
                    },
                    "potential_issues": parsed_result.potential_issues,
                    "best_practices": {
                        "followed": parsed_result.best_practices.followed,
                        "violations": parsed_result.best_practices.violations
                    },
                    "dependencies": parsed_result.dependencies,
                    "purpose": parsed_result.purpose,
                    "confidence": parsed_result.confidence
                }
                
                return result_dict
            except Exception as e:
                logger.error(f"Error parsing code analysis result: {e}")
                return {
                    "languages": ["unknown"],
                    "key_components": [],
                    "complexity": {
                        "level": "unknown",
                        "explanation": "Error parsing analysis"
                    },
                    "potential_issues": ["Error parsing analysis"],
                    "best_practices": {
                        "followed": [],
                        "violations": []
                    },
                    "dependencies": [],
                    "purpose": "Error parsing analysis",
                    "confidence": 0.0
                }
                
        except Exception as e:
            logger.error(f"Error in code analysis: {e}")
            return {
                "languages": ["unknown"],
                "key_components": [],
                "complexity": {
                    "level": "unknown",
                    "explanation": f"Error: {str(e)}"
                },
                "potential_issues": [f"Error: {str(e)}"],
                "best_practices": {
                    "followed": [],
                    "violations": []
                },
                "dependencies": [],
                "purpose": f"Error: {str(e)}",
                "confidence": 0.0
            }
            
    def _get_current_file_content(self) -> Optional[Dict[str, Any]]:
        """Get content of the currently open file in VSCode/Cursor."""
        try:
            # Get current file info from ActivityWatch
            logger.debug("Attempting to get current VSCode/Cursor file info from ActivityWatch")
            file_info = self.aw_client.get_current_vscode_file()
            
            if not file_info or not file_info.get("file"):
                logger.info("No current file detected in VSCode/Cursor")
                return None
                
            file_path = file_info.get("file")
            language = file_info.get("language", "unknown")
            
            logger.info(f"Found current file: {file_path} (Language: {language})")
            
            # Get file content
            logger.debug(f"Attempting to read content from file: {file_path}")
            content = self.aw_client.get_file_content(file_path)
            
            if not content:
                logger.warning(f"Could not read content from file: {file_path}")
                return None
                
            logger.info(f"Successfully read {len(content)} characters from file")
            return {
                "file_path": file_path,
                "language": language,
                "content": content
            }
                
        except Exception as e:
            logger.error(f"Error getting current file content: {e}", exc_info=True)
            # Continue with OCR-based analysis as fallback
            logger.info("Falling back to OCR-based analysis")
            return None

    def analyze_image(self, image: Image.Image,
                     context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze an image and extract information."""
        try:
            logger.info("Starting image analysis")
            
            # Try to get current file content first
            file_data = self._get_current_file_content()
            
            # Get OCR text as fallback
            logger.debug("Extracting OCR text from image")
            ocr_text = self.ocr.get_text_only(image)
            logger.debug(f"OCR Text extracted (first 500 chars):\n{ocr_text[:500]}...")
            
            # Analyze code if code-like content is detected
            code_analysis = None
            
            # If we have file content, use that for code analysis
            if file_data:
                logger.info(f"Using content from file: {file_data['file_path']}")
                try:
                    code_analysis = self._analyze_code(file_data['content'])
                    logger.debug(f"Code analysis result from file: {json.dumps(code_analysis, indent=2)}")
                except Exception as e:
                    logger.error(f"Error analyzing file content: {e}", exc_info=True)
                    logger.info("Falling back to OCR text for code analysis")
                    code_analysis = None
            
            # Fall back to OCR text for code analysis if needed
            if not code_analysis:
                code_indicators = [
                    "def ", "class ", "import ", "function", "#include", "package",
                    "public class", "const ", "var ", "let ", "fn ", "func "
                ]
                
                if ocr_text and any(indicator in ocr_text for indicator in code_indicators):
                    logger.info("Code-like content detected in OCR text, initiating code analysis")
                    try:
                        code_analysis = self._analyze_code(ocr_text)
                        logger.debug(f"Code analysis result from OCR: {json.dumps(code_analysis, indent=2)}")
                    except Exception as e:
                        logger.error(f"Error analyzing OCR text: {e}", exc_info=True)
                else:
                    logger.info("No code-like content detected")
            
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
                    result = self.image_parser.parse(response.choices[0].message.content)
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
                    result_dict = {
                        "prompts": [],
                        "full_analysis": response.choices[0].message.content,
                        "code_insights": code_analysis
                    }
                    return result_dict
                
            except Exception as e:
                logger.error(f"Error in OpenAI API call: {e}", exc_info=True)
                return {
                    "prompts": [],
                    "full_analysis": f"Error in analysis: {str(e)}",
                    "code_insights": code_analysis
                }
                
        except Exception as e:
            logger.error(f"Error in analyze_image: {str(e)}", exc_info=True)
            return {
                "prompts": [],
                "full_analysis": f"Error analyzing image: {str(e)}",
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
                "full_analysis": f"Error: {str(e)}",
                "code_insights": None
            } 