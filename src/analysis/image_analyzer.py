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

# Set up logging
logger = logging.getLogger(__name__)

def setup_logger():
    """Setup module logger with proper handlers and cleanup."""
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Register cleanup
    def cleanup():
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    
    atexit.register(cleanup)

setup_logger()

class CodeAnalysisOutput(BaseModel):
    """Schema for code analysis output."""
    languages: List[str] = Field(description="List of programming languages")
    key_components: List[str] = Field(description="List of important functions/classes")
    complexity: Dict[str, str] = Field(description="Complexity assessment with level and explanation")
    potential_issues: List[str] = Field(description="List of potential problems")
    best_practices: Dict[str, List[str]] = Field(description="Best practices followed and violations")
    dependencies: List[str] = Field(description="List of identified dependencies")
    purpose: str = Field(description="Brief description of code purpose")
    confidence: float = Field(description="Confidence in analysis (0-1)")

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

class ImageAnalysisOutput(BaseModel):
    """Schema for image analysis output."""
    prompts: List[Dict[str, Any]] = Field(description="List of detected LLM prompts")
    image_summary: str = Field(description="Detailed summary of user context and activities")
    code_insights: Dict[str, Any] = Field(description="Code analysis results if code is detected")

    class Config:
        extra = 'allow'

    def __init__(self, **data):
        logger.debug(f"Initializing ImageAnalysisOutput with data: {json.dumps(data, indent=2)}")
        try:
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
                raise ValueError("OpenAI API key not found in environment")

            # Setup LangSmith tracing
            logger.debug("Setting up LangSmith tracing")
            try:
                tracer = LangChainTracer()
                callback_manager = CallbackManager([tracer])
            except Exception as e:
                logger.error(f"Failed to initialize LangChain tracer: {str(e)}", exc_info=True)
                tracer = None
                callback_manager = None

            self.model_name = model_name
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.openai_client = OpenAI(api_key=self.api_key)
            
            logger.debug("Initializing LangChain models")
            # Initialize LangChain models with tracing
            self.code_analyzer = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=self.api_key,
                callbacks=[tracer] if tracer else None
            )
            
            self.image_analyzer_llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=self.api_key,
                callbacks=[tracer] if tracer else None
            )
            
            logger.debug("Creating output parsers")
            # Initialize output parsers
            try:
                logger.debug("Creating code parser")
                self.code_parser = PydanticOutputParser(pydantic_object=CodeAnalysisOutput)
                logger.debug("Successfully created code parser")
            except Exception as e:
                logger.error(f"Error creating code parser: {str(e)}", exc_info=True)
                raise
                
            try:
                logger.debug("Creating image parser")
                self.image_parser = PydanticOutputParser(pydantic_object=ImageAnalysisOutput)
                logger.debug("Successfully created image parser")
            except Exception as e:
                logger.error(f"Error creating image parser: {str(e)}", exc_info=True)
                raise
            
            # Setup chains
            logger.debug("Setting up LangChain chains")
            self._setup_chains()
            
            self.ocr = OCRProcessor()
            logger.info(f"Successfully initialized image analyzer with model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error during ImageAnalyzer initialization: {str(e)}", exc_info=True)
            raise

    def _setup_chains(self):
        """Setup LangChain chains for analysis."""
        # Code analysis chain
        code_system_template = """You are an expert code analyzer. Analyze the provided code snippet and extract structured information.
{format_instructions}"""
        
        code_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(code_system_template),
                HumanMessagePromptTemplate.from_template("Analyze this code:\n\n{code}")
            ]
        )
        
        self.code_chain = LLMChain(
            llm=self.code_analyzer,
            prompt=code_prompt,
            output_parser=self.code_parser,
            verbose=True,
            tags=["code-analysis"]
        )
        
        # Image analysis chain
        image_system_template = """You are a system that analyzes screen content and OCR text to identify LLM interactions.
{format_instructions}"""
        
        image_prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template(image_system_template),
                HumanMessagePromptTemplate.from_template("Analyze this content:\n\nImage: {image_data}\n\nOCR Text:\n{content}")
            ]
        )
        
        self.image_chain = LLMChain(
            llm=self.image_analyzer_llm,
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
        """Analyze code content using GPT-3.5-turbo.
        
        Args:
            text: Text content from OCR
            
        Returns:
            Dict[str, Any]: Code analysis results
        """
        try:
            # Run analysis through LangChain
            result = self.code_chain.run(
                code=text,
                format_instructions=self.code_parser.get_format_instructions()
            )
            logger.debug("Successfully analyzed code content")
            return result.dict()
                
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return CodeAnalysisOutput(
                languages=[],
                key_components=[],
                complexity={"level": "unknown", "explanation": f"Error: {str(e)}"},
                potential_issues=[],
                best_practices={"followed": [], "violations": []},
                dependencies=[],
                purpose=f"Error: {str(e)}",
                confidence=0.0
            ).dict()

    def analyze_image(self, image: Image.Image,
                     context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Analyze an image and extract information."""
        try:
            # Get OCR text
            ocr_text = self.ocr.get_text_only(image)
            logger.debug(f"OCR Text extracted:\n{ocr_text[:500]}...")
            
            # Analyze code if code-like content is detected
            code_analysis = None
            code_indicators = [
                "def ", "class ", "import ", "function", "#include", "package",
                "public class", "const ", "var ", "let ", "fn ", "func "
            ]
            
            if ocr_text and any(indicator in ocr_text for indicator in code_indicators):
                logger.debug("Code-like content detected, initiating code analysis")
                code_analysis = self._analyze_code(ocr_text)
                logger.debug(f"Code analysis result: {json.dumps(code_analysis, indent=2)}")
            else:
                logger.debug("No code-like content detected")
            
            # Encode image to base64
            base64_image = self._encode_image(image)
            
            # Prepare system message with format instructions
            system_message = {
                "role": "system",
                "content": "You are a system that analyzes screen content to identify LLM interactions. " + 
                          self.image_parser.get_format_instructions()
            }
            
            # Prepare user message with image and context
            context_text = f"\nContext: {json.dumps(context)}" if context else ""
            user_message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Analyze this screenshot and identify any LLM interactions, prompts, or relevant activities.{context_text}"
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
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[system_message, user_message],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Parse the response
                try:
                    result = self.image_parser.parse(response.choices[0].message.content)
                except Exception as e:
                    logger.error(f"Error parsing image analysis response: {e}")
                    result = ImageAnalysisOutput(
                        prompts=[],
                        image_summary=response.choices[0].message.content,
                        code_insights={}
                    )
                
                # Add code analysis if available
                if code_analysis:
                    result.code_insights = code_analysis
                
                logger.debug(f"Final analysis result: {json.dumps(result.dict(), indent=2)}")
                return result.dict()
                
            except Exception as e:
                logger.error(f"Error in image analysis: {e}")
                return ImageAnalysisOutput(
                    prompts=[],
                    image_summary=f"Error in analysis: {str(e)}",
                    code_insights=code_analysis if code_analysis else {}
                ).dict()
                
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return ImageAnalysisOutput(
                prompts=[],
                image_summary=f"Error analyzing image: {str(e)}",
                code_insights={}
            ).dict()

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