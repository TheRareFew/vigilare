"""OCR functionality using docTR."""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR processing using docTR."""
    
    def __init__(self, model_name: str = "doctr"):
        """Initialize OCR processor.
        
        Args:
            model_name: Name of the OCR model to use
        """
        try:
            self.model = ocr_predictor(pretrained=True)
            logger.info(f"Initialized OCR processor with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing OCR model: {e}")
            raise

    def process_image(self, image: Image.Image) -> Optional[Dict[str, Any]]:
        """Process an image and extract text with positions.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Optional[Dict[str, Any]]: Extracted text and positions or None if failed
        """
        try:
            # Convert image to RGB if it's RGBA
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Log image properties
            logger.debug(f"Processing image of size {img_array.shape}")
            
            # Perform OCR
            result = self.model([img_array])
            
            # Extract text and positions
            extracted_data = []
            full_text = []
            
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        # Get text from words
                        line_words = [word.value for word in line.words]
                        line_text = " ".join(line_words)
                        
                        if line_text.strip():  # Only process non-empty lines
                            full_text.append(line_text)
                            
                            # Get coordinates
                            coords = line.geometry
                            
                            # Store both the text and coordinates
                            extracted_data.append({
                                'text': line_text,
                                'coordinates': {
                                    'x1': float(coords[0][0]),
                                    'y1': float(coords[0][1]),
                                    'x2': float(coords[1][0]),
                                    'y2': float(coords[1][1])
                                }
                            })
                            logger.debug(f"Extracted text: '{line_text}' at {coords}")
            
            result = {
                'text': " ".join(full_text),
                'lines': extracted_data
            }
            
            logger.info(f"Extracted {len(extracted_data)} text regions")
            logger.debug(f"Full text: {result['text'][:200]}...")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image with OCR: {e}")
            logger.exception(e)
            return None

    def get_text_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Get regions containing text from an image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List[Dict[str, Any]]: List of text regions with coordinates
        """
        try:
            result = self.process_image(image)
            if result:
                return result['lines']
            return []
            
        except Exception as e:
            logger.error(f"Error getting text regions: {e}")
            logger.exception(e)
            return []

    def get_text_only(self, image: Image.Image) -> str:
        """Extract text only from an image.
        
        Args:
            image: PIL Image to process
            
        Returns:
            str: Extracted text
        """
        try:
            result = self.process_image(image)
            if result:
                return result['text']
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            logger.exception(e)
            return "" 