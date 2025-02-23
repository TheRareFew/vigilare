"""Sensitive information blurring functionality."""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple

import cv2
from PIL import Image

from src.vision.ocr import OCRProcessor
from src.vision.ner import NERProcessor

logger = logging.getLogger(__name__)

class SensitiveInfoBlur:
    """Handles detection and blurring of sensitive information."""
    
    def __init__(self, blur_method: str = "gaussian", blur_intensity: int = 31):
        """Initialize sensitive information blurring.
        
        Args:
            blur_method: Method to use for blurring ('gaussian' or 'pixelate')
            blur_intensity: Intensity of the blur effect. Must be odd and > 0 for gaussian.
        """
        self.blur_method = blur_method
        # Ensure blur_intensity is odd and > 0 for gaussian blur
        if blur_method == "gaussian":
            self.blur_intensity = max(3, blur_intensity + (blur_intensity % 2 == 0))
        else:
            self.blur_intensity = blur_intensity
        self.ocr = OCRProcessor()
        self.ner = NERProcessor()
        logger.info(f"Initialized sensitive info blurring with method: {blur_method}, intensity: {self.blur_intensity}")

    def _apply_gaussian_blur(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply Gaussian blur to a region of an image.
        
        Args:
            image: Image as numpy array
            region: Region to blur (x, y, width, height)
            
        Returns:
            np.ndarray: Image with blur applied
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        kernel_size = (self.blur_intensity, self.blur_intensity)
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
        image[y:y+h, x:x+w] = blurred_roi
        return image

    def _apply_pixelation(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply pixelation to a region of an image.
        
        Args:
            image: Image as numpy array
            region: Region to pixelate (x, y, width, height)
            
        Returns:
            np.ndarray: Image with pixelation applied
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]
        
        # Reduce size to create pixelation effect
        scale = self.blur_intensity // 10 or 1
        small = cv2.resize(roi, (w//scale, h//scale), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        image[y:y+h, x:x+w] = pixelated
        return image

    def _get_regions_to_blur(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Get regions containing sensitive information.
        
        Args:
            image: PIL Image to process
            
        Returns:
            List[Dict[str, Any]]: List of regions to blur
        """
        try:
            # Get text regions from OCR
            text_regions = self.ocr.get_text_regions(image)
            logger.info(f"OCR found {len(text_regions)} text regions")
            for region in text_regions:
                logger.debug(f"Text region: {region['text'][:50]}...")
            
            # Convert image to text for NER
            full_text = self.ocr.get_text_only(image)
            logger.info(f"Full OCR text: {full_text[:200]}...")
            
            # Get sensitive entities
            sensitive_entities = self.ner.get_sensitive_entities(full_text)
            logger.info(f"Found {len(sensitive_entities)} sensitive entities")
            for entity in sensitive_entities:
                logger.debug(f"Sensitive entity: {entity['type']} - {entity['text']}")
            
            # Map entities to regions
            regions_to_blur = []
            for entity in sensitive_entities:
                entity_text = entity['text'].lower().strip()
                found = False
                for region in text_regions:
                    region_text = region['text'].lower().strip()
                    # Try different matching strategies
                    if (entity_text in region_text or  # Substring match
                        region_text in entity_text or  # Region might be part of entity
                        any(word in region_text for word in entity_text.split())):  # Word-level match
                        regions_to_blur.append({
                            'text': region['text'],
                            'coordinates': region['coordinates'],
                            'entity_type': entity['type']
                        })
                        found = True
                        logger.debug(f"Matched entity '{entity_text}' to region '{region_text}'")
                if not found:
                    logger.warning(f"Could not find region for entity: {entity_text}")
            
            logger.info(f"Found {len(regions_to_blur)} regions to blur")
            return regions_to_blur
            
        except Exception as e:
            logger.error(f"Error getting regions to blur: {e}")
            logger.exception(e)
            return []

    def _normalize_coordinates(self, coords: Dict[str, float],
                             image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        try:
            width, height = image_size
            # Handle both list/tuple and dictionary coordinate formats
            if isinstance(coords, (list, tuple)):
                x1, y1 = int(coords[0][0] * width), int(coords[0][1] * height)
                x2, y2 = int(coords[1][0] * width), int(coords[1][1] * height)
            else:
                x1, y1 = int(coords['x1'] * width), int(coords['y1'] * height)
                x2, y2 = int(coords['x2'] * width), int(coords['y2'] * height)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            # Ensure width and height are positive
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
                
            return (x1, y1, x2-x1, y2-y1)
        except Exception as e:
            logger.error(f"Error normalizing coordinates: {e}")
            return (0, 0, 10, 10)  # Return small default region on error

    def process_image(self, image: Image.Image) -> Image.Image:
        """Process an image and blur sensitive information."""
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Get regions to blur
            regions = self._get_regions_to_blur(image)
            logger.info(f"Processing {len(regions)} regions for blurring")
            
            # Apply blurring to each region
            for region in regions:
                try:
                    coords = self._normalize_coordinates(
                        region['coordinates'],
                        (img_array.shape[1], img_array.shape[0])
                    )
                    logger.debug(f"Blurring region: {coords} ({region.get('entity_type', 'unknown')})")
                    
                    if self.blur_method == "gaussian":
                        img_array = self._apply_gaussian_blur(img_array, coords)
                    else:
                        img_array = self._apply_pixelation(img_array, coords)
                except Exception as e:
                    logger.error(f"Error blurring region: {e}")
                    continue
            
            # Convert back to PIL Image
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            logger.exception(e)
            return image

def process_screenshot(image: Image.Image, enable_ner: bool = True) -> Image.Image:
    """Process a screenshot and blur sensitive information.
    
    Args:
        image: PIL Image to process
        enable_ner: Whether to use NER for sensitive information detection
        
    Returns:
        Image.Image: Processed image with sensitive information blurred
    """
    try:
        processor = SensitiveInfoBlur()
        if not enable_ner:
            # If NER is disabled, only use pattern matching and secrets detection
            processor.ner = None
        return processor.process_image(image)
    except Exception as e:
        logger.error(f"Error processing screenshot: {e}")
        return image 