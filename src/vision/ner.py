"""Named Entity Recognition using GLiNER and detect-secrets."""

import logging
import re
from typing import List, Dict, Any
import tempfile
import os
import math  # Add math import
from pathlib import Path

from detect_secrets import SecretsCollection
from detect_secrets.settings import transient_settings
from detect_secrets.core.scan import scan_file
from gliner import GLiNER
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class NERProcessor:
    """Handles Named Entity Recognition using GLiNER and detect-secrets."""
    
    # Define the default model path relative to the user's home directory
    DEFAULT_MODEL_PATH = os.path.join(os.path.expanduser("~"), ".vigilare", "models", "gliner")
    
    # Common patterns for sensitive information - only keep what detect-secrets might miss
    PATTERNS = {
        # Network information that detect-secrets might miss - enhanced IP detection
        'IP_ADDRESS': r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)',
        
        # Cryptographic material that might be split across lines
        'PRIVATE_KEY': r'(?:-----BEGIN[^-]+?-----.*?-----END[^-]+?-----|'  # PEM format
                      r'MIIE[\w+/=\s-]{20,}|'  # PKCS key format
                      r'MIIC[\w+/=\s-]{20,}|'  # Another PKCS variant
                      r'MIID[\w+/=\s-]{20,}|'  # Another PKCS variant
                      r'MIIBIj[\w+/=\s-]{20,}|'  # Public key format
                      r'MIIBCg[\w+/=\s-]{20,})',  # Another public key variant
                      
        'CERTIFICATE': r'(?:-----BEGIN\s+CERTIFICATE-----.*?-----END\s+CERTIFICATE-----|'
                      r'-----BEGIN\s+CSR-----.*?-----END\s+CSR-----|'
                      r'-----BEGIN\s+PUBLIC\s+KEY-----.*?-----END\s+PUBLIC\s+KEY-----)',
        
        # OpenAI project keys pattern - separate from other API keys for special handling
        'OPENAI_KEY': r'sk-proj-[a-zA-Z0-9_-]{20,}(?:[ \t]*[a-zA-Z0-9_-]{20,})*',
        
        # Additional patterns for API keys and tokens
        'API_KEY': r'(?:'
                  r'(?:[Aa][Pp][Ii][-_]?[Kk][Ee][Yy][-_]?|[Kk][Ee][Yy][-_]?)[a-zA-Z0-9_-]{16,}|'  # Standard API key format
                  r'(?:sk|pk)_(?:live|test)_[0-9a-zA-Z]{24,}|'  # Stripe-like keys
                  r'[a-zA-Z]{2,}_[a-zA-Z0-9_-]{16,}|'  # Prefix_value format
                  r'[A-Z0-9]{20,}(?:[-_][A-Z0-9]{10,}){0,2}'  # AWS-like keys
                  r')',
                  
        'JWT_TOKEN': r'ey[A-Za-z0-9_-]{10,}\.ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
        
        'GENERIC_TOKEN': r'(?:'
                        r'[a-zA-Z0-9_-]{32,}|'  # Basic token format
                        r'[A-Z0-9]{20,}|'  # All caps token
                        r'ghp_[A-Za-z0-9_]{36}|'  # GitHub token
                        r'gho_[A-Za-z0-9_]{36}|'  # GitHub OAuth token
                        r'(?:v1\.|v2\.)?[a-f0-9]{32,}'  # Version prefixed tokens
                        r')',
                        
        'BASE64_DATA': r'(?:[A-Za-z0-9+/]{32,}={0,3}|[A-Za-z0-9_-]{32,})',
        
        'HASH': r'\b[a-fA-F0-9]{32,}\b'
    }
    
    # GLiNER labels for PII detection - focus on personal information
    GLINER_LABELS = [
        "ssn",
        "first_name",
        "email",
        "last_name",
        "customer_id",
        "employee_id",
        "name",
        "street_address",
        "phone_number",
        "ipv4",
        "credit_card_number",
        "license_plate",
        "address",
        "user_name",
        "device_identifier",
        "bank_routing_number",
        "company_name",
        "unique_identifier",
        "biometric_identifier",
        "account_number",
        "city",
        "certificate_license_number",
        "postcode",
        "vehicle_identifier",
        "coordinate",
        "api_key",
        "ipv6",
        "password",
        "health_plan_beneficiary_number",
        "national_id",
        "tax_id",
        "state",
        "swift_bic",
        "cvv",
        "pin"
    ]
    
    # Mapping of GLiNER labels to internal types
    LABEL_MAPPING = {
        'name': 'PERSON',
        'first_name': 'PERSON',
        'last_name': 'PERSON',
        'street_address': 'ADDRESS',
        'address': 'ADDRESS',
        'city': 'LOCATION',
        'state': 'LOCATION',
        'country': 'LOCATION',
        'postcode': 'LOCATION',
        'email': 'EMAIL',
        'phone_number': 'PHONE',
        'credit_card_number': 'CREDIT_CARD',
        'ssn': 'SSN',
        'company_name': 'ORGANIZATION',
        'user_name': 'USERNAME',
        'api_key': 'API_KEY',
        'password': 'PASSWORD',
        'national_id': 'ID',
        'tax_id': 'ID',
        'customer_id': 'ID',
        'employee_id': 'ID',
        'unique_identifier': 'ID',
        'device_identifier': 'ID',
        'vehicle_identifier': 'ID',
        'certificate_license_number': 'ID',
        'license_plate': 'ID',
        'bank_routing_number': 'BANK_INFO',
        'account_number': 'BANK_INFO',
        'swift_bic': 'BANK_INFO',
        'cvv': 'BANK_INFO',
        'pin': 'BANK_INFO',
        'ipv4': 'IP_ADDRESS',
        'ipv6': 'IP_ADDRESS',
        'coordinate': 'LOCATION',
        'biometric_identifier': 'BIOMETRIC'
    }
    
    # detect-secrets plugins configuration - increase sensitivity
    SECRETS_PLUGINS = [
        {'name': 'ArtifactoryDetector'},
        {'name': 'AWSKeyDetector', 'limit': 3.0},  # More sensitive
        {'name': 'AzureStorageKeyDetector'},
        {'name': 'BasicAuthDetector'},
        {'name': 'CloudantDetector'},
        {'name': 'DiscordBotTokenDetector'},
        {'name': 'GitHubTokenDetector'},
        {'name': 'GitLabTokenDetector'},
        {'name': 'Base64HighEntropyString', 'limit': 4.0},  # More sensitive
        {'name': 'HexHighEntropyString', 'limit': 2.5},    # More sensitive
        {'name': 'IbmCloudIamDetector'},
        {'name': 'IbmCosHmacDetector'},
        {'name': 'IPPublicDetector'},
        {'name': 'JwtTokenDetector'},
        {'name': 'KeywordDetector', 
         'keyword_exclude': None,  # Don't exclude any keywords
         'keyword_groups': {
             'api': ['api[_-]?key', 'api[_-]?secret', 'access[_-]?key', 'access[_-]?secret'],
             'token': ['token', 'secret', 'private[_-]?key'],
             'password': ['password', 'passwd', 'pwd'],
             'key': ['key', 'encryption[_-]?key', 'secret[_-]?key']
         }},
        {'name': 'MailchimpDetector'},
        {'name': 'NpmDetector'},
        {'name': 'OpenAIDetector'},
        {'name': 'PrivateKeyDetector'},
        {'name': 'PypiTokenDetector'},
        {'name': 'SendGridDetector'},
        {'name': 'SlackDetector'},
        {'name': 'SoftlayerDetector'},
        {'name': 'SquareOAuthDetector'},
        {'name': 'StripeDetector'},
        {'name': 'TelegramBotTokenDetector'},
        {'name': 'TwilioKeyDetector'}
    ]
    
    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1", model_path: str = None):
        """Initialize NER processor with GLiNER and detect-secrets.
        
        Args:
            model_name: Name of the model on HuggingFace Hub
            model_path: Optional local path to store/load the model. If None, uses DEFAULT_MODEL_PATH
        """
        try:
            # Set up model path
            self.model_path = model_path or self.DEFAULT_MODEL_PATH
            
            # Ensure model directory exists
            os.makedirs(self.model_path, exist_ok=True)
            
            # Check if model exists locally
            if not os.path.exists(os.path.join(self.model_path, "config.json")):
                logger.info(f"Model not found locally at {self.model_path}. Downloading from HuggingFace...")
                try:
                    # Download model from HuggingFace
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=self.model_path,
                        local_dir_use_symlinks=False  # Ensure files are copied, not symlinked
                    )
                    logger.info(f"Successfully downloaded model to {self.model_path}")
                except Exception as e:
                    logger.error(f"Error downloading model: {e}")
                    raise
            else:
                logger.info(f"Loading model from local path: {self.model_path}")
            
            # Load the model from local path
            self.model = GLiNER.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device="cpu",
                local_files_only=True  # Ensure no network calls are made
            )
            
            # Initialize detect-secrets with all plugins
            with transient_settings({
                'plugins_used': self.SECRETS_PLUGINS
            }) as settings:
                self.secrets_scanner = SecretsCollection()
            
            # Character weights for entropy calculation
            self.char_weights = {
                # Special characters that might indicate secrets
                '_': 3.0,  # Underscore gets high weight for API keys
                '-': 3.0,  # Hyphen gets high weight for API keys
                
                # Common coding characters get very low weights
                '(': 0.2,  # Parentheses
                ')': 0.2,
                '[': 0.2,  # Brackets
                ']': 0.2,
                '{': 0.2,  # Braces
                '}': 0.2,
                '<': 0.2,  # Angle brackets
                '>': 0.2,
                '/': 0.2,  # Path separators
                '\\': 0.2,
                '.': 0.2,  # Dot notation
                ':': 0.2,  # Namespace separator
                ';': 0.2,  # Statement terminator
                '=': 0.2,  # Assignment
                '+': 0.2,  # Operators
                '*': 0.2,
                '&': 0.2,
                '|': 0.2,
                '!': 0.2,
                '@': 0.2,  # Decorators
                '#': 0.2,  # Comments
                '$': 0.2,  # Shell variables
                
                # Normal characters get standard weight
                **{str(d): 1.0 for d in range(10)},  # Numbers
                **{chr(i): 1.0 for i in range(65, 91)},  # Uppercase letters
                **{chr(i): 1.0 for i in range(97, 123)}  # Lowercase letters
            }
                
            logger.info("Successfully initialized GLiNER model and detect-secrets")
        except Exception as e:
            logger.error(f"Error initializing NER processor: {e}")
            logger.exception(e)
            raise

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text to improve entity detection."""
        # Add newlines before common labels to help with detection
        labels = [
            "Name:", "Email:", "Phone:", "Address:", "SSN:", "Credit Card:",
            "Account:", "Password:", "Username:", "ID:", "License:", "DOB:",
            "Contact:", "Location:", "Organization:"
        ]
        
        processed = text
        for label in labels:
            processed = processed.replace(f" {label}", f"\n{label}")
        
        # Add spaces around special characters to help tokenization
        special_chars = [":", "/", "-", "(", ")", "@", "."]
        for char in special_chars:
            processed = processed.replace(char, f" {char} ")
        
        return processed

    def _find_pattern_matches(self, text: str) -> List[Dict[str, Any]]:
        """Find matches for sensitive information patterns."""
        matches = []
        
        # Skip if text is too short
        if len(text) < 8:
            return []
            
        # First find all URLs in the text
        url_pattern = r'(?:https?://)?(?:(?:[\w-]+\.)+[\w-]+|(?:\d{1,3}\.){3}\d{1,3})(?:/[\w\-./?%&=]*/?)?'
        urls = []
        sensitive_urls = []  # Track URLs containing sensitive info
        
        # Helper function to validate IP address
        def is_valid_ip(ip_str: str) -> bool:
            try:
                parts = ip_str.split('.')
                if len(parts) != 4:
                    return False
                return all(0 <= int(part) <= 255 for part in parts)
            except (ValueError, AttributeError):
                return False
        
        # Helper function to check if URL contains sensitive info
        def has_sensitive_info(url: str) -> bool:
            # Check for API keys and tokens - be more selective
            token_patterns = [
                r'(?:api[_-]?key|access[_-]?token)[=/]([a-zA-Z0-9_-]{16,})',  # API keys in URL params
                r'(?:bearer|auth)[=/]([a-zA-Z0-9_-]{16,})',  # Bearer/auth tokens
                r'(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24,}',  # Stripe-like keys
                r'gh[po]_[A-Za-z0-9_]{36}'  # GitHub tokens
            ]
            
            # Only consider URLs sensitive if they contain actual secrets
            for pattern in token_patterns:
                match = re.search(pattern, url, re.IGNORECASE)
                if match:
                    # If we matched a group, check that group for entropy
                    token = match.group(1) if match.groups() else match.group(0)
                    # Calculate entropy to verify it's not a false positive
                    entropy = self._calculate_weighted_entropy(token)
                    if entropy > 0.6:  # Higher threshold for URLs
                        return True
            
            # Check for IP addresses - only if they're not common patterns
            ip_pattern = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)/?'
            if re.search(ip_pattern, url):
                ip_match = re.search(ip_pattern, url)
                if ip_match:
                    ip_str = ip_match.group().rstrip('/')
                    # Skip common development IPs
                    if ip_str in {'127.0.0.1', '192.168.0.1', '192.168.1.1', 'localhost', '0.0.0.0'}:
                        return False
                    if is_valid_ip(ip_str):
                        # Check if it's a private IP
                        parts = [int(p) for p in ip_str.split('.')]
                        if (parts[0] == 10 or  # 10.x.x.x
                            (parts[0] == 172 and 16 <= parts[1] <= 31) or  # 172.16.x.x to 172.31.x.x
                            (parts[0] == 192 and parts[1] == 168)):  # 192.168.x.x
                            return False
                        return True
            
            return False
        
        # Find and categorize URLs
        for m in re.finditer(url_pattern, text, re.IGNORECASE):
            url = m.group()
            # Check if URL contains an IP address
            ip_pattern = r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)/?'
            ip_match = re.search(ip_pattern, url)
            
            # Add to URLs list if it's a URL (has protocol or domain-like structure)
            if url.startswith(('http:', 'https:')) or '.' in url:
                urls.append((m.start(), m.end()))
                
                # Check if it contains an IP or other sensitive info
                if (ip_match and is_valid_ip(ip_match.group().rstrip('/'))) or has_sensitive_info(url):
                    sensitive_urls.append((m.start(), m.end()))
                    matches.append({
                        'text': url,
                        'type': 'SENSITIVE_URL',
                        'start_char': m.start(),
                        'end_char': m.end(),
                        'confidence': 0.95
                    })
        
        # Helper function to check if a position is within any URL
        def is_within_url(start, end):
            return any(url_start <= start and end <= url_end for url_start, url_end in urls)
        
        # Helper function to check if a position is within a sensitive URL
        def is_within_sensitive_url(start, end):
            return any(url_start <= start and end <= url_end for url_start, url_end in sensitive_urls)
        
        # Process each pattern
        for pattern_type, pattern in self.PATTERNS.items():
            try:
                # Special handling for IP addresses
                if pattern_type == 'IP_ADDRESS':
                    # Find all potential IP addresses
                    for match in re.finditer(pattern, text, re.MULTILINE):
                        matched_text = match.group().strip()
                        
                        # Skip if not a valid IP
                        if not is_valid_ip(matched_text):
                            continue
                            
                        # Only skip if in non-sensitive URL
                        if is_within_url(match.start(), match.end()) and not is_within_sensitive_url(match.start(), match.end()):
                            continue
                        
                        matches.append({
                            'text': matched_text,
                            'type': pattern_type,
                            'start_char': match.start(),
                            'end_char': match.end(),
                            'confidence': 0.95
                        })
                    continue
                
                # Special handling for OpenAI project keys
                if pattern_type == 'OPENAI_KEY':
                    # Find all potential key parts
                    key_parts = []
                    current_key = []
                    last_end = 0
                    
                    # First find all key-like segments
                    key_matches = list(re.finditer(r'sk-proj-[a-zA-Z0-9_-]{20,}|[a-zA-Z0-9_-]{20,}', text, re.MULTILINE))
                    
                    for i, match in enumerate(key_matches):
                        matched_text = match.group().strip()
                        
                        # If this is a key start
                        if matched_text.startswith('sk-proj-'):
                            # If we have a current key, save it
                            if current_key:
                                key_parts.append((current_key[0][0], current_key[-1][1], 
                                               ''.join(part[2] for part in current_key)))
                            current_key = [(match.start(), match.end(), matched_text)]
                            last_end = match.end()
                        # If we have a current key and this part looks like it belongs
                        elif current_key:
                            # Check if this part is close enough (within 2 lines)
                            text_between = text[last_end:match.start()]
                            newlines = text_between.count('\n')
                            
                            if newlines <= 2 and len(text_between.strip()) <= 5:
                                current_key.append((match.start(), match.end(), matched_text))
                                last_end = match.end()
                            else:
                                # Too far, save current key and start fresh
                                key_parts.append((current_key[0][0], current_key[-1][1], 
                                               ''.join(part[2] for part in current_key)))
                                current_key = []
                    
                    # Add final key if exists
                    if current_key:
                        key_parts.append((current_key[0][0], current_key[-1][1], 
                                       ''.join(part[2] for part in current_key)))
                    
                    # Add all found OpenAI keys
                    for start, end, key_text in key_parts:
                        # Clean up any whitespace in the key
                        key_text = re.sub(r'\s+', '', key_text)
                        matches.append({
                            'text': key_text,
                            'type': 'API_KEY',
                            'start_char': start,
                            'end_char': end,
                            'confidence': 0.99  # Highest confidence for OpenAI keys
                        })
                    continue
                
                # Handle multi-line patterns
                if pattern_type in ['PRIVATE_KEY', 'CERTIFICATE']:
                    for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE | re.DOTALL):
                        matched_text = match.group().strip()
                        if matched_text:
                            matches.append({
                                'text': matched_text,
                                'type': pattern_type,
                                'start_char': match.start(),
                                'end_char': match.end(),
                                'confidence': 0.98
                            })
                    continue
                
                # Regular single-line patterns
                match_count = 0
                for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
                    matched_text = match.group().strip()
                    if not matched_text:
                        continue
                        
                    # For IP addresses, only skip if in http/https URL
                    if pattern_type == 'IP_ADDRESS':
                        # Check if this IP is part of a URL with http/https
                        if is_within_url(match.start(), match.end()):
                            continue
                    
                    # For API keys, clean up any whitespace
                    if pattern_type == 'API_KEY':
                        matched_text = re.sub(r'\s+', '', matched_text)
                    
                    matches.append({
                        'text': matched_text,
                        'type': pattern_type,
                        'start_char': match.start(),
                        'end_char': match.end(),
                        'confidence': 0.95
                    })
                    
                    match_count += 1
                    if match_count >= 100:  # Limit matches per pattern
                        break
                        
            except Exception as e:
                logger.error(f"Error matching pattern {pattern_type}: {e}")
                continue
        
        # Filter out matches that are likely false positives
        filtered_matches = []
        for match in matches:
            # Always keep OpenAI keys and crypto material
            if match['type'] in ['PRIVATE_KEY', 'CERTIFICATE'] or (match['type'] == 'API_KEY' and 'sk-proj-' in match['text'].lower()):
                filtered_matches.append(match)
                continue
            
            # Skip if too short (except for specific types)
            if (len(match['text']) < 16 and 
                match['type'] in ['API_KEY', 'GENERIC_TOKEN', 'BASE64_DATA'] and
                not any(prefix in match['text'].lower() for prefix in ['sk-', 'pk-', 'sk_', 'pk_', 'ghp_', 'gho_'])):
                continue
            
            # Skip if looks like a common word
            if match['type'] in ['GENERIC_TOKEN', 'BASE64_DATA', 'HASH']:
                has_mixed_case = any(c.isupper() for c in match['text']) and any(c.islower() for c in match['text'])
                has_special = any(c in '_-+/=' for c in match['text'])
                has_numbers = any(c.isdigit() for c in match['text'])
                if not ((has_mixed_case or has_special) and has_numbers):
                    continue
            
            filtered_matches.append(match)
        
        return filtered_matches

    def _scan_secrets(self, text: str) -> List[Dict[str, Any]]:
        """Scan text for secrets using detect-secrets."""
        secrets = []
        temp_file = None
        
        try:
            # Skip if text is too short
            if len(text) < 8:  # Minimum length for most secrets
                return []

            # Create a temporary file for detect-secrets
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
            temp_file.write(text)
            temp_file.close()  # Close file before scanning
            
            # Import plugins directly
            from detect_secrets.plugins.aws import AWSKeyDetector
            from detect_secrets.plugins.artifactory import ArtifactoryDetector
            from detect_secrets.plugins.azure_storage import AzureStorageKeyDetector
            from detect_secrets.plugins.basic_auth import BasicAuthDetector
            from detect_secrets.plugins.cloudant import CloudantDetector
            from detect_secrets.plugins.discord import DiscordBotTokenDetector
            from detect_secrets.plugins.github import GitHubTokenDetector
            from detect_secrets.plugins.gitlab import GitLabTokenDetector
            from detect_secrets.plugins.high_entropy_strings import Base64HighEntropyString, HexHighEntropyString
            from detect_secrets.plugins.ibm_cloud_iam import IbmCloudIamDetector
            from detect_secrets.plugins.ibm_cos_hmac import IbmCosHmacDetector
            from detect_secrets.plugins.jwt import JwtTokenDetector
            from detect_secrets.plugins.keyword import KeywordDetector
            from detect_secrets.plugins.mailchimp import MailchimpDetector
            from detect_secrets.plugins.npm import NpmDetector
            from detect_secrets.plugins.openai import OpenAIDetector
            from detect_secrets.plugins.private_key import PrivateKeyDetector
            from detect_secrets.plugins.pypi import PypiTokenDetector
            from detect_secrets.plugins.sendgrid import SendGridDetector
            from detect_secrets.plugins.slack import SlackDetector
            from detect_secrets.plugins.softlayer import SoftlayerDetector
            from detect_secrets.plugins.square_oauth import SquareOAuthDetector
            from detect_secrets.plugins.stripe import StripeDetector
            from detect_secrets.plugins.telegram import TelegramBotTokenDetector
            from detect_secrets.plugins.twilio import TwilioKeyDetector
            
            # Initialize plugins with their respective settings
            plugins = [
                AWSKeyDetector(),
                ArtifactoryDetector(),
                AzureStorageKeyDetector(),
                BasicAuthDetector(),
                CloudantDetector(),
                DiscordBotTokenDetector(),
                GitHubTokenDetector(),
                GitLabTokenDetector(),
                Base64HighEntropyString(limit=4.5),
                HexHighEntropyString(limit=3.0),
                IbmCloudIamDetector(),
                IbmCosHmacDetector(),
                JwtTokenDetector(),
                KeywordDetector(keyword_exclude=None),
                MailchimpDetector(),
                NpmDetector(),
                OpenAIDetector(),
                PrivateKeyDetector(),
                PypiTokenDetector(),
                SendGridDetector(),
                SlackDetector(),
                SoftlayerDetector(),
                SquareOAuthDetector(),
                StripeDetector(),
                TelegramBotTokenDetector(),
                TwilioKeyDetector()
            ]
            
            # Use scan_file directly with plugins
            with transient_settings({
                'plugins_used': [{'name': plugin.__class__.__name__} for plugin in plugins]
            }):
                all_secrets = scan_file(temp_file.name)
                
                if all_secrets:
                    # Process results
                    lines = text.split('\n')
                    line_starts = [0]  # Cache line start positions
                    for line in lines[:-1]:
                        line_starts.append(line_starts[-1] + len(line) + 1)
                    
                    for secret in all_secrets:
                        try:
                            line_number = secret.line_number - 1  # Convert to 0-based index
                            if line_number >= len(lines):
                                continue
                                
                            line = lines[line_number]
                            line_start = line_starts[line_number]
                            
                            # Get the secret text
                            secret_text = getattr(secret, 'secret_value', line.strip())
                            if not secret_text:
                                continue
                                
                            # Find position in line
                            start_pos = line_start
                            end_pos = line_start + len(line)
                            
                            # Get the type
                            secret_type = getattr(secret, 'type_name', None) or getattr(secret, 'type', 'UNKNOWN')
                            
                            secrets.append({
                                'text': secret_text,
                                'type': secret_type.upper(),
                                'start_char': start_pos,
                                'end_char': end_pos,
                                'confidence': 0.95
                            })
                        except Exception as e:
                            logger.error(f"Error processing secret: {e}")
                            continue
                    
        except Exception as e:
            logger.error(f"Error scanning for secrets: {e}")
            
        finally:
            # Clean up temporary file
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except Exception:
                    pass
            
        return secrets

    def _process_gliner(self, text: str) -> List[Dict[str, Any]]:
        """Process text with GLiNER for PII detection."""
        entities = []
        seen_positions = set()  # Track positions we've already processed
        
        try:
            # Process text in chunks to handle long texts
            max_length = 512
            overlap = 100
            
            # Split text into chunks at natural boundaries
            chunks = []
            positions = []  # Track start position of each chunk
            
            start_pos = 0
            text_length = len(text)
            
            while start_pos < text_length:
                end_pos = min(start_pos + max_length, text_length)
                
                # Try to end at a natural boundary
                if end_pos < text_length:
                    # Look for newline or period in the overlap region
                    overlap_text = text[end_pos - overlap:end_pos]
                    last_newline = overlap_text.rfind('\n')
                    last_period = overlap_text.rfind('.')
                    
                    # Use the latest natural boundary found
                    if last_newline != -1:
                        end_pos = end_pos - overlap + last_newline
                    elif last_period != -1:
                        end_pos = end_pos - overlap + last_period + 1  # Include the period
                
                chunk = text[start_pos:end_pos].strip()
                if chunk:  # Only add non-empty chunks
                    chunks.append(chunk)
                    positions.append(start_pos)
                
                start_pos = end_pos
            
            # Process each chunk
            for chunk, chunk_start in zip(chunks, positions):
                # Process with GLiNER using confidence threshold
                predictions = self.model.predict_entities(chunk, labels=self.GLINER_LABELS, threshold=0.3)
                
                for prediction in predictions:
                    try:
                        entity_text = prediction.get('text', '').strip()
                        entity_type = prediction.get('label', '')
                        score = prediction.get('score', 0.3)
                        
                        if not entity_text or not entity_type:
                            continue
                        
                        # Map GLiNER type to internal type
                        mapped_type = self.LABEL_MAPPING.get(entity_type.lower(), entity_type.upper())
                        
                        # Get positions within chunk
                        chunk_pos = chunk.find(entity_text)
                        if chunk_pos == -1:
                            continue
                            
                        # Calculate global position
                        global_start = chunk_start + chunk_pos
                        global_end = global_start + len(entity_text)
                        
                        # Check if we've already processed this position
                        position_key = (global_start, global_end, mapped_type)
                        if position_key in seen_positions:
                            continue
                        
                        # Check for near-duplicate positions
                        is_duplicate = False
                        for existing in entities:
                            # Check for significant overlap and same type
                            if (existing['type'] == mapped_type and
                                abs(existing['start_char'] - global_start) < 5):
                                is_duplicate = True
                                # Keep the one with higher confidence
                                if score > existing['confidence']:
                                    entities.remove(existing)
                                    is_duplicate = False
                                break
                        
                        if not is_duplicate:
                            entities.append({
                                'text': entity_text,
                                'type': mapped_type,
                                'start_char': global_start,
                                'end_char': global_end,
                                'confidence': score
                            })
                            seen_positions.add(position_key)
                            logger.debug(f"Found {mapped_type}: {entity_text} (confidence: {score})")
                    
                    except Exception as e:
                        logger.error(f"Error processing GLiNER prediction: {e}")
                        continue
            
        except Exception as e:
            logger.error(f"Error in GLiNER processing: {e}")
            logger.exception(e)
            
        return entities

    def _calculate_weighted_entropy(self, text: str) -> float:
        """Calculate weighted entropy of a string based on character types and distribution."""
        if not text or len(text) < 2:
            return 0.0
        
        # Count characters with their weights
        weighted_counts = {}
        total_weight = 0.0
        valid_chars = 0
        
        # Track character type distributions
        char_types = {
            'upper': 0,
            'lower': 0,
            'digit': 0,
            'special': 0
        }
        
        # Track consecutive characters without spaces
        continuous_length = len(text.strip())
        # Bonus for longer continuous strings (no spaces)
        length_bonus = min(0.3, continuous_length / 100)  # Max 30% bonus for length
        
        for char in text:
            # Update character type counts
            if char.isupper():
                char_types['upper'] += 1
            elif char.islower():
                char_types['lower'] += 1
            elif char.isdigit():
                char_types['digit'] += 1
            else:
                char_types['special'] += 1
            
            if char in self.char_weights:
                weight = self.char_weights[char]
                weighted_counts[char] = weighted_counts.get(char, 0) + weight
                total_weight += weight
                valid_chars += 1
        
        # If no valid characters found or all characters are the same
        if valid_chars < 2:
            return 0.0
        
        # Calculate weighted entropy
        entropy = 0.0
        for count in weighted_counts.values():
            prob = count / total_weight
            entropy -= prob * math.log2(prob)
        
        # Normalize by maximum possible entropy for the length
        max_entropy = valid_chars * (1/valid_chars) * (-math.log2(1/valid_chars))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Calculate character type distribution scores
        total_chars = sum(char_types.values())
        type_ratios = {k: v/total_chars for k, v in char_types.items()}
        
        # Calculate variance between letters and numbers with increased weight
        letter_ratio = (type_ratios['upper'] + type_ratios['lower'])
        number_ratio = type_ratios['digit']
        special_ratio = type_ratios['special']
        
        # Calculate variance score (how well-distributed the characters are)
        # Ideal distribution for API keys/hashes: mix of types without heavy bias
        variance_score = 1.0 - (
            abs(letter_ratio - 0.5) +  # Letters should be ~50%
            abs(number_ratio - 0.3) +  # Numbers should be ~30%
            abs(special_ratio - 0.2)   # Special chars should be ~20%
        ) / 3
        
        # Penalize if any type is too dominant (>70%)
        if any(ratio > 0.7 for ratio in type_ratios.values()):
            variance_score *= 0.5
        
        # Bonus for having a good mix of upper and lower case
        if type_ratios['upper'] > 0.1 and type_ratios['lower'] > 0.1:
            variance_score *= 1.2
        
        # Calculate character distribution score with higher weight on variance
        distribution_score = (
            variance_score * 0.6 +                    # 60% weight on character variance
            (1.0 - abs(0.5 - letter_ratio)) * 0.2 +  # 20% weight on letter ratio
            (1.0 - abs(0.3 - number_ratio)) * 0.1 +  # 10% weight on number ratio
            min(1.0, special_ratio * 5) * 0.1        # 10% weight on special chars
        )
        
        # Apply length bonus to both entropy and distribution scores
        normalized_entropy = normalized_entropy * (1 + length_bonus)
        distribution_score = distribution_score * (1 + length_bonus)
        
        # Combine entropy and distribution scores with adjusted weights
        # Give more weight to distribution/variance (40%) and entropy (60%)
        final_score = (0.6 * normalized_entropy) + (0.4 * distribution_score)
        
        return min(1.0, final_score)  # Ensure score doesn't exceed 1.0

    def _find_high_entropy_strings(self, text: str) -> List[Dict[str, Any]]:
        """Find strings with high weighted entropy that might be sensitive."""
        matches = []
        
        # Split text into potential tokens - be more selective
        token_pattern = r'[a-zA-Z0-9][a-zA-Z0-9_-]{11,}[a-zA-Z0-9]'  # At least 13 chars
        
        # Common development terms to ignore
        common_terms = {
            'configuration', 'development', 'production', 'application',
            'environment', 'authentication', 'authorization', 'management',
            'integration', 'deployment', 'connection', 'documentation',
            'javascript', 'typescript', 'properties', 'dependency'
        }
        
        for match in re.finditer(token_pattern, text):
            token = match.group()
            
            # Skip if token is a common word or pattern
            if token.lower() in common_terms:
                continue
                
            # Skip if token looks like a URL or file path
            if ('http' in token.lower() or 
                'www.' in token.lower() or 
                '.com' in token.lower() or 
                ':\\' in token or 
                './' in token or 
                '../' in token):
                continue
            
            # Calculate entropy
            entropy = self._calculate_weighted_entropy(token)
            
            # Tokens with high entropy and special characters are more likely to be sensitive
            special_char_ratio = len([c for c in token if c in '_-']) / len(token)
            
            # More strict confidence calculation
            confidence = min(0.95, entropy * (1 + special_char_ratio))
            
            # Higher entropy threshold, especially for longer strings
            length_factor = min(1.0, len(token) / 32)
            entropy_threshold = 0.7 - (0.1 * length_factor)  # More strict threshold
            
            # Additional checks for likely sensitive content
            has_mixed_case = any(c.isupper() for c in token) and any(c.islower() for c in token)
            has_numbers = any(c.isdigit() for c in token)
            
            if (entropy > entropy_threshold and 
                confidence > 0.6 and  # Higher confidence threshold
                has_mixed_case and 
                has_numbers and 
                special_char_ratio > 0.1):  # Require some special characters
                matches.append({
                    'text': token,
                    'type': 'HIGH_ENTROPY_STRING',
                    'start_char': match.start(),
                    'end_char': match.end(),
                    'confidence': confidence
                })
        
        return matches

    def get_sensitive_entities(self, text: str) -> List[Dict[str, Any]]:
        """Get potentially sensitive entities from text using GLiNER, detect-secrets, and patterns."""
        try:
            if not text or not isinstance(text, str):
                logger.warning("Invalid input text")
                return []
            
            # Get PII from GLiNER first (primary source for PII)
            pii_entities = self._process_gliner(text)
            logger.info(f"GLiNER found {len(pii_entities)} PII entities")
            
            # Get secrets from detect-secrets (primary source for secrets)
            secrets = self._scan_secrets(text)
            logger.info(f"detect-secrets found {len(secrets)} secrets")
            
            # Get pattern matches (for specific cases)
            pattern_matches = self._find_pattern_matches(text)
            logger.info(f"Pattern matching found {len(pattern_matches)} matches")
            
            # Get high entropy strings
            entropy_matches = self._find_high_entropy_strings(text)
            logger.info(f"Entropy analysis found {len(entropy_matches)} potential sensitive strings")
            
            # Combine all entities with minimal overlap checking
            all_entities = {}  # Use dict for deduplication
            
            # Helper function to normalize text
            def normalize_text(text):
                return re.sub(r'[\s-]', '', text.lower())
            
            # Add entities in priority order (pattern matches first, then entropy matches)
            for entity in pii_entities + secrets + pattern_matches + entropy_matches:
                norm_text = normalize_text(entity['text'])
                key = (norm_text, entity['type'])
                
                # Keep highest confidence version
                if key not in all_entities or entity['confidence'] > all_entities[key]['confidence']:
                    all_entities[key] = entity
            
            result = list(all_entities.values())
            logger.info(f"Found {len(result)} unique sensitive entities")
            return result
            
        except Exception as e:
            logger.error(f"Error getting sensitive entities: {e}")
            logger.exception(e)
            return []

    def get_entity_positions(self, text: str) -> List[Dict[str, Any]]:
        """Get positions of entities in text."""
        return self.get_sensitive_entities(text)
