#@title `LanguageModelAPI`
import asyncio
import json
import os
import re
import time
import logging
from copy import deepcopy
from pathlib import Path
from tqdm.auto import tqdm
from abc import ABC
from abc import abstractmethod
from tqdm.asyncio import tqdm as async_tqdm
from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict, Counter

from google import genai
from google.genai.types import GenerateContentConfig
from json_repair import repair_json
from openai import AsyncOpenAI

from utils import validate_genai_response_constraint

async def _run_and_return_index(index, coro):
    try:
        result = await coro
        return index, result
    except Exception as e:
        return index, e

def _add_additional_properties_false(schema: Any) -> Any:
    if isinstance(schema, dict):
        normalized = {k: _add_additional_properties_false(v) for k, v in schema.items()}
        if normalized.get("type") == "object":
            normalized["additionalProperties"] = False
        return normalized
    if isinstance(schema, list):
        return [_add_additional_properties_false(item) for item in schema]
    return schema

def normalize_openai_schema(schema: Dict[str, Any], name: str = "response") -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": _add_additional_properties_false(deepcopy(schema)),
    }

def _extract_openai_response_text(response: Any) -> str:
    text = getattr(response, 'output_text', None)
    if text and text.strip():
        return text.strip()

    output_items = getattr(response, 'output', None) or []
    text_parts = []
    for item in output_items:
        for content in getattr(item, 'content', None) or []:
            if getattr(content, 'type', None) == 'output_text':
                content_text = getattr(content, 'text', None)
                if content_text:
                    text_parts.append(content_text)

    return ''.join(text_parts).strip()


def _resolve_log_file_path(logger: logging.Logger) -> Optional[Path]:
    for handler in logger.handlers:
        stream = getattr(handler, "stream", None)
        stream_name = getattr(stream, "name", None)
        if isinstance(stream_name, str) and stream_name not in {"<stderr>", "<stdout>"}:
            return Path(stream_name)
    return None


def _build_api_call_logger(
    base_logger: logging.Logger,
    logger_name_suffix: str,
    file_suffix: str,
) -> tuple[logging.Logger, Optional[Path]]:
    logger_name = f"{base_logger.name}.{logger_name_suffix}"
    api_logger = logging.getLogger(logger_name)
    api_logger.setLevel(logging.INFO)
    api_logger.propagate = False

    while api_logger.handlers:
        handler = api_logger.handlers.pop()
        api_logger.removeHandler(handler)
        handler.close()

    base_log_path = _resolve_log_file_path(base_logger)
    if base_log_path is None:
        return api_logger, None

    api_log_path = base_log_path.with_name(f"{base_log_path.stem}.{file_suffix}{base_log_path.suffix}")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(api_log_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    api_logger.addHandler(file_handler)
    return api_logger, api_log_path


def _strip_markdown_json_fences(text: str) -> str:
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()
    return text.strip()


def _extract_balanced_json_fragment(text: str) -> Optional[str]:
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start:index + 1].strip()
    return None


def _candidate_json_texts(text: str) -> List[str]:
    candidates: List[str] = []
    for candidate in (
        text.strip(),
        _strip_markdown_json_fences(text),
        _extract_balanced_json_fragment(_strip_markdown_json_fences(text) or text),
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _schema_instruction(schema: Dict[str, Any]) -> str:
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []

    field_descriptions = []
    for field_name, field_schema in properties.items():
        field_type = field_schema.get("type", "value") if isinstance(field_schema, dict) else "value"
        requirement = "required" if field_name in required else "optional"
        field_descriptions.append(f'"{field_name}" ({field_type}, {requirement})')

    fields_text = ", ".join(field_descriptions) if field_descriptions else "the requested schema"
    return (
        "Return only one valid JSON object that matches the requested schema. "
        "Do not include markdown fences, explanations, prose before the JSON, or prose after the JSON. "
        f"The JSON must contain: {fields_text}."
    )


def _inject_schema_instruction(
    messages: List[Dict[str, str]],
    response_schema: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    if not response_schema:
        return messages

    instruction = _schema_instruction(response_schema)
    updated_messages = [dict(message) for message in messages]

    if updated_messages and updated_messages[0].get("role") == "system":
        existing_content = updated_messages[0].get("content", "")
        updated_messages[0]["content"] = f"{existing_content.rstrip()}\n\n{instruction}".strip()
        return updated_messages

    return [{"role": "system", "content": instruction}, *updated_messages]


def _schema_requires_traversal_fields(constraint: Dict[str, Any]) -> bool:
    required = set(constraint.get("required", [])) if isinstance(constraint, dict) else set()
    return {"reasoning", "ranking", "relevance_scores"}.issubset(required)


def _valid_candidate_ids_from_prompt(prompt: Any) -> List[int]:
    if isinstance(prompt, list):
        prompt_text = "\n".join(str(message.get("content", message)) for message in prompt)
    else:
        prompt_text = str(prompt)

    match = re.search(r"Valid candidate IDs for this request:\s*([^\n.]+)", prompt_text)
    if not match:
        return []

    return [int(x) for x in re.findall(r"\d+", match.group(1))]


def _fallback_traversal_json(response_text: str, valid_candidate_ids: List[int]) -> Optional[str]:
    if len(valid_candidate_ids) != 1:
        return None

    candidate_id = valid_candidate_ids[0]
    score = 50
    try:
        repaired_object = repair_json(response_text, return_objects=True)
        for value in repaired_object.values() if isinstance(repaired_object, dict) else []:
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                score = round(float(value) * 100)
                break
            if isinstance(value, (int, float)) and 1 < value <= 100:
                score = round(float(value))
                break
    except Exception:
        repaired_object = response_text

    fallback = {
        "reasoning": (
            "Local model did not return the required traversal JSON. "
            f"Using the only valid candidate ID {candidate_id} as a fallback. "
            f"Raw response: {str(repaired_object)[:500]}"
        ),
        "ranking": [candidate_id],
        "relevance_scores": [[candidate_id, int(max(0, min(100, score)))]],
    }
    return json.dumps(fallback, ensure_ascii=False)


def _coerce_text_to_schema_json(
    response_text: str,
    constraint: Dict[str, Any],
    prompt: Any = None,
) -> str:
    last_error = "response did not match the requested schema"

    for candidate in _candidate_json_texts(response_text):
        is_valid, error = validate_genai_response_constraint(candidate, constraint)
        if is_valid:
            return candidate.strip()
        last_error = error

        try:
            repaired_object = repair_json(candidate, return_objects=True)
            repaired_text = json.dumps(repaired_object, ensure_ascii=False)
            is_valid, error = validate_genai_response_constraint(repaired_text, constraint)
            if is_valid:
                return repaired_text
            last_error = error
        except Exception as repair_error:
            last_error = str(repair_error)

    if _schema_requires_traversal_fields(constraint):
        fallback_text = _fallback_traversal_json(response_text, _valid_candidate_ids_from_prompt(prompt))
        if fallback_text:
            is_valid, error = validate_genai_response_constraint(fallback_text, constraint)
            if is_valid:
                return fallback_text
            last_error = error

    raise ValueError(f"Response does not conform to schema: {last_error}")

class BatchMetrics:
    """Class to track batch processing metrics and errors."""
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.retry_counts = Counter()
        self.completed_requests = 0
        self.total_errors = 0
        self.active_requests = 0
        self.start_time = time.time()
        self.error_details = []
        self.success_details = []
        
    def add_error(self, error_type: str, error_msg: str, prompt_index: int, attempt: int):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        self.total_errors += 1
        self.error_details.append({
            'type': error_type,
            'message': error_msg,
            'prompt_index': prompt_index,
            'attempt': attempt,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_success(self, prompt_index: int, attempts_taken: int):
        """Record a successful completion."""
        self.completed_requests += 1
        self.retry_counts[attempts_taken] += 1
        self.success_details.append({
            'prompt_index': prompt_index,
            'attempts': attempts_taken,
            'timestamp': datetime.now().isoformat()
        })
    
    def increment_active(self):
        """Increment active request counter."""
        self.active_requests += 1
    
    def decrement_active(self):
        """Decrement active request counter."""
        self.active_requests = max(0, self.active_requests - 1)
    
    def get_tqdm_postfix(self) -> Dict[str, Any]:
        """Get postfix dictionary for tqdm progress bar."""
        return {
            'errors': self.total_errors,
            'active': self.active_requests,
            'completed': self.completed_requests,
            '429s': self.error_counts.get('rate_limit', 0),
            '503s': self.error_counts.get('service_unavailable', 0)
        }
    
    def print_summary_report(self, total_prompts: int, logger: logging.Logger):
        """Print comprehensive batch processing report."""
        duration = time.time() - self.start_time
        success_rate = (self.completed_requests / total_prompts) * 100 if total_prompts > 0 else 0
        
        logger.info("=" * 60)
        logger.info("BATCH PROCESSING SUMMARY REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Duration: {duration:.2f} seconds")
        logger.info(f"Total Prompts: {total_prompts}")
        logger.info(f"Successful: {self.completed_requests} ({success_rate:.1f}%)")
        logger.info(f"Failed: {total_prompts - self.completed_requests}")
        logger.info(f"Total Error Occurrences: {self.total_errors}")
        
        if self.error_counts:
            logger.info("\nERROR BREAKDOWN:")
            for error_type, count in sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / self.total_errors) * 100
                logger.info(f"  {error_type}: {count} ({percentage:.1f}%)")
        
        if self.retry_counts:
            logger.info("\nRETRY STATISTICS:")
            for attempts, count in sorted(self.retry_counts.items()):
                percentage = (count / self.completed_requests) * 100 if self.completed_requests > 0 else 0
                logger.info(f"  {attempts} attempt(s): {count} requests ({percentage:.1f}%)")
            
            avg_attempts = sum(attempts * count for attempts, count in self.retry_counts.items()) / self.completed_requests if self.completed_requests > 0 else 0
            logger.info(f"  Average attempts per successful request: {avg_attempts:.2f}")
        
        if duration > 0:
            logger.info(f"\nTHROUGHPUT:")
            logger.info(f"  Requests per second: {self.completed_requests / duration:.2f}")
            logger.info(f"  Average time per request: {duration / total_prompts:.2f} seconds")
        
        logger.info("=" * 60)

class LanguageModelAPI(ABC):
    """
    An abstract base class for interacting with various Language Model APIs.

    This class provides a standardized interface for making single and batch API
    calls, with built-in support for asynchronous operations, parallelization,
    logging, and history tracking. Subclasses must implement the backend-specific
    logic for making API calls and handling their responses.

    Attributes:
        model_name (str): The specific model to be used for the API calls.
        api_key (Optional[str]): The API key for authentication.
        timeout (int): The default timeout in seconds for API requests.
        max_retries (int): The maximum number of retries for a failed request.
        logger (logging.Logger): A logger instance for logging API interactions.
        history (List[Dict[str, Any]]): A list to record all successful interactions.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initializes the LanguageModelAPI client.

        Args:
            model_name (str): The identifier for the language model to use.
            api_key (Optional[str]): The API key for the service. It's recommended
                                     to load this from a secure source.
            timeout (int): Default request timeout in seconds.
            max_retries (int): Default maximum number of retries.
            logger (Optional[logging.Logger]): A pre-configured logger instance.
                                               If None, a default one is created.
            **kwargs: Additional keyword arguments for the specific backend.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = None # Placeholder for a potential persistent session
        self.history: List[Dict[str, Any]] = []

        # Set up logging
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            # Configure a default handler if none exists to ensure logs are visible.
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.logger.info(f"Initialized client for model: {self.model_name}")

    @abstractmethod
    async def _call_api(self, prompt: Any, **kwargs: Any) -> Any:
        """
        Makes a single, raw API call to the specific LM backend.
        """
        pass

    @abstractmethod
    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        """
        Validates the raw API response and extracts the generated text.
        """
        pass

    def _format_prompt(self, prompt: Any) -> Any:
        """
        Formats the prompt into the required structure for the backend.
        Base implementation handles strings. Subclasses should override for
        more complex formats like chat history.
        """
        if isinstance(prompt, str):
            return prompt
        # If not a string, return as-is. Subclass should handle it.
        self.logger.debug(f"Prompt type is {type(prompt)}, passing as-is. Subclass should override _format_prompt if needed.")
        return prompt

    async def run(self, prompt: Any, batch_metrics: Optional[BatchMetrics] = None, **kwargs: Any) -> str:
        """
        Executes a single prompt, logs, records, and returns the response with timeout.
        """
        # Create a log-friendly version of the prompt
        if isinstance(prompt, str):
            log_prompt = prompt
        elif isinstance(prompt, list) and prompt:
            log_prompt = f"Conversation with {len(prompt)} messages, last: {prompt[-1]}"
        else:
            log_prompt = str(prompt)

        self.logger.debug(f"Running prompt: '{log_prompt[:100].strip()}...'")
        retries = kwargs.pop("max_retries", self.max_retries)
        timeout = kwargs.pop("timeout", self.timeout)
        prompt_index = kwargs.pop("prompt_index", -1)

        if batch_metrics:
            batch_metrics.increment_active()

        try:
            for attempt in range(retries):
                try:
                    # Format the prompt before calling the API
                    formatted_prompt = self._format_prompt(prompt)

                    # Wrap the API call with asyncio.wait_for to enforce a timeout.
                    raw_response = await asyncio.wait_for(
                        self._call_api(formatted_prompt, **kwargs),
                        timeout=timeout
                    )
                    processed_response = self._validate_and_parse_response(
                        raw_response,
                        _formatted_prompt=formatted_prompt,
                        **kwargs,
                    )

                    # Record the successful interaction in history
                    history_record = {
                        "prompt": prompt,
                        "response": processed_response,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.history.append(history_record)

                    # Record success metrics
                    if batch_metrics:
                        batch_metrics.add_success(prompt_index, attempt + 1)

                    self.logger.debug(f"Success. Response: '{processed_response[:100]}...'")
                    return processed_response

                except asyncio.TimeoutError as e:
                    error_type = "timeout"
                    error_msg = f"API call timed out after {timeout}s"
                    
                    if batch_metrics:
                        batch_metrics.add_error(error_type, error_msg, prompt_index, attempt + 1)
                    
                    self.logger.warning(f"{error_msg} on attempt {attempt + 1}.")
                    if attempt + 1 >= retries:
                        self.logger.critical("Max retries reached after timeout. Raising final exception.")
                        raise

                except Exception as e:
                    error_str = str(e).lower()
                    
                    # Categorize errors
                    if "429" in error_str or "too many requests" in error_str:
                        error_type = "rate_limit"
                        wait_time = min(60 * (2 ** attempt), 300)
                    elif "503" in error_str or "service unavailable" in error_str:
                        error_type = "service_unavailable"
                        wait_time = min(30 * (2 ** attempt), 120)
                    elif "timeout" in error_str:
                        error_type = "timeout"
                        wait_time = min(10 * (2 ** attempt), 60)
                    elif "400" in error_str or "bad request" in error_str:
                        error_type = "bad_request"
                        wait_time = min(5 * (2 ** attempt), 30)
                    elif "auth" in error_str or "401" in error_str or "403" in error_str:
                        error_type = "authentication"
                        wait_time = min(5 * (2 ** attempt), 30)
                    else:
                        error_type = error_str
                        wait_time = min(1 * (4 ** attempt), timeout)
                    
                    if batch_metrics:
                        batch_metrics.add_error(error_type, str(e), prompt_index, attempt + 1)
                    
                    self.logger.debug(f"API call failed on attempt {attempt + 1} - {error_type}: {str(e)}")
                    
                    if attempt + 1 >= retries:
                        self.logger.critical(f"Max retries reached. Final error type: {error_type}")
                        raise

                    # Wait before retrying with categorized backoff
                    if attempt + 1 < retries:
                        self.logger.debug(f"Waiting {wait_time}s before retry {attempt + 2}")
                        await asyncio.sleep(wait_time)

        finally:
            if batch_metrics:
                batch_metrics.decrement_active()

        return ""  # Should not be reached

    async def run_batch(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Executes a batch of prompts concurrently, with optional rate limiting and comprehensive metrics.

        Args:
            prompts (List[str]): A list of prompts to execute.
            max_concurrent_calls (Optional[int]): The maximum number of API calls to run in parallel.
                                                   If None, all prompts are run concurrently.
            **kwargs: Backend-specific parameters to apply to all prompts in the batch.

        Returns:
            A list of generated texts, in the same order as the input prompts.
        """
        self.logger.info(f"Running a batch of {len(prompts)} prompts...")
        
        # Initialize metrics tracking
        metrics = BatchMetrics()
        max_concurrent_calls = kwargs.pop('max_concurrent_calls', None)
        print_summary_report = kwargs.pop('print_summary_report', True)
        staggering_delay = kwargs.pop('staggering_delay', 0.1)
        
        if max_concurrent_calls:
            semaphore = asyncio.Semaphore(max_concurrent_calls)
            self.logger.info(f"Concurrency limited to {max_concurrent_calls} parallel calls.")

            async def _semaphore_run(prompt: str, index: int) -> str:
                async with semaphore:
                    # Add small delay to space out requests
                    await asyncio.sleep((index % max_concurrent_calls) * staggering_delay)
                    return await self.run(prompt, batch_metrics=metrics, prompt_index=index, **kwargs)

            tasks = [
                asyncio.create_task(_run_and_return_index(i, _semaphore_run(prompt, i)))
                for i, prompt in enumerate(prompts)
            ]
        else:
            # Handle case without semaphore
            tasks = [
                asyncio.create_task(_run_and_return_index(i, self.run(
                    prompt, 
                    batch_metrics=metrics, 
                    prompt_index=i,
                    **{k: (v[i] if (isinstance(v, list) or isinstance(v, tuple)) else v) for k, v in kwargs.items()}
                )))
                for i, prompt in enumerate(prompts)
            ]

        # Create results placeholder
        results = [None] * len(prompts)

        # Process tasks with progress bar showing metrics
        with tqdm(total=len(tasks), desc="Processing batch") as pbar:
            for future in asyncio.as_completed(tasks):
                index, result_or_exception = await future
                results[index] = result_or_exception
                
                # Update progress bar with current metrics
                pbar.set_postfix(metrics.get_tqdm_postfix())
                pbar.update(1)

        # Process final results
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.warning(f"Prompt {i+1} in batch failed: {str(res)}")
                final_results.append(f"Error: {res}")
            else:
                final_results.append(res)

        # Print comprehensive summary report
        if print_summary_report:
            metrics.print_summary_report(len(prompts), self.logger)
        
        return final_results

#@title `GenAIAPI` implementation
class GenAIAPI(LanguageModelAPI):
    """
    Google Generative AI implementation of the LanguageModelAPI.
    
    Supports both string prompts and chat-style conversation lists.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        **kwargs: Any,
    ):
        """
        Initialize Google GenAI API client.
        
        Args:
            model_name (str): Model to use (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key (Optional[str]): Google API key. If None, loads from GOOGLE_API_KEY env var
            timeout (int): Request timeout in seconds
            max_retries (int): Maximum retry attempts
            logger (Optional[logging.Logger]): Logger instance
            **kwargs: Additional config parameters (temperature, max_output_tokens, etc.)
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("API key must be provided or set in GOOGLE_API_KEY environment variable")
        
        super().__init__(model_name, api_key, timeout, max_retries, logger, **kwargs)
        
        # Initialize Google GenAI client
        self.client = genai.Client(api_key=self.api_key)
        
        # Store default config parameters
        self.default_config = {}
        
        self.logger.info(f"Initialized Google GenAI client with model: {self.model_name}")
    
    def _format_prompt(self, prompt: Any) -> str:
        """
        Format prompt for Google GenAI API.
        
        Args:
            prompt: Can be a string or list of chat messages
            
        Returns:
            Formatted prompt string
        """
        if isinstance(prompt, str):
            return prompt
        elif isinstance(prompt, list):
            # Handle chat-style conversations
            # Convert list of messages to a single string
            formatted_parts = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', str(msg))
                    formatted_parts.append(f"{role}: {content}")
                else:
                    formatted_parts.append(str(msg))
            return "\n".join(formatted_parts)
        else:
            self.logger.warning(f"Unexpected prompt type: {type(prompt)}, converting to string")
            return str(prompt)
    
    async def _call_api(self, prompt: str, **kwargs: Any) -> Any:
        """
        Make async API call to Google GenAI.
        
        Args:
            prompt (str): Formatted prompt string
            **kwargs: Additional parameters for this specific call
            
        Returns:
            Raw API response
        """
        # Merge default config with call-specific parameters
        config_params = {**self.default_config, **kwargs}
        
        # Remove non-config parameters that might be passed from parent class
        config_params.pop('max_retries', None)
        config_params.pop('timeout', None)
        
        # Create config object
        config = GenerateContentConfig(**config_params) if config_params else None
        
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            return response
        except Exception as e:
            self.logger.debug(f"Google GenAI API call failed: {str(e)}")
            raise
    
    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        """
        Validate and extract text from Google GenAI response.
        
        Args:
            response: Raw API response object
            
        Returns:
            Generated text string
            
        Raises:
            ValueError: If response is invalid or empty
        """
        constraint = kwargs.get('response_schema', None)
        try:
            if not response:
                raise ValueError("Empty response from API")
            
            if not hasattr(response, 'text'):
                raise ValueError("Response missing 'text' attribute")
            
            text = response.text
            if not text or not text.strip():
                raise ValueError("Empty text in response")

            if constraint:
                is_valid, error = validate_genai_response_constraint(text, constraint)
                if not is_valid:
                    raise ValueError(f"Response does not conform to schema: {error}")
                
            return text.strip()
            
        except Exception as e:
            self.logger.debug(f"Failed to parse response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")
    
    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get usage information from the last response in history.
        
        Returns:
            Dictionary with usage metadata if available
        """
        if not self.history:
            return {}
        
        last_entry = self.history[-1]
        return last_entry.get('usage', {})
    
    def update_config(self, **kwargs: Any) -> None:
        """
        Update default configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        self.default_config.update(kwargs)
        self.logger.info(f"Updated config: {kwargs}")

#@title `OpenAIResponsesAPI` implementation
class OpenAIResponsesAPI(LanguageModelAPI):
    """
    OpenAI Responses API implementation of the LanguageModelAPI.

    Supports string prompts and structured JSON outputs.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        log_api_calls: bool = False,
        **kwargs: Any,
    ):
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("API key must be provided or set in OPENAI_API_KEY environment variable")

        super().__init__(model_name, api_key, timeout, max_retries, logger, **kwargs)

        self.client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        self.default_config = {}
        self.log_api_calls = log_api_calls
        self.api_call_logger, self.api_call_log_path = _build_api_call_logger(
            self.logger,
            logger_name_suffix="openai_api_calls",
            file_suffix="openai_api",
        ) if self.log_api_calls else (None, None)

        self.logger.info(
            "Initialized OpenAI Responses client with model: %s (log_api_calls=%s)",
            self.model_name,
            self.log_api_calls,
        )
        if self.log_api_calls and self.api_call_log_path is not None:
            self.logger.info("OpenAI prompt/response log file: %s", self.api_call_log_path)

    def _log_prompt_and_response(self, prompt: Any, response_text: str) -> None:
        if not self.log_api_calls or self.api_call_logger is None:
            return

        try:
            prompt_text = json.dumps(prompt, ensure_ascii=False, indent=2)
        except Exception:
            prompt_text = str(prompt)

        self.api_call_logger.info(
            "%s\nPrompt:\n%s\nResponse:\n%s\n%s",
            "=" * 80,
            prompt_text,
            response_text,
            "=" * 80,
        )

    def _format_prompt(self, prompt: Any) -> Any:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            return prompt
        self.logger.warning(f"Unexpected prompt type: {type(prompt)}, converting to string")
        return str(prompt)

    async def _call_api(self, prompt: Any, **kwargs: Any) -> Any:
        config_params = {**self.default_config, **kwargs}

        config_params.pop('max_retries', None)
        config_params.pop('timeout', None)
        config_params.pop('prompt_index', None)
        config_params.pop('response_mime_type', None)
        config_params.pop('thinking_config', None)

        response_schema = config_params.pop('response_schema', None)

        request_params = {
            'model': self.model_name,
            'input': prompt,
            **config_params,
        }

        if response_schema:
            request_params['text'] = {
                'format': normalize_openai_schema(response_schema),
            }

        try:
            response = await self.client.responses.create(**request_params)
            self._log_prompt_and_response(prompt, _extract_openai_response_text(response))
            return response
        except Exception as e:
            self.logger.debug(f"OpenAI Responses API call failed: {str(e)}")
            raise

    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        constraint = kwargs.get('response_schema', None)
        try:
            if not response:
                raise ValueError("Empty response from API")

            text = _extract_openai_response_text(response)
            if not text:
                raise ValueError("Empty text in response")

            if constraint:
                is_valid, error = validate_genai_response_constraint(text, constraint)
                if not is_valid:
                    raise ValueError(f"Response does not conform to schema: {error}")

            return text

        except Exception as e:
            self.logger.debug(f"Failed to parse response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    def update_config(self, **kwargs: Any) -> None:
        self.default_config.update(kwargs)
        self.logger.info(f"Updated config: {kwargs}")

#@title `LocalModelAPI` implementation
class LocalModelAPI(LanguageModelAPI):
    """
    Local in-process model implementation of the LanguageModelAPI.

    This backend wraps a single locally loaded Hugging Face/PEFT model and exposes
    it through the same async interface used by the remote API backends.
    """

    def __init__(
        self,
        model_name: str,
        timeout: int = 60,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        adapter_path: Optional[str] = None,
        use_4bit: bool = True,
        enable_thinking: bool = False,
        serialize_requests: bool = True,
        log_api_calls: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, api_key=None, timeout=timeout, max_retries=max_retries, logger=logger, **kwargs)

        from llm_rl_playground.local_retrieval_model import load_local_retrieval_model

        self.runtime = load_local_retrieval_model(
            model_id=self.model_name,
            adapter_path=adapter_path,
            use_4bit=use_4bit,
            enable_thinking=enable_thinking,
        )
        self.default_config: Dict[str, Any] = {}
        self.serialize_requests = serialize_requests
        self._request_lock: Optional[asyncio.Lock] = None
        self._request_lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self.log_api_calls = log_api_calls
        self.api_call_logger, self.api_call_log_path = _build_api_call_logger(
            self.logger,
            logger_name_suffix="local_model_api_calls",
            file_suffix="local_model_api",
        ) if self.log_api_calls else (None, None)

        self.logger.info(
            "Initialized local model client with model: %s (adapter_path=%s, use_4bit=%s, serialize_requests=%s, log_api_calls=%s)",
            self.model_name,
            adapter_path,
            use_4bit,
            serialize_requests,
            self.log_api_calls,
        )
        if self.log_api_calls and self.api_call_log_path is not None:
            self.logger.info("Local model prompt/response log file: %s", self.api_call_log_path)

    def _log_prompt_and_response(self, prompt: List[Dict[str, str]], response: str) -> None:
        """
        Write the local prompt and generated response to the active logger.

        This uses a dedicated file logger so prompt/response dumps stay separate
        from the main run log.
        """
        if not self.log_api_calls or self.api_call_logger is None:
            return

        try:
            prompt_text = json.dumps(prompt, ensure_ascii=False, indent=2)
        except Exception:
            prompt_text = str(prompt)

        self.api_call_logger.info(
            "%s\nPrompt:\n%s\nResponse:\n%s\n%s",
            "=" * 80,
            prompt_text,
            response,
            "=" * 80,
        )

    def _get_request_lock(self) -> Optional[asyncio.Lock]:
        """
        Return a lock bound to the currently running event loop.

        Notebook environments like Colab can recreate the event loop between
        cells, so a lock created once in `__init__` may later belong to the
        wrong loop and raise `is bound to a different event loop`.
        """
        if not self.serialize_requests:
            return None

        current_loop = asyncio.get_running_loop()
        if self._request_lock is None or self._request_lock_loop is not current_loop:
            self._request_lock = asyncio.Lock()
            self._request_lock_loop = current_loop
        return self._request_lock

    def _format_prompt(self, prompt: Any) -> List[Dict[str, str]]:
        """
        Format prompt for the local model in chat-message form.

        Args:
            prompt: Can be a string or a list of chat messages.

        Returns:
            List of message dictionaries in chat format.
        """
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, list):
            formatted_messages = []
            for msg in prompt:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    formatted_messages.append(msg)
                elif isinstance(msg, dict):
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", str(msg)),
                    })
                else:
                    formatted_messages.append({"role": "user", "content": str(msg)})
            return formatted_messages

        self.logger.warning(f"Unexpected prompt type: {type(prompt)}, converting to string")
        return [{"role": "user", "content": str(prompt)}]

    async def _call_api(self, prompt: List[Dict[str, str]], **kwargs: Any) -> Any:
        """
        Run one local generation call in a worker thread so the async loop remains responsive.
        """
        config_params = {**self.default_config, **kwargs}
        response_schema = config_params.get("response_schema")

        # Remove framework-only parameters that are not generation arguments.
        config_params.pop("max_retries", None)
        config_params.pop("timeout", None)
        config_params.pop("prompt_index", None)
        config_params.pop("response_mime_type", None)
        config_params.pop("response_schema", None)
        config_params.pop("staggering_delay", None)
        config_params.pop("print_summary_report", None)
        config_params.pop("max_concurrent_calls", None)

        max_new_tokens = config_params.pop("max_tokens", config_params.pop("max_new_tokens", 256))
        default_temperature = 0.0 if response_schema else 0.7
        default_top_p = 1.0 if response_schema else 0.9
        temperature = config_params.pop("temperature", default_temperature)
        top_p = config_params.pop("top_p", default_top_p)
        enable_thinking = config_params.pop("enable_thinking", None)
        thinking_config = config_params.pop("thinking_config", None)
        if enable_thinking is None and thinking_config is not None:
            thinking_budget = getattr(thinking_config, "thinking_budget", None)
            enable_thinking = thinking_budget is None or thinking_budget != 0
        if response_schema and enable_thinking is None:
            enable_thinking = False

        prompt = _inject_schema_instruction(prompt, response_schema)

        async def _generate() -> str:
            response = await asyncio.to_thread(
                self.runtime.chat,
                messages=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
            self._log_prompt_and_response(prompt, response)
            return response

        request_lock = self._get_request_lock()
        if request_lock is not None:
            async with request_lock:
                return await _generate()
        return await _generate()

    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        """
        Validate and extract text from the local model response.
        """
        constraint = kwargs.get("response_schema", None)
        try:
            if not isinstance(response, str):
                raise ValueError(f"Local model response must be a string, got {type(response)}")

            text = response.strip()
            if not text:
                raise ValueError("Empty text in response")

            if constraint:
                text = _coerce_text_to_schema_json(
                    text,
                    constraint,
                    prompt=kwargs.get("_formatted_prompt"),
                )

            return text

        except Exception as e:
            preview = response[:500] if isinstance(response, str) else repr(response)
            self.logger.debug(f"Failed to parse local response: {str(e)} | raw={preview}")
            raise ValueError(f"Invalid local response format: {str(e)}")

    def update_config(self, **kwargs: Any) -> None:
        self.default_config.update(kwargs)
        self.logger.info(f"Updated config: {kwargs}")

    def unload(self) -> None:
        """Release the in-process model runtime when the caller is done using it."""
        self.runtime.unload()

#@title `VllmAPI` implementation
class VllmAPI(LanguageModelAPI):
    """
    vLLM implementation of the LanguageModelAPI.

    vLLM is a fast and memory-efficient inference and serving engine for LLMs.
    This implementation uses the OpenAI-compatible API interface that vLLM provides.

    Supports both string prompts and chat-style conversation lists.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = "EMPTY",
        timeout: int = 60,
        max_retries: int = 3,
        logger: Optional[logging.Logger] = None,
        load_balance: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize vLLM API client.

        Args:
            model_name (str): Model to use (should match the model served by vLLM)
            base_url (str): Base URL of the vLLM server (default: "http://localhost:8000/v1")
                           Can be a comma-separated list of URLs for load balancing
            api_key (Optional[str]): API key. vLLM typically doesn't require authentication,
                                     so "EMPTY" is used by default
            timeout (int): Request timeout in seconds
            max_retries (int): Maximum retry attempts
            logger (Optional[logging.Logger]): Logger instance
            load_balance (bool): Enable round-robin load balancing across multiple servers
            **kwargs: Additional config parameters (temperature, max_tokens, top_p, etc.)
        """
        # Get base URL from environment if available
        base_url = os.getenv('VLLM_BASE_URL', base_url)

        super().__init__(model_name, api_key, timeout, max_retries, logger, **kwargs)

        # Handle multiple base URLs for load balancing
        self.base_urls = [url.strip() for url in base_url.split(',')]
        self.load_balance = load_balance or len(self.base_urls) > 1
        self.current_url_index = 0

        # Initialize OpenAI clients for each server
        if self.load_balance:
            self.clients = [
                AsyncOpenAI(base_url=url, api_key=self.api_key, timeout=self.timeout)
                for url in self.base_urls
            ]
            self.logger.info(f"Initialized vLLM client with load balancing across {len(self.base_urls)} servers")
            for i, url in enumerate(self.base_urls):
                self.logger.info(f"  Server {i}: {url}")
        else:
            self.client = AsyncOpenAI(
                base_url=self.base_urls[0],
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self.logger.info(f"Initialized vLLM client with model: {self.model_name} at {self.base_urls[0]}")

        # Store default config parameters
        self.default_config = {}

    def _format_prompt(self, prompt: Any) -> List[Dict[str, str]]:
        """
        Format prompt for vLLM API (OpenAI-compatible format).

        Args:
            prompt: Can be a string or list of chat messages

        Returns:
            List of message dictionaries in OpenAI chat format
        """
        if isinstance(prompt, str):
            # Convert simple string to chat format
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Handle chat-style conversations
            formatted_messages = []
            for msg in prompt:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Already in correct format
                    formatted_messages.append(msg)
                elif isinstance(msg, dict):
                    # Has dict structure but might need reformatting
                    role = msg.get('role', 'user')
                    content = msg.get('content', str(msg))
                    formatted_messages.append({"role": role, "content": content})
                else:
                    # Convert other types to user messages
                    formatted_messages.append({"role": "user", "content": str(msg)})
            return formatted_messages
        else:
            self.logger.warning(f"Unexpected prompt type: {type(prompt)}, converting to string")
            return [{"role": "user", "content": str(prompt)}]

    def _get_client(self) -> AsyncOpenAI:
        """
        Get the appropriate client for the next request.
        Uses round-robin load balancing if multiple servers are configured.

        Returns:
            AsyncOpenAI client instance
        """
        if not self.load_balance:
            return self.client

        # Round-robin load balancing
        client = self.clients[self.current_url_index]
        self.current_url_index = (self.current_url_index + 1) % len(self.clients)
        return client

    async def _call_api(self, prompt: List[Dict[str, str]], **kwargs: Any) -> Any:
        """
        Make async API call to vLLM server.

        Args:
            prompt (List[Dict[str, str]]): Formatted prompt as list of message dicts
            **kwargs: Additional parameters for this specific call

        Returns:
            Raw API response
        """
        # Merge default config with call-specific parameters
        config_params = {**self.default_config, **kwargs}

        # Remove non-API parameters that might be passed from parent class
        config_params.pop('max_retries', None)
        config_params.pop('timeout', None)
        config_params.pop('prompt_index', None)

        # Handle response_schema for constrained decoding
        response_schema = config_params.pop('response_schema', None)
        guided_decoding_backend = config_params.pop('guided_decoding_backend', None)

        if response_schema:
            # Convert to vLLM's guided_json format
            # vLLM expects the schema as a JSON object
            import json

            # Prepare extra_body for guided generation
            extra_body = config_params.get('extra_body', {})

            if isinstance(response_schema, dict):
                extra_body['guided_json'] = response_schema
            elif isinstance(response_schema, str):
                extra_body['guided_json'] = json.loads(response_schema)

            # Optionally specify the backend (outlines, lm-format-enforcer, or xgrammar)
            # If not specified, vLLM will use its default
            if guided_decoding_backend:
                extra_body['guided_decoding_backend'] = guided_decoding_backend

            config_params['extra_body'] = extra_body
            self.logger.debug(f"Using guided JSON generation with schema")

        try:
            # Get client (handles load balancing if enabled)
            client = self._get_client()

            response = await client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **config_params
            )
            return response
        except Exception as e:
            self.logger.debug(f"vLLM API call failed: {str(e)}")
            raise

    def _validate_and_parse_response(self, response: Any, **kwargs) -> str:
        """
        Validate and extract text from vLLM response.

        Args:
            response: Raw API response object (OpenAI-compatible format)
            **kwargs: Additional validation parameters

        Returns:
            Generated text string

        Raises:
            ValueError: If response is invalid or empty
        """
        try:
            if not response:
                raise ValueError("Empty response from API")

            if not hasattr(response, 'choices') or not response.choices:
                raise ValueError("Response missing 'choices' attribute or choices are empty")

            # Extract text from first choice
            choice = response.choices[0]
            if not hasattr(choice, 'message'):
                raise ValueError("Choice missing 'message' attribute")

            message = choice.message
            if not hasattr(message, 'content'):
                raise ValueError("Message missing 'content' attribute")

            text = message.content
            if not text or not text.strip():
                raise ValueError("Empty text in response")

            # Optional: validate against response schema if provided
            constraint = kwargs.get('response_schema', None)
            if constraint:
                is_valid, error = validate_genai_response_constraint(text, constraint)
                if not is_valid:
                    raise ValueError(f"Response does not conform to schema: {error}")

            return text.strip()

        except Exception as e:
            self.logger.debug(f"Failed to parse response: {str(e)}")
            raise ValueError(f"Invalid response format: {str(e)}")

    def get_usage_info(self) -> Dict[str, Any]:
        """
        Get usage information from the last response in history.

        Returns:
            Dictionary with usage metadata if available
        """
        if not self.history:
            return {}

        last_entry = self.history[-1]
        return last_entry.get('usage', {})

    def update_config(self, **kwargs: Any) -> None:
        """
        Update default configuration parameters.

        Common parameters for vLLM:
        - temperature: Float (0.0 to 2.0)
        - max_tokens: Integer
        - top_p: Float (0.0 to 1.0)
        - top_k: Integer
        - frequency_penalty: Float (-2.0 to 2.0)
        - presence_penalty: Float (-2.0 to 2.0)
        - stop: List[str] or str

        Args:
            **kwargs: Configuration parameters to update
        """
        self.default_config.update(kwargs)
        self.logger.info(f"Updated config: {kwargs}")
