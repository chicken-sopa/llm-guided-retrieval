from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_ID = "google/gemma-2-2b-it"
DEFAULT_ADAPTER_PATH = Path(__file__).resolve().parent / "outputs" / "gemma-2-2b-it-qlora"


def resolve_compute_dtype() -> torch.dtype:
    """Pick the safest dtype for the available device."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_quantization_config(compute_dtype: torch.dtype) -> BitsAndBytesConfig:
    """Build the 4-bit loading config used for local QLoRA-style inference."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def clear_cuda_memory() -> None:
    """Release unused CUDA cache after unloading model objects."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def default_device_map() -> dict[str, int] | None:
    """Place the whole model on GPU 0 when CUDA is available."""
    if not torch.cuda.is_available():
        return None
    return {"": 0}


def build_messages(question: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """Create a chat-style message list from a user question and optional system instruction."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def render_messages_as_text(messages: list[dict[str, str]]) -> str:
    """Fallback prompt rendering for checkpoints that do not expose a chat template."""
    parts = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def is_processor_first_model(model_id: str) -> bool:
    """Gemma 4 and similar newer multimodal checkpoints prefer AutoProcessor over AutoTokenizer."""
    model_name = model_id.lower()
    return "gemma-4" in model_name


def parse_generated_text(processor: Any | None, tokenizer: Any, token_ids: torch.Tensor) -> str:
    """
    Decode generated tokens into the final assistant text.

    For Gemma 4 style processors we first decode the raw text and then try
    `processor.parse_response(...)`. If that does not yield a plain string, we
    fall back to regular tokenizer decoding.
    """
    if processor is not None:
        raw_text = processor.decode(token_ids, skip_special_tokens=False).strip()
        parse_response = getattr(processor, "parse_response", None)
        if callable(parse_response):
            try:
                parsed = parse_response(raw_text)
                if isinstance(parsed, str) and parsed.strip():
                    return parsed.strip()
                if isinstance(parsed, dict):
                    for key in ("text", "response", "content"):
                        value = parsed.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()
            except Exception:
                pass
        if raw_text:
            return raw_text

    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


@dataclass
class LocalRetrievalModelRuntime:
    model_id: str = DEFAULT_MODEL_ID
    adapter_path: str | Path | None = DEFAULT_ADAPTER_PATH
    use_4bit: bool = True
    enable_thinking: bool = False
    processor: Any | None = field(default=None, init=False)
    tokenizer: Any | None = field(default=None, init=False)
    model: Any | None = field(default=None, init=False)
    compute_dtype: torch.dtype = field(default_factory=resolve_compute_dtype, init=False)

    def _resolved_adapter_path(self) -> Path | None:
        """Return the adapter path only if it exists on disk."""
        if self.adapter_path is None:
            return None
        adapter_path = Path(self.adapter_path)
        if adapter_path.exists():
            return adapter_path
        return None

    def load(self) -> "LocalRetrievalModelRuntime":
        """
        Load tokenizer + model once and return the live runtime.

        Output:
            LocalRetrievalModelRuntime with `tokenizer` and `model` ready to use.
        """
        if self.model is not None and self.tokenizer is not None:
            return self

        if is_processor_first_model(self.model_id):
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model_kwargs: dict[str, Any] = {
            "torch_dtype": self.compute_dtype,
            "low_cpu_mem_usage": True,
        }

        if torch.cuda.is_available():
            if self.use_4bit:
                model_kwargs["quantization_config"] = build_quantization_config(self.compute_dtype)
            model_kwargs["device_map"] = default_device_map()

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        base_model.config.use_cache = True

        adapter_path = self._resolved_adapter_path()
        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
        else:
            self.model = base_model

        self.model.eval()
        return self

    def unload(self) -> None:
        """Drop loaded model/tokenizer objects and clear any unused CUDA cache."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        clear_cuda_memory()

    def prepare_prompt(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """
        Tokenize chat messages and move them to the loaded model device.

        Input:
            messages: Chat messages in `[{role, content}, ...]` format.

        Output:
            A tensor dict ready for `model.generate(**prompt)`.
        """
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

        device = _model_device(self.model)

        if self.processor is not None:
            try:
                rendered_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                rendered_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                rendered_prompt = render_messages_as_text(messages)
            prompt = self.processor(text=rendered_prompt, return_tensors="pt")
        else:
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except Exception:
                prompt = self.tokenizer(
                    render_messages_as_text(messages),
                    return_tensors="pt",
                )

        return {key: value.to(device) for key, value in prompt.items()}

    def chat(
        self,
        messages: list[dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool | None = None,
    ) -> str:
        """
        Generate one assistant reply from a full chat message list.

        Inputs:
            messages: Chat messages in `[{role, content}, ...]` format.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. Set near 0 for more deterministic output.
            top_p: Nucleus sampling threshold.

        Output:
            The decoded assistant reply as a string.
        """
        self.load()
        assert self.tokenizer is not None
        assert self.model is not None

        original_enable_thinking = self.enable_thinking
        if enable_thinking is not None:
            self.enable_thinking = enable_thinking

        prompt = self.prepare_prompt(messages)
        do_sample = temperature > 0

        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                )

            input_len = prompt["input_ids"].shape[-1]
            new_tokens = generated_ids[0][input_len:]
            return parse_generated_text(self.processor, self.tokenizer, new_tokens)
        finally:
            self.enable_thinking = original_enable_thinking

    def ask(
        self,
        question: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        enable_thinking: bool | None = None,
    ) -> str:
        """
        Ask one question without manually building the message list.

        Inputs:
            question: User question to send to the model.
            system_prompt: Optional instruction that conditions the assistant.
            max_new_tokens / temperature / top_p: Generation settings.

        Output:
            The assistant reply as a plain string.
        """
        messages = build_messages(question=question, system_prompt=system_prompt)
        return self.chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )


def load_local_retrieval_model(
    model_id: str = DEFAULT_MODEL_ID,
    adapter_path: str | Path | None = DEFAULT_ADAPTER_PATH,
    use_4bit: bool = True,
    enable_thinking: bool = False,
) -> LocalRetrievalModelRuntime:
    """
    Create and load a reusable local retrieval model runtime.

    Inputs:
        model_id: Base Hugging Face model id.
        adapter_path: Optional LoRA/QLoRA adapter directory.
        use_4bit: Whether to load the base model in 4-bit when CUDA is available.

    Output:
        A loaded `LocalRetrievalModelRuntime`.
    """
    runtime = LocalRetrievalModelRuntime(
        model_id=model_id,
        adapter_path=adapter_path,
        use_4bit=use_4bit,
        enable_thinking=enable_thinking,
    )
    return runtime.load()


def _model_device(model: Any) -> torch.device:
    """Read the device from the first model parameter."""
    return next(model.parameters()).device
