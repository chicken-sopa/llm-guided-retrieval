from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_SAMPLES = [
    {
        "user": "Summarize the idea of retrieval-augmented generation in one sentence.",
        "assistant": "Retrieval-augmented generation combines a language model with external document retrieval so responses can use relevant grounded context.",
    },
    {
        "user": "Classify the sentiment of: 'The paper is ambitious and surprisingly clear.'",
        "assistant": "Positive",
    },
    {
        "user": "Rewrite this title to sound more formal: 'Cool tricks for better search.'",
        "assistant": "Practical Techniques for Improving Search Performance",
    },
    {
        "user": "Answer briefly: what does overfitting mean in machine learning?",
        "assistant": "Overfitting happens when a model memorizes training patterns too closely and performs poorly on new data.",
    },
    {
        "user": "Extract the country from: 'Lisbon, Portugal'.",
        "assistant": "Portugal",
    },
    {
        "user": "Turn this note into a polite email opener: 'need the report by friday'.",
        "assistant": "I hope you are doing well. Could you please send me the report by Friday?",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 2 2B Instruct with QLoRA.")
    parser.add_argument(
        "--model-id",
        default="google/gemma-2-2b-it",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to a JSON or JSONL file. Supported formats: user/assistant, prompt/completion, or messages.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/gemma-2-2b-it-qlora",
        help="Directory to save adapters and tokenizer.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_instruction_dataset(data_path: str | None) -> Dataset:
    if data_path is None:
        return Dataset.from_list(DEFAULT_SAMPLES)
    return load_dataset("json", data_files=data_path, split="train")


def normalize_completion(example: dict) -> tuple[list[dict[str, str]], str]:
    if "messages" in example:
        messages = example["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("When using `messages`, the last message must be from the assistant.")
        prompt_messages = messages[:-1]
        assistant_text = messages[-1]["content"]
        return prompt_messages, assistant_text

    if "user" in example and "assistant" in example:
        return [{"role": "user", "content": example["user"]}], example["assistant"]

    if "prompt" in example and "completion" in example:
        prompt = example["prompt"]
        completion = example["completion"]

        if isinstance(prompt, str):
            prompt_messages = [{"role": "user", "content": prompt}]
        else:
            prompt_messages = prompt

        if isinstance(completion, str):
            assistant_text = completion
        else:
            assistant_parts = [item["content"] for item in completion if item["role"] == "assistant"]
            assistant_text = "\n".join(assistant_parts)

        return prompt_messages, assistant_text

    raise ValueError(
        "Unsupported dataset format. Expected `user`/`assistant`, `prompt`/`completion`, or `messages`."
    )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training here expects a CUDA GPU.")

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw_dataset = load_instruction_dataset(args.data_path)

    def preprocess(example: dict) -> dict:
        prompt_messages, assistant_text = normalize_completion(example)
        full_messages = prompt_messages + [{"role": "assistant", "content": assistant_text}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=args.max_length,
        )

        input_ids = full_tokens["input_ids"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": full_tokens["attention_mask"],
            "labels": labels,
        }

    tokenized_dataset = raw_dataset.map(
        preprocess,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing dataset",
    )
    tokenized_dataset = tokenized_dataset.filter(
        lambda example: any(label != -100 for label in example["labels"]),
        desc="Dropping fully masked examples",
    )

    if len(tokenized_dataset) >= 10:
        split = tokenized_dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        evaluation_strategy = "epoch"
    else:
        train_dataset = tokenized_dataset
        eval_dataset = None
        evaluation_strategy = "no"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy=evaluation_strategy,
        optim="paged_adamw_8bit",
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            label_pad_token_id=-100,
        ),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    prompt_messages = [{"role": "user", "content": "Explain QLoRA in 3 short bullet points."}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][prompt_ids.shape[-1] :]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
