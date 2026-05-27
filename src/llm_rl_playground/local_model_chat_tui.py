from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.95
DEFAULT_USE_4BIT = True
DEFAULT_ENABLE_THINKING = False

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Terminal TUI version of local_model_chat_test_notebook.ipynb. "
            "Only the model id and optional adapter path are accepted at startup."
        )
    )
    parser.add_argument(
        "model_id",
        nargs="?",
        help=f"Hugging Face model id or local model path. Default prompt value: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "adapter_path",
        nargs="?",
        help="Optional local PEFT adapter path. Leave blank / omit for base-model chat.",
    )
    return parser.parse_args()


def ask_initial_value(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def normalize_adapter_path(raw_adapter_path: str | None) -> Path | None:
    if raw_adapter_path is None:
        return None

    value = raw_adapter_path.strip()
    if not value or value.lower() in {"none", "no", "n", "false", "-"}:
        return None

    adapter_path = Path(value).expanduser()
    if adapter_path.is_absolute():
        return adapter_path

    cwd_path = (Path.cwd() / adapter_path).resolve()
    if cwd_path.exists():
        return cwd_path

    script_path = (SCRIPT_DIR / adapter_path).resolve()
    if script_path.exists():
        return script_path

    return cwd_path


def import_runtime_loader() -> Any:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))

    try:
        from llm_rl_playground.local_retrieval_model import load_local_retrieval_model
    except ModuleNotFoundError:
        from local_retrieval_model import load_local_retrieval_model

    return load_local_retrieval_model


def terminal_size() -> tuple[int, int]:
    size = shutil.get_terminal_size(fallback=(100, 32))
    return max(size.columns, 72), max(size.lines, 20)


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def wrap_lines(text: str, width: int, indent: str = "") -> list[str]:
    lines: list[str] = []
    for raw_line in str(text).splitlines() or [""]:
        if not raw_line:
            lines.append(indent.rstrip())
            continue
        lines.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
                replace_whitespace=False,
                drop_whitespace=False,
            )
        )
    return lines


class LocalModelChatTui:
    def __init__(self, runtime: Any, model_id: str, adapter_path: Path | None) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.messages: list[dict[str, str]] = []
        self.system_prompt: str | None = None
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.enable_thinking = DEFAULT_ENABLE_THINKING
        self.status = "Ready."

    def run(self) -> None:
        self.print_banner()
        while True:
            try:
                user_input = input("\nYou > ").strip()
            except EOFError:
                self.status = "Input closed."
                return
            except KeyboardInterrupt:
                self.status = "Interrupted."
                return

            if not user_input:
                continue

            if user_input.startswith(":"):
                if self.handle_command(user_input):
                    return
                self.print_status()
                continue

            self.ask(user_input)

    def ask(self, question: str) -> None:
        chat_messages = self.chat_messages(question)
        self.status = "Generating..."
        print("\nGenerating...")
        try:
            reply = self.runtime.chat(
                messages=chat_messages,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                enable_thinking=self.enable_thinking,
            )
        except Exception as exc:
            self.status = f"Generation failed: {exc}"
            return

        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": reply})
        self.status = "Ready."
        self.print_message("Model", reply)

    def chat_messages(self, question: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.messages)
        messages.append({"role": "user", "content": question})
        return messages

    def handle_command(self, command: str) -> bool:
        name, _, value = command.partition(" ")
        name = name.lower()
        value = value.strip()

        if name in {":quit", ":exit", ":q"}:
            self.status = "Exiting."
            return True
        if name == ":help":
            self.status = self.help_text()
            return False
        if name == ":clear":
            self.messages.clear()
            self.status = "Chat history cleared."
            return False
        if name == ":system":
            self.set_system_prompt(value)
            return False
        if name == ":system-clear":
            self.system_prompt = None
            self.status = "System prompt cleared."
            return False
        if name == ":tokens":
            self.set_int_setting("max_new_tokens", value, minimum=1, maximum=4096)
            return False
        if name == ":temp":
            self.set_float_setting("temperature", value, minimum=0.0, maximum=2.0)
            return False
        if name == ":top-p":
            self.set_float_setting("top_p", value, minimum=0.01, maximum=1.0)
            return False
        if name == ":thinking":
            self.set_bool_setting("enable_thinking", value)
            return False
        if name == ":settings":
            self.status = self.settings_text()
            return False

        self.status = f"Unknown command: {name}. Type :help for commands."
        return False

    def set_system_prompt(self, value: str) -> None:
        if value:
            self.system_prompt = value
            self.status = "System prompt updated."
            return

        print("\nEnter a system prompt. Submit an empty line to keep the current value.")
        prompt = input("System > ").strip()
        if prompt:
            self.system_prompt = prompt
            self.status = "System prompt updated."
        else:
            self.status = "System prompt unchanged."

    def set_int_setting(self, attr: str, value: str, minimum: int, maximum: int) -> None:
        if not value:
            self.status = f"{attr} = {getattr(self, attr)}"
            return
        try:
            parsed = int(value)
        except ValueError:
            self.status = f"{attr} expects an integer."
            return
        if parsed < minimum or parsed > maximum:
            self.status = f"{attr} must be between {minimum} and {maximum}."
            return
        setattr(self, attr, parsed)
        self.status = f"{attr} set to {parsed}."

    def set_float_setting(self, attr: str, value: str, minimum: float, maximum: float) -> None:
        if not value:
            self.status = f"{attr} = {getattr(self, attr)}"
            return
        try:
            parsed = float(value)
        except ValueError:
            self.status = f"{attr} expects a number."
            return
        if parsed < minimum or parsed > maximum:
            self.status = f"{attr} must be between {minimum:g} and {maximum:g}."
            return
        setattr(self, attr, parsed)
        self.status = f"{attr} set to {parsed:g}."

    def set_bool_setting(self, attr: str, value: str) -> None:
        normalized = value.lower()
        if normalized in {"on", "true", "yes", "y", "1"}:
            setattr(self, attr, True)
            self.status = f"{attr} enabled."
            return
        if normalized in {"off", "false", "no", "n", "0"}:
            setattr(self, attr, False)
            self.status = f"{attr} disabled."
            return
        self.status = f"{attr} = {getattr(self, attr)}. Use on/off."

    def header_lines(self, width: int) -> list[str]:
        adapter = str(self.adapter_path) if self.adapter_path is not None else "none"
        settings = (
            f"tokens={self.max_new_tokens} temp={self.temperature:g} "
            f"top_p={self.top_p:g} thinking={'on' if self.enable_thinking else 'off'}"
        )
        return [
            "Local Model Chat TUI",
            f"model: {clip(self.model_id, width - 8)}",
            f"adapter: {clip(adapter, width - 10)}",
            settings,
            "commands: :help :settings :system :system-clear :tokens :temp :top-p",
            "          :thinking :clear :quit",
        ]

    def draw(self) -> None:
        width, height = terminal_size()
        inner_width = width - 4
        clear_screen()
        self.print_rule(width)
        for line in self.header_lines(inner_width):
            for wrapped_line in wrap_lines(line, inner_width):
                print(f"| {wrapped_line.ljust(inner_width)} |")
        self.print_rule(width)

        transcript_budget = max(height - 12, 7)
        transcript_lines = self.transcript_lines(inner_width)
        for line in transcript_lines[-transcript_budget:]:
            print(f"| {line.ljust(inner_width)} |")
        if not transcript_lines:
            empty = "Ask a question, or type :help for commands."
            print(f"| {empty.ljust(inner_width)} |")

        self.print_rule(width)
        for line in wrap_lines(self.status, inner_width, indent=""):
            print(f"| {line.ljust(inner_width)} |")
        self.print_rule(width)

    def transcript_lines(self, width: int) -> list[str]:
        lines: list[str] = []
        for message in self.messages:
            role = "You" if message["role"] == "user" else "Model"
            lines.append(f"{role}:")
            lines.extend(wrap_lines(message["content"], width, indent="  "))
            lines.append("")
        return lines

    def print_rule(self, width: int) -> None:
        print("+" + "-" * (width - 2) + "+")

    def print_banner(self) -> None:
        width, _ = terminal_size()
        inner_width = width - 4
        self.print_rule(width)
        for line in [
            "Local Model Chat TUI",
            f"model: {clip(self.model_id, inner_width - 8)}",
            f"adapter: {clip(str(self.adapter_path) if self.adapter_path is not None else 'none', inner_width - 10)}",
            "Type :help for commands, :settings for generation settings, or :quit to exit.",
        ]:
            for wrapped_line in wrap_lines(line, inner_width):
                print(f"| {wrapped_line.ljust(inner_width)} |")
        self.print_rule(width)

    def print_status(self) -> None:
        if self.status:
            self.print_message("Status", self.status)

    def print_message(self, label: str, text: str) -> None:
        width, _ = terminal_size()
        content_width = width - 4
        print(f"\n{label}:")
        for line in wrap_lines(text, content_width):
            print(line)

    def help_text(self) -> str:
        return (
            "Commands: :quit exits, :clear forgets chat history, :system <text> sets the system prompt, "
            ":system-clear removes it, :tokens <n> changes max new tokens, :temp <n> changes temperature, "
            ":top-p <n> changes nucleus sampling, :thinking on/off toggles thinking templates when supported."
        )

    def settings_text(self) -> str:
        return (
            f"max_new_tokens={self.max_new_tokens}, temperature={self.temperature:g}, "
            f"top_p={self.top_p:g}, enable_thinking={self.enable_thinking}"
        )


def main() -> None:
    args = parse_args()

    model_id = args.model_id or ask_initial_value("Model name", DEFAULT_MODEL_ID)
    raw_adapter_path = args.adapter_path
    if raw_adapter_path is None:
        raw_adapter_path = ask_initial_value("Adapter path, or none", "none")
    adapter_path = normalize_adapter_path(raw_adapter_path)

    clear_screen()
    print("Loading local model...")
    print(f"Model: {model_id}")
    print(f"Adapter: {adapter_path if adapter_path is not None else 'none'}")

    load_local_retrieval_model = import_runtime_loader()
    runtime = load_local_retrieval_model(
        model_id=model_id,
        adapter_path=adapter_path,
        use_4bit=DEFAULT_USE_4BIT,
        enable_thinking=DEFAULT_ENABLE_THINKING,
    )

    tui = LocalModelChatTui(runtime=runtime, model_id=model_id, adapter_path=adapter_path)
    try:
        tui.run()
    finally:
        runtime.unload()
        print("\nLocal model unloaded.")


if __name__ == "__main__":
    main()
