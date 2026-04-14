"""
training_utils.py — Helper utilities for NLG SFT dataset management.

Provides functions for loading, formatting, validating, and sampling
training data produced by data/build_nlg_training_data.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Avoid hard dependency on transformers at import time
    from transformers import PreTrainedTokenizerBase  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED_ROLES = ("system", "user", "assistant")
_MIN_ASSISTANT_LEN = 20   # chars — very short reports are likely broken


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_sft_dataset(jsonl_path: str) -> list[dict[str, Any]]:
    """Load JSONL training data produced by build_nlg_training_data.py.

    Parameters
    ----------
    jsonl_path:
        Path to a ``.jsonl`` file where each line is a JSON object with a
        ``"messages"`` key containing the SFT conversation.

    Returns
    -------
    list[dict]
        Parsed training items.  Malformed lines are silently skipped.

    Raises
    ------
    FileNotFoundError
        If *jsonl_path* does not exist.
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {jsonl_path}")

    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                # Log at warning level if a logger is available; otherwise skip silently
                pass

    return items


def format_for_qwen(
    item: dict[str, Any],
    tokenizer: "PreTrainedTokenizerBase",
    add_generation_prompt: bool = False,
) -> dict[str, Any]:
    """Format a training item for Qwen3 SFT using ``apply_chat_template``.

    Parameters
    ----------
    item:
        A single training dict with ``"messages"`` list of role/content dicts.
    tokenizer:
        A HuggingFace tokenizer that supports ``apply_chat_template``.
    add_generation_prompt:
        Whether to append a generation-trigger suffix (set ``True`` for
        inference, ``False`` for SFT training where labels include the
        assistant turn).

    Returns
    -------
    dict with keys:
        ``"input_ids"`` (list[int])
            Token IDs of the formatted conversation.
        ``"text"``  (str)
            The raw formatted string (for inspection / debugging).
    """
    messages = item.get("messages", [])
    text: str = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    input_ids: list[int] = tokenizer(text, return_tensors=None)["input_ids"]
    return {
        "input_ids": input_ids,
        "text": text,
    }


def validate_training_item(item: dict[str, Any]) -> tuple[bool, str]:
    """Validate that a training item has the correct format and non-empty fields.

    Checks performed
    ----------------
    1. Top-level ``"messages"`` key exists and is a list of length >= 3.
    2. Roles appear in order: ``system``, ``user``, ``assistant``.
    3. Each message has a non-empty ``"content"`` string.
    4. The assistant turn is at least ``_MIN_ASSISTANT_LEN`` characters long.

    Parameters
    ----------
    item:
        Training item dict to validate.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` if valid; ``(False, reason)`` if invalid.
    """
    if not isinstance(item, dict):
        return False, "Item is not a dict"

    messages = item.get("messages")
    if not isinstance(messages, list):
        return False, "'messages' key missing or not a list"

    if len(messages) < 3:
        return False, f"Expected at least 3 messages, got {len(messages)}"

    # Check role order
    for i, expected_role in enumerate(_REQUIRED_ROLES):
        msg = messages[i]
        if not isinstance(msg, dict):
            return False, f"messages[{i}] is not a dict"
        actual_role = msg.get("role", "")
        if actual_role != expected_role:
            return False, (
                f"messages[{i}].role expected '{expected_role}', got '{actual_role}'"
            )
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            return False, f"messages[{i}].content is empty or not a string"

    # Check assistant response length
    assistant_content: str = messages[2].get("content", "")
    if len(assistant_content.strip()) < _MIN_ASSISTANT_LEN:
        return False, (
            f"Assistant response too short: {len(assistant_content)} chars "
            f"(minimum {_MIN_ASSISTANT_LEN})"
        )

    return True, ""


def sample_training_items(jsonl_path: str, n: int = 5) -> list[dict[str, Any]]:
    """Sample and pretty-print *n* training items for inspection.

    Parameters
    ----------
    jsonl_path:
        Path to the training JSONL file.
    n:
        Number of samples to return.

    Returns
    -------
    list[dict]
        Up to *n* randomly sampled training items.

    Side effects
    ------------
    Prints a formatted summary of each sampled item to stdout.
    """
    items = load_sft_dataset(jsonl_path)
    if not items:
        print(f"[sample_training_items] No items found in {jsonl_path}")
        return []

    sample = random.sample(items, min(n, len(items)))

    for idx, item in enumerate(sample, start=1):
        messages = item.get("messages", [])
        print(f"\n{'=' * 60}")
        print(f"Sample {idx}/{len(sample)}")
        print(f"{'=' * 60}")
        for msg in messages:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            # Truncate long fields for readability
            preview = content[:300] + ("..." if len(content) > 300 else "")
            print(f"\n[{role.upper()}]\n{preview}")

    return sample
