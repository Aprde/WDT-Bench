"""Prompt construction for the Diagnostic Tests (Tests 4-6).

Six narrative templates exist in both English and Chinese; the prompt id
selects the template (the paper uses ``prompt_id = 0``).
"""
from __future__ import annotations

from typing import Sequence, Tuple

DemoPair = Tuple[str, str]

_SUFFIX_EN = [
    "John developed a very special way of speaking. ",
]
_POSTFIX_EN = [
    "Please carefully examine John's speaking style, and guess what he would say for the sentence: ",
]
_INSTRUCTION_EN = [
    ".",
]
_EXAMPLE_EN = [
    ("For ", ", he would say "),
]

_SUFFIX_ZH = [
    "小张有一种非常奇特的说话方式。",
]
_POSTFIX_ZH = [
    "请你认真分析小张的说话方式，然后猜测他会如何表达",
]
_INSTRUCTION_ZH = [
    "这个句子。",
]
_EXAMPLE_ZH = [
    ("对于", "，他会说成"),
]


N_PROMPT_VARIANTS = len(_SUFFIX_EN)


def build_prompt(
    demonstrations: Sequence[DemoPair],
    test_sentence: str,
    language: str,
    prompt_id: int = 0,
) -> str:
    """Construct the few-shot diagnostic prompt.

    ``demonstrations`` is a list of ``(sentence, deleted_form)`` pairs;
    ``language`` is ``"english"`` or ``"chinese"``.
    """
    if language == "chinese":
        suffix, postfix, instruction, example = _SUFFIX_ZH, _POSTFIX_ZH, _INSTRUCTION_ZH, _EXAMPLE_ZH
        q_open, q_close, sep = "\u201c", "\u201d", "\u3002"
    else:
        suffix, postfix, instruction, example = _SUFFIX_EN, _POSTFIX_EN, _INSTRUCTION_EN, _EXAMPLE_EN
        q_open, q_close, sep = "'", "'", "."

    demo_text = "".join(
        f"{example[prompt_id][0]}{q_open}{sent}{q_close}{example[prompt_id][1]}{q_open}{label}{q_close}{sep}\n"
        for sent, label in demonstrations
    )

    prompt = (
        f"{suffix[prompt_id]}{demo_text}{postfix[prompt_id]}"
        f"{q_open}{test_sentence}{q_close}{instruction[prompt_id]}"
    )
    if language == "chinese":
        prompt += "\n请直接给出答案，不要解释。"
    else:
        prompt += "\nAnswer directly without explanation."
    return prompt
