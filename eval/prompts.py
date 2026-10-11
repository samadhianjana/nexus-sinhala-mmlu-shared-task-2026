"""Prompt construction. Options are numbered 1..N because Codabench answers are digits ("1".."4").

Every model gets the same template text; only the wrapping differs (chat template vs raw text).
Few-shot examples are a fixed list, identical for every question (shared-task rule).
"""
from __future__ import annotations

TEMPLATES = {
    "en": {
        "header": "The following is a multiple choice question about {subject}. "
                  "Reply with only the number (1-{n}) of the correct answer.",
        "answer": "Answer:",
    },
    # Sinhala wording should be reviewed by a native speaker.
    "si": {
        "header": "පහත දැක්වෙන්නේ {subject} පිළිබඳ බහුවරණ ප්‍රශ්නයකි. "
                  "නිවැරදි පිළිතුරේ අංකය (1-{n}) පමණක් ලියන්න.",
        "answer": "පිළිතුර:",
    },
}


def subject_name(item: dict, lang: str) -> str:
    if lang == "si" and item.get("meta", {}).get("subject_original"):
        return item["meta"]["subject_original"]
    return item["subject"].replace("_", " ")


def question_block(question: str, options: list[str]) -> str:
    lines = [question] + [f"{i}. {opt}" for i, opt in enumerate(options, 1)]
    return "\n".join(lines)


def user_text(item: dict, options: list[str], lang: str) -> str:
    t = TEMPLATES[lang]
    header = t["header"].format(subject=subject_name(item, lang), n=len(options))
    return f"{header}\n\n{question_block(item['question'], options)}"


class PromptBuilder:
    def __init__(self, tokenizer, lang: str = "en", chat: bool = True,
                 fewshot: list[dict] | None = None, answer_prefix: str = "",
                 chat_kwargs: dict | None = None):
        self.tok, self.lang, self.chat = tokenizer, lang, chat
        self.fewshot = fewshot or []
        self.answer_prefix = answer_prefix
        self.chat_kwargs = {"enable_thinking": False, **(chat_kwargs or {})}

    @property
    def add_special_tokens(self) -> bool:
        # Chat templates already insert BOS; raw prompts need the tokenizer to add it.
        return not self.chat

    def mcq(self, item: dict, options: list[str]) -> str:
        """Prompt whose next token should be the answer number."""
        ans = TEMPLATES[self.lang]["answer"]
        if self.chat:
            msgs = []
            for ex in self.fewshot:
                msgs.append({"role": "user", "content": user_text(ex, ex["options"], self.lang)})
                msgs.append({"role": "assistant", "content": str(ex["answer"])})
            msgs.append({"role": "user", "content": user_text(item, options, self.lang)})
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  **self.chat_kwargs)
            return prompt + self.answer_prefix
        shots = [f"{user_text(ex, ex['options'], self.lang)}\n{ans} {ex['answer']}" for ex in self.fewshot]
        return "\n\n".join(shots + [f"{user_text(item, options, self.lang)}\n{ans}"])

    def cloze_context(self, item: dict) -> str:
        """Context for option-likelihood scoring: question only, options not listed."""
        return f"{item['question']}\n{TEMPLATES[self.lang]['answer']}"
