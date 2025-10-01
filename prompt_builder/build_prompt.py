import json

def _base_user_block(source: str, translated: str) -> str:
    return (
        f"Source:\n{source.strip()}\n"
        f"Translation:\n{translated.strip()}\n"
        "Evaluate and return the result."
    )
    
def build_emoji_check_prompt(source: str, translated: str):
    system_msg = {
        "role": "system",
        "content": (
            "You are a localization QA AI focusing ONLY on emoji consistency.\n"
            "If any emoji is moved, reordered, or placed in a different part of the sentence than in the source, this must be flagged as an issue.\n"
            "If any emoji is missing or added in the translation sentence compared to the source, this must be flagged as an issue.\n"
            "Ignore words, punctuation, and formatting — look ONLY at emojis (including ZWJ sequences, skin tones).\n"
            "Return strictly in this format:\n"
            "{\n"
            "  \"emoji_issue\": true|false,\n"
            "  \"suggestions\": [\"corrected translation string only\" ...]\n"
            "}\n"
            "- If emoji_issue is true, the suggestions MUST be the final translation text itself (no meta words like 'Include', 'Add', or explanations).\n"
            )
    }
    user_msg = {"role": "user", "content": _base_user_block(source, translated)}
    return system_msg, user_msg


def build_missing_check_prompt(source: str, translated: str):
    system_msg = {
        "role": "system",
        "content": (
            "You are a localization QA AI. Detect omissions at the sentence/phrase/word level over the entire document.\n"
            "[Translation Rules]\n"
            "- Flag ONLY when a source sentence/clause/single-word or multi-word phrase has no counterpart.\n"
            "- Paraphrases and verbalized relations are acceptable, allowing for slight nuances, provided the core meaning remains unchanged.\n"
            "- Reordering allowed ONLY within the same sentence; cross-sentence presence does NOT count.\n"
            "- Ignore formatting/casing/spacing not in the violation.\n"
            "- Do NOT provide parenthetical explanations.\n"
            "- If the same source sentence/clause/phrase is missing multiple times, record ALL occurrences in `missing_spans` and `revised_spans`.\n"
            "- Process sentence-by-sentence: split both texts by sentence delimiters (. ! ? and line breaks). For each source sentence S[i], search ONLY within the corresponding translation sentence(s); cross-sentence matches do NOT count.\n"
            "- Record ALL occurrences and order `missing_spans` by the source order of appearance.\n"
            "- Context-aware: Before flagging, assess the sentence-level meaning; do NOT flag if the paraphrase preserves the core meaning, but DO flag (even for a single word) if it changes facts/conditions/quantities, named entities, dates/times, amounts, references, or obligations.\n"
            "Return strictly in this format:\n"
            "{\n"
            "  \"missing_content\": true|false,\n"
            "  \"missing_spans\": [\"exact source sentence/phrase\" ...],\n"
            "  \"revised_spans\": [\"exact fragment inserted\" ...],\n"
            "  \"reasons\": [\"short reason\" ...],\n"
            "  \"suggestions\": [\"full revised translation string, preserving \\n\"]\n"
            "}\n"
            "- If false: arrays are [], no suggestions. If true: suggestions = full revised translation (no meta talk).\n"
        )
    }
    user_msg = {"role": "user", "content": _base_user_block(source, translated)}
    return system_msg, user_msg

def build_addition_check_prompt(source: str, translated: str):
    system_msg = {
        "role": "system",
        "content": (
            "You are a localization QA AI. Detect additions at the sentence/phrase/word level over the entire document.\n"
            "[Rules]\n"
            "- Flag ONLY when the translation introduces a new sentence/clause/single-word or multi-word phrase not in the source.\n"
            "- Paraphrases and verbalized relations are acceptable, allowing for slight nuances, provided the core meaning remains unchanged.\n"
            "- Natural connectives/politeness/fluency are acceptable when they do not create new facts/conditions/quantities.\n"
            "- Reordering allowed ONLY within the same sentence; cross-sentence introduction counts as addition.\n"
            "- Ignore formatting/casing/spacing. Preserve ALL original \\n in Translation.\n"
            "- If the same translation sentence/clause/phrase is added multiple times, record ALL occurrences in `added_spans`.\n"
            "- Process sentence-by-sentence: split both texts by sentence delimiters (. ! ? and line breaks). For each source sentence S[i], additions must be checked ONLY within the corresponding translation sentence(s); cross-sentence introductions count as additions but must NOT be used to excuse omissions elsewhere.\n"
            "- Record ALL occurrences and order `added_spans` by the translation order of appearance.\n"
            "- Context-aware: Before flagging, assess the sentence-level meaning; DO flag (even for a single word) when it introduces new facts/conditions/quantities, named entities, dates/times, amounts, references, or obligations; do NOT flag purely stylistic fluency that does not change meaning.\n"
            "Return strictly in this format:\n"
            "{\n"
            "  \"faithfulness_issue\": true|false,\n"
            "  \"added_spans\": [\"exact translation sentence/phrase\" ...],\n"
            "  \"reasons\": [\"short reason\" ...],\n"
            "  \"suggestions\": [\"full revised translation string, preserving \\n\"]\n"
            "}\n"
            "- If false: arrays are [], no suggestions. If true: suggestions = full revised translation (no meta talk).\n"
        )
    }
    user_msg = {"role": "user", "content": _base_user_block(source, translated)}
    return system_msg, user_msg



def build_category_prompt(sentence: str):
    system_msg = {
        "role": "system",
        "content": (
            "You are a localization quality checker AI.\n"
            "Your task is to identify which formatting categories are present in a given translated sentence.\n"
            "The only valid categories are: currency, date, time.\n"
            "Only return the category if a clear formatting pattern appears in the sentence.\n"
            "Do NOT infer based on meaning or context.\n"
            "If no formatting is detected, return an empty list [].\n"
            "Return format: a JSON list of strings. Example: [\"currency\"] or [\"time\", \"date\"] or []."
        )
    }
    user_msg = {
        "role": "user",
        "content": f"Translated sentence: {sentence}\n\nWhich categories apply?"
    }
    return system_msg, user_msg

            # "- Preserve ALL escape characters (e.g., \\n, \", \\\\) exactly.\n"
def build_check_prompt(sentence: str, guideline: str, source_text: str):
    src_literal = json.dumps(source_text, ensure_ascii=False)
    trans_literal = json.dumps(sentence, ensure_ascii=False)
    
    system_msg = {
        "role": "system",
        "content": (
            "[GUIDELINE]\n"
            f"{guideline}\n"
            "[INSTRUCTIONS]\n"
            "You are a localization format validator AI.\n"
            "\n"
            "Task:\n"
            "- Check if the translated sentence follows the locale-specific guideline.\n"
            "- Revise ONLY the parts of the translation that violate the guideline.\n"
            "- If no violation, return the translation unchanged.\n"
            "\n"
            "Formatting rules:\n"
            "- Always consider grammar and context before applying guidelines.\n"
            "- Revised sentence must be identical to the original except for corrected formatting.\n"
            "- Do not modify any character outside the minimal required span.\n"
            "Important:\n"
            "- Do NOT remove or insert \", \', \\n or brackets in  exactly unless explicitly part of the violation span.\n"
            "\n"
            "Return strictly in this format:\n"
            "{\n"
            "  \"revised\": \"<final revised translation>\",\n"
            "  \"source_spans\": [\"<exact spans from source>\" ...],\n"
            "  \"trans_spans\": [\"<exact spans from original translation>\" ...],\n"
            "  \"revised_spans\": [\"<exact spans from revised translation>\" ...]\n"
            "}\n"
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Source sentence:\n{src_literal.strip()}\n"
            f"Translated sentence:\n{trans_literal.strip()}\n"
        )
    }
    return system_msg, user_msg

