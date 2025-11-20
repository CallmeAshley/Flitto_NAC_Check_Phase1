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
            "Follow these rules literally; do not infer beyond them.\n"
            "- Scope: Look ONLY at emojis, including ZWJ sequences, variation selectors, and skin-tone modifiers. Ignore all words, punctuation, spacing, and formatting that are not emojis.\n"
            "- Identity & Composition: Treat emojis with ZWJ/skin tone/variation selectors as a single unit; types must match exactly.\n"
            "- Position & Order: If any emoji is moved, reordered, or placed in a different part of the sentence than in the source, flag as an issue.\n"
            "- Presence: If any emoji is missing in translation or newly added (vs. source), flag as an issue.\n"
            "- Do not remove or modify any leading numbering/bullets/punctuation marks such as: a), 1., (1), •, -.\n"
            "Return strictly in JSON format:\n"
            "{\n"
            "  \"emoji_issue\": true|false,\n"
            "  \"suggestions\": [\"corrected translation string only\" ...]\n"
            "}\n"
            "- If emoji_issue is true: suggestions MUST contain the final corrected translation text(s) only (no meta words like 'Include' or explanations), preserving all original \\n in the translation where applicable.\n"
            "- If emoji_issue is false: suggestions MUST be [].\n"
            "[Important]\n"
            "- Return only the JSON object. No code fences, no prose, no extra keys.\n"
            "- Escape every ASCII double quote (\") inside string values as \\\".\n"
            
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
            "- If the same source sentence/clause/phrase is missing multiple times, record ALL occurrences in 'missing_spans' and 'revised_spans'.\n"
            "- Process sentence-by-sentence: split both texts by sentence delimiters (. ! ? and line breaks). For each source sentence S[i], search ONLY within the corresponding translation sentence(s); cross-sentence matches do NOT count.\n"
            "- Record ALL occurrences and order 'missing_spans' by the source order of appearance.\n"
            "- Context-aware: Before flagging, assess the sentence-level meaning; do NOT flag if the paraphrase preserves the core meaning, but DO flag (even for a single word) if it changes facts/conditions/quantities, named entities, dates/times, amounts, references, or obligations.\n"
            "Return strictly in JSON format:\n"
            "{\n"
            "  \"missing_content\": true|false,\n"
            "  \"missing_spans\": [\"exact source sentence/phrase\" ...],\n"
            "  \"revised_spans\": [\"exact fragment inserted\" ...],\n"
            "  \"suggestions\": [\"full revised translation string, preserving \\n\"]\n"
            "}\n"
            "- If false: arrays are [], no suggestions. If true: suggestions = full revised translation (no meta talk).\n"
            "[Important]\n"
            "- The suggestions field must restore all missing_spans using the corresponding revised_spans and present the fully corrected translation.\n"
            "- NEVER modify line breaks, quotation marks, or any original punctuation in suggestions — even one change will be treated as a critical error.\n"
            "- Escape every ASCII double quote (\") inside string values as \\\".\n"
        )
    }
    user_msg = {"role": "user", "content": _base_user_block(source, translated)}
    return system_msg, user_msg

# def build_addition_check_prompt(source: str, translated: str):
#     system_msg = {
#         "role": "system",
#         "content": (
#             "You are a localization QA AI. Detect additions at the sentence/phrase/word level over the entire document.\n"
#             "[Rules]\n"
#             "- Flag ONLY when the translation introduces a new sentence/clause/single-word or multi-word phrase not in the source.\n"
#             "- Paraphrases and verbalized relations are acceptable, allowing for slight nuances, provided the core meaning remains unchanged.\n"
#             "- Natural connectives/politeness/fluency are acceptable when they do not create new facts/conditions/quantities.\n"
#             "- Reordering allowed ONLY within the same sentence; cross-sentence introduction counts as addition.\n"
#             "- Ignore formatting/casing/spacing. Preserve ALL original \\n in Translation.\n"
#             "- If the same translation sentence/clause/phrase is added multiple times, record ALL occurrences in 'added_spans'.\n"
#             "- Process sentence-by-sentence: split both texts by sentence delimiters (. ! ? and line breaks). For each source sentence S[i], additions must be checked ONLY within the corresponding translation sentence(s); cross-sentence introductions count as additions but must NOT be used to excuse omissions elsewhere.\n"
#             "- Record ALL occurrences and order 'added_spans' by the translation order of appearance.\n"
#             "- Context-aware: Before flagging, assess the sentence-level meaning; DO flag (even for a single word) when it introduces new facts/conditions/quantities, named entities, dates/times, amounts, references, or obligations; do NOT flag purely stylistic fluency that does not change meaning.\n"
#             "Return strictly in JSON format:\n"
#             "{\n"
#             "  \"faithfulness_issue\": true|false,\n"
#             "  \"added_spans\": [\"exact translation sentence/phrase\" ...],\n"
#             "  \"suggestions\": [\"full revised translation string, preserving \\n\"]\n"
#             "}\n"
#             "- If false: arrays are [], no suggestions. If true: suggestions = full revised translation (no meta talk).\n"
#             "[Important]\n"
#             "- The suggestions field must remove every added_spans fragment and present the revised translation.\n"
#             "- NEVER modify line breaks, quotation marks, or any original punctuation in suggestions — even one change will be treated as a critical error.\n"
#             "- Escape every ASCII double quote (\") inside string values as \\\".\n"
#         )
#     }
#     user_msg = {"role": "user", "content": _base_user_block(source, translated)}
#     return system_msg, user_msg

def build_addition_check_prompt(source: str, translated: str):
    system_msg = {
        "role": "system",
        "content": (
            "You are a localization QA AI. Detect and handle additions at the sentence/phrase/word level over the entire document.\n"
            "[Rules]\n"
            "- Flag ONLY when the translation introduces a new sentence/clause/single-word or multi-word phrase not in the source.\n"
            "- Paraphrases and verbalized relations are acceptable if the core meaning remains unchanged.\n"
            "- Natural connectives/politeness/fluency are acceptable when they do not create new facts/conditions/quantities.\n"
            "- Reordering allowed ONLY within the same sentence; cross-sentence introduction counts as addition.\n"
            "- Ignore formatting/casing/spacing. Preserve ALL original \\n in Translation.\n"
            "- If the same translation sentence/clause/phrase is added multiple times, record ALL occurrences in 'added_spans'.\n"
            "- Process sentence-by-sentence: split texts by delimiters (. ! ? 。 ！ ？ ； … and line breaks). For each source sentence S[i], check only its corresponding translation sentence(s). Cross-sentence introductions still count as additions.\n"
            "- Record ALL occurrences and order 'added_spans' by their appearance in the translation.\n"
            "- Context-aware: Before flagging, assess the sentence-level meaning. DO flag (even for a single word) when it introduces new facts, conditions, quantities, named entities, dates/times, amounts, references, or obligations.\n"
            "- DO NOT flag stylistic or fluency-related insertions that do not change meaning.\n\n"

            "[Reconstruction Guideline]\n"
            "- After removing each added_span, rewrite the surrounding part to be grammatically and semantically natural.\n"
            "- When deletion leaves an incomplete or fragmented sentence, analyze the neighboring sentences (previous and next) and complete it naturally to maintain fluency.\n"
            "- Do not introduce any new factual or contextual content beyond what exists in the source.\n"
            "- Preserve all original line breaks (\\n), quotation marks, and punctuation exactly as they appear.\n"
            "- The goal is to produce a fully natural, coherent, and faithful translation after removing additions.\n\n"

            "Return strictly in JSON format:\n"
            "{\n"
            "  \"faithfulness_issue\": true|false,\n"
            "  \"added_spans\": [\"exact translation sentence/phrase\" ...],\n"
            "  \"suggestions\": [\"full revised translation string, preserving \\n\"]\n"
            "}\n"
            "- If false: arrays are [], no suggestions.\n"
            "- If true: suggestions = full revised translation (no meta talk).\n"
            "- Escape every ASCII double quote (\") inside string values as \\\".\n"
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
# def build_check_prompt(sentence: str, guideline: str, source_text: str):
#     src_literal = json.dumps(source_text, ensure_ascii=False)
#     trans_literal = json.dumps(sentence, ensure_ascii=False)
    
#     system_msg = {
#         "role": "system",
#         "content": (
#             "[GUIDELINE]\n"
#             f"{guideline}\n"
#             "[INSTRUCTIONS]\n"
#             "You are a localization format validator AI.\n"
#             "\n"
#             "Task:\n"
#             "- Check if the translated sentence follows the locale-specific guideline.\n"
#             "- Revise ONLY the parts of the translation that violate the guideline.\n"
#             "- If no violation, return the translation unchanged.\n"
#             "\n"
#             "Formatting rules:\n"
#             "- Always consider grammar and context before applying guidelines.\n"
#             "- Revised sentence must be identical to the original except for corrected formatting.\n"
#             "- Do not modify any character outside the minimal required span.\n"
#             "Important:\n"
#             "- Do NOT remove or insert \", \', \\n or brackets in exactly unless explicitly part of the violation span.\n"
#             "- Escape every ASCII double quote (\") inside string values as \\\".\n"
#             "\n"
#             "Return strictly in this format:\n"
#             "{\n"
#             "  \"revised\": \"<final revised translation>\",\n"
#             "  \"source_spans\": [\"<exact spans from source>\" ...],\n"
#             "  \"trans_spans\": [\"<exact spans from original translation>\" ...],\n"
#             "  \"revised_spans\": [\"<exact spans from revised translation>\" ...]\n"
#             "}\n"
#         )
#     }
#     user_msg = {
#         "role": "user",
#         "content": (
#             f"Source sentence:\n{src_literal.strip()}\n"
#             f"Translated sentence:\n{trans_literal.strip()}\n"
#         )
#     }
#     return system_msg, user_msg

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
            "- Focus purely on locale formatting (currency, date, time).\n"
            "- Revised sentence must be identical to the original except for corrected formatting.\n"
            "- Do not modify any character outside the minimal required span.\n"
            "- Do NOT remove or insert \", \', \\n, or brackets unless explicitly part of the violation span.\n"
            "- Escape every ASCII double quote (\") inside string values as \\\".\n"
            "\n"
            "[Span Consistency & Minimal Edit]\n"
            "- If there is no violation, 'revised' must be exactly identical to the original translation, "
            "and all span arrays must be empty ([]).\n"
            "- If there is a violation:\n"
            "  1) Each element in 'trans_spans' corresponds exactly to a replaced segment in the original translation.\n"
            "  2) Each element in 'revised_spans' corresponds to the corrected segment.\n"
            "  3) 'revised' MUST equal the original translation after replacing every 'trans_spans[i]' "
            "with its corresponding 'revised_spans[i]' — no other characters may be changed.\n"
            "  4) If this consistency rule would not hold, reconstruct your answer so that it does.\n"
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


