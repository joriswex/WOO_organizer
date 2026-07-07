"""
llm_providers.py — shared dispatch layer for calling OpenAI or Gemini.

Used by pipeline_vlm.py, pipeline_inventarislijst.py, and server.py's
/api/llm/complete endpoint (dossier summary, entity bios, timeline
chronology). Those last three are client-side (browser) features — they
can't call Gemini directly the way they call OpenAI today, because Gemini's
API blocks direct cross-origin browser requests (CORS), unlike OpenAI's.
Routing them through this same server-side dispatch fixes that.
"""
from __future__ import annotations

import base64


def make_client(provider: str, api_key: str):
    """Construct the right SDK client for the given provider."""
    if provider == "gemini":
        from google import genai
        return genai.Client(api_key=api_key)
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def call_llm(
    provider: str,
    client,
    model: str,
    system_msg: str,
    user_text: str,
    image_b64: str | None = None,
    max_tokens: int = 8192,
    timeout: float | None = None,
    json_mode: bool = True,
) -> tuple[str, bool]:
    """One (non-retried) call to the given provider.

    Returns (raw_text, was_truncated) — was_truncated is True when the model
    hit its output-token limit mid-response, so the caller can skip retrying
    with the same input (it would just truncate again).

    json_mode=True requests strict JSON-object output (used by every
    pipeline call site). Set json_mode=False for free-form text output
    (e.g. the dossier summary, which is prose, not JSON).
    """
    if provider == "gemini":
        from google.genai import types
        parts: list = []
        if image_b64:
            parts.append(types.Part.from_bytes(data=base64.b64decode(image_b64), mime_type="image/jpeg"))
        parts.append(user_text)
        config_kwargs: dict = dict(
            system_instruction=system_msg,
            temperature=0,
            max_output_tokens=max_tokens,
        )
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        response = client.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        raw = response.text or ("{}" if json_mode else "")
        truncated = bool(response.candidates) and response.candidates[0].finish_reason == "MAX_TOKENS"
        return raw, truncated

    # openai (default)
    messages: list = [{"role": "system", "content": system_msg}]
    if image_b64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}},
                {"type": "text", "text": user_text},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_text})

    kwargs = dict(model=model, max_tokens=max_tokens, temperature=0, messages=messages)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if timeout is not None:
        kwargs["timeout"] = timeout
    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    raw = choice.message.content or ("{}" if json_mode else "")
    return raw, choice.finish_reason == "length"
