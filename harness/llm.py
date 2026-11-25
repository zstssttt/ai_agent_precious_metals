"""Provider adapter — the only module in the harness that talks to a model.

Everything above this file speaks one interface (`LLMClient.complete`), so
swapping DeepSeek for OpenAI or Anthropic is an environment variable, not a
code change. That matters for two reasons beyond tidiness:

* the loop can be re-run on a second provider for a few dollars, which turns
  "which model did you use" from a caveat in the write-up into a measured
  ablation;
* if the primary provider is unreachable mid-project, nothing downstream has to
  be rewritten.

C1: no other harness module may import an SDK. `builder`, `validator` and the
gate half of `analyst` must stay deterministic.

Provider facts verified against api-docs.deepseek.com on 2026-11-23; see
`PROVIDERS` below for the exact strings and what each one is for.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from harness import config

logger = logging.getLogger(__name__)


def retryable_errors() -> tuple[type[BaseException], ...]:
    """Exception types for which re-sending the identical request may succeed.

    Everything NOT in this tuple propagates out of `LLMClient.complete`
    immediately, with its traceback intact.

    Measured cost of getting this wrong (2026-11-23 smoke run): a local
    `TypeError` raised while decoding a successful 200 was retried four times
    — four paid calls, 15s of backoff, and the traceback buried under three
    `retrying in ...` warnings. The bug was in our process the whole time.

    Returns `()` while unimplemented, i.e. nothing is retried. That fails fast,
    which is the safe direction to be wrong in.
    """
    from openai import (      # noqa: PLC0415 - optional dependency, see __init__
        APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError,
        BadRequestError, InternalServerError, NotFoundError, PermissionDeniedError,
        RateLimitError, UnprocessableEntityError,
    )

    # Retry only transient server/network state — a resend of the identical
    # request could plausibly succeed. Deterministic 4xx (auth/bad-request/
    # not-found/permission/422-schema) are NOT here: they fail the same way
    # every time, and 422 in particular must reach loop.py as feedback rather
    # than be silently retried (see complete() docstring and C7).
    #
    # APIStatusError is deliberately excluded: it is the parent of every 4xx/5xx
    # subclass, so adding it would isinstance-match the deterministic failures
    # above and drag them back into the retry path.
    #
    # APITimeoutError subclasses APIConnectionError; listed explicitly for intent.
    return (
        RateLimitError,
        InternalServerError,
        APIConnectionError,
        APITimeoutError,
    )


class LLMError(RuntimeError):
    """Raised when the provider cannot be reached or returns something unusable."""


class MissingAPIKey(LLMError):
    """Raised when no credential is configured — carries the fix in its message."""


# ======================================================================
# Providers
# ======================================================================


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    beta_base_url: str          # strict-mode endpoint; "" means same as base_url
    key_env: str
    default_model: str
    # USD per million tokens. DeepSeek publishes off-peak and peak rates; we
    # record off-peak since the loop is usually run outside 01:00-04:00 and
    # 06:00-10:00 UTC Mon-Fri. Used only for the cost line in the run report.
    price_in: float
    price_cached_in: float
    price_out: float
    supports_thinking_param: bool


PROVIDERS: dict[str, Provider] = {
    # https://api-docs.deepseek.com/  +  /guides/tool_calls/  +  /quick_start/pricing/
    "deepseek": Provider(
        name="deepseek",
        base_url="https://api.deepseek.com",
        beta_base_url="https://api.deepseek.com/beta",
        key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-pro",
        price_in=0.66, price_cached_in=0.022, price_out=1.98,
        supports_thinking_param=True,
    ),
    "openai": Provider(
        name="openai",
        base_url="https://api.openai.com/v1",
        beta_base_url="",
        key_env="OPENAI_API_KEY",
        default_model="gpt-5.6-sol",
        price_in=0.0, price_cached_in=0.0, price_out=0.0,   # fill in before relying on the cost line
        supports_thinking_param=False,
    ),
}

# DeepSeek strict mode rejects several JSON Schema keywords. Documented as
# unsupported: minLength/maxLength on strings, minItems/maxItems on arrays.
# `tools.py` must not emit them; `test_phase4.py` enforces that.
UNSUPPORTED_SCHEMA_KEYS = ("minLength", "maxLength", "minItems", "maxItems")


# ======================================================================
# Call records
# ======================================================================


@dataclass
class Usage:
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.cached_tokens += other.cached_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.calls += other.calls

    def cost_usd(self, provider: Provider) -> float:
        fresh = max(self.prompt_tokens - self.cached_tokens, 0)
        return (
            fresh / 1e6 * provider.price_in
            + self.cached_tokens / 1e6 * provider.price_cached_in
            + self.completion_tokens / 1e6 * provider.price_out
        )

    def as_dict(self) -> dict:
        return {
            "calls": self.calls, "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class LLMResponse:
    """One model turn, normalised across providers."""

    text: str
    tool_calls: tuple[dict, ...]        # [{"id", "name", "arguments": dict}]
    finish_reason: str
    usage: Usage
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def wants_tool(self) -> bool:
        return bool(self.tool_calls)


# ======================================================================
# Client
# ======================================================================


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, provider: str | None = None, model: str | None = None,
                 *, strict_tools: bool | None = None,
                 effort: str | None = None, thinking: str | None = None,
                 trace_path: Path | None = None, api_key: str | None = None):
        name = (provider or config.LLM_PROVIDER).lower()
        if name not in PROVIDERS:
            raise LLMError(
                f'unknown provider "{name}". Known: {sorted(PROVIDERS)}. '
                f"Set HARNESS_LLM_PROVIDER, or add an entry to harness/llm.py:PROVIDERS."
            )
        self.provider = PROVIDERS[name]
        self.model = model or config.LLM_MODEL or self.provider.default_model
        self.strict_tools = (
            config.LLM_STRICT_TOOLS if strict_tools is None else strict_tools
        )
        self.effort = effort or config.LLM_REASONING_EFFORT
        self.thinking = thinking or config.LLM_THINKING
        self.trace_path = trace_path if trace_path is not None else config.TRACE_PATH
        self.total = Usage()

        key_env = config.LLM_API_KEY_ENV or self.provider.key_env
        self.api_key = api_key or os.environ.get(key_env, "")
        if not self.api_key:
            raise MissingAPIKey(
                f"no API key found.\n"
                f"  export {key_env}='your-key-here'\n"
                f"and re-run. (Provider: {self.provider.name}. To use a different "
                f"provider set HARNESS_LLM_PROVIDER.)"
            )

        base = config.LLM_BASE_URL or (
            self.provider.beta_base_url
            if (self.strict_tools and self.provider.beta_base_url)
            else self.provider.base_url
        )
        self.base_url = base

        try:
            from openai import OpenAI       # noqa: PLC0415 - optional dependency
        except ImportError as exc:          # pragma: no cover
            raise LLMError(
                "the `openai` package is required (DeepSeek speaks the same "
                "protocol).\n  pip install openai"
            ) from exc

        self._client = OpenAI(
            api_key=self.api_key, base_url=self.base_url,
            timeout=config.LLM_TIMEOUT_SECONDS, max_retries=0,   # we retry ourselves
            # Measured 2026-11-23: DeepSeek serves `Content-Encoding: br`, and
            # httpx2's BrotliDecoder is broken against the `brotli` package that
            # ships with anaconda — `_decoders.py:183` binds `.process` (a C
            # function taking positional args only) and `:192` calls it with
            # `output_buffer_limit=`, raising `TypeError: process() takes no
            # keyword arguments` *after* a 200. Patching site-packages would not
            # survive the next `pip install`, so we opt out of brotli instead.
            # gzip costs nothing at this payload size. Remove once httpx2 fixes
            # it or `brotlicffi` is the installed variant (it has `.decompress`,
            # which takes the keyword and is therefore unaffected).
            default_headers={"Accept-Encoding": "gzip"},
        )
        logger.info("LLM %s/%s via %s (strict_tools=%s)",
                    self.provider.name, self.model, self.base_url, self.strict_tools)

    # ------------------------------------------------------------------

    def _extra_body(self, effort: str | None) -> dict:
        if not self.provider.supports_thinking_param:
            return {}
        body: dict[str, Any] = {"thinking": {"type": self.thinking}}
        return body

    def _trace(self, record: dict) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def complete(self, messages: Sequence[Mapping[str, Any]],
                 tools: Sequence[Mapping[str, Any]] | None = None,
                 *, effort: str | None = None, label: str = "") -> LLMResponse:
        """One chat turn. Retries transport failures; never retries a bad answer.

        A 429 or a 502 is worth retrying — the request was fine. A schema
        violation is not: the request itself was wrong, and the loop's own
        feedback path (`loop.py`) is what repairs it, because that path also
        teaches the model what it got wrong. Retrying silently here would hide
        the one signal we want recorded in the journal.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": config.LLM_MAX_OUTPUT_TOKENS,
        }
        eff = effort or self.effort
        if eff:
            kwargs["reasoning_effort"] = eff
        extra = self._extra_body(eff)
        if extra:
            kwargs["extra_body"] = extra
        if tools:
            kwargs["tools"] = list(tools)
            # DeepSeek: forced tool_choice is accepted, but `auto` plus a system
            # instruction naming the tool keeps the same code working on
            # providers that reject forcing (Claude Opus 5.5 returns 400).
            kwargs["tool_choice"] = "auto"

        last: Exception | None = None
        for attempt in range(1, config.LLM_MAX_RETRIES + 1):
            started = time.time()
            try:
                raw = self._client.chat.completions.create(**kwargs)
            except retryable_errors() as exc:
                last = exc
                if attempt == config.LLM_MAX_RETRIES:
                    break
                backoff = min(2 ** attempt, 30) + random.random()
                logger.warning("LLM call failed (%s/%s): %s — retrying in %.1fs",
                               attempt, config.LLM_MAX_RETRIES, exc, backoff)
                time.sleep(backoff)
                continue

            response = self._parse(raw)
            if response.finish_reason == "length":
                # finish_reason="length" is NOT an exception — the call was a
                # paid HTTP 200. With a reasoning model, thinking and answer
                # share max_tokens, so a long think can hit the ceiling before
                # a single answer/tool-call token lands: full cost, no usable
                # output, and the loop silently re-asks. Surface it here so the
                # leak can never hide in finish_reason again (see config note on
                # LLM_MAX_OUTPUT_TOKENS).
                logger.warning(
                    "%s hit max_tokens=%s (finish=length): output truncated, "
                    "reasoning_tokens=%s, completion_tokens=%s. This turn may "
                    "carry no answer — raise LLM_MAX_OUTPUT_TOKENS.",
                    label or "call", config.LLM_MAX_OUTPUT_TOKENS,
                    response.usage.reasoning_tokens, response.usage.completion_tokens,
                )
            self.total.add(response.usage)
            self._trace({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "label": label, "model": self.model, "attempt": attempt,
                "elapsed_s": round(time.time() - started, 2),
                "request": {"messages": list(messages),
                            "tools": [t.get("function", {}).get("name") for t in (tools or [])],
                            "reasoning_effort": eff},
                "response": {"text": response.text,
                             "tool_calls": list(response.tool_calls),
                             "finish_reason": response.finish_reason},
                "usage": response.usage.as_dict(),
            })
            return response

        raise LLMError(
            f"{self.provider.name} unreachable after {config.LLM_MAX_RETRIES} "
            f"attempts: {last}"
        ) from last

    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: Any) -> LLMResponse:
        choice = raw.choices[0]
        message = choice.message

        calls: list[dict] = []
        for call in (getattr(message, "tool_calls", None) or []):
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"tool call `{call.function.name}` returned unparseable JSON: "
                    f"{exc}. Arguments were: {call.function.arguments!r}"
                ) from exc
            calls.append({"id": call.id, "name": call.function.name, "arguments": args})

        u = getattr(raw, "usage", None)
        details = getattr(u, "prompt_tokens_details", None) if u else None
        comp_details = getattr(u, "completion_tokens_details", None) if u else None
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            cached_tokens=getattr(details, "cached_tokens", 0) or 0 if details else 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            reasoning_tokens=(
                getattr(comp_details, "reasoning_tokens", 0) or 0 if comp_details else 0
            ),
            calls=1,
        )

        return LLMResponse(
            text=(message.content or ""),
            tool_calls=tuple(calls),
            finish_reason=choice.finish_reason or "",
            usage=usage,
            raw={},
        )

    # ------------------------------------------------------------------

    def cost_so_far(self) -> float:
        return self.total.cost_usd(self.provider)

    def report(self) -> str:
        u = self.total
        return (
            f"{self.provider.name}/{self.model}: {u.calls} calls, "
            f"{u.prompt_tokens:,} in ({u.cached_tokens:,} cached), "
            f"{u.completion_tokens:,} out ({u.reasoning_tokens:,} reasoning) "
            f"≈ ${self.cost_so_far():.3f}"
        )


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "MissingAPIKey",
    "PROVIDERS",
    "Provider",
    "UNSUPPORTED_SCHEMA_KEYS",
    "Usage",
]
