from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from typing import Any, cast

from .auth import ChatGPTOAuthProtocolError
from .messages import AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage
from .protocol import reasoning_parts_from_response_items
from .reasoning_signature import (
    decode_reasoning_signature,
    encode_reasoning_signature,
    extract_reasoning_item_for_model,
    is_recognized_reasoning_signature,
)
from .strict_json import as_js_safe_integer, strict_json_loads

# ---------------------------------------------------------------------------
# Request conversion: Anthropic → internal
# ---------------------------------------------------------------------------


def anthropic_request_to_internal(
    *,
    model: object,
    messages: object,
    system: str | list[dict[str, Any]] | None = None,
    tools: object = None,
    tool_choice: dict[str, Any] | None = None,
    stop_sequences: list[str] | None = None,
    thinking: dict[str, Any] | None = None,
    max_tokens: int | None = None,
    output_format: dict[str, Any] | None = None,
    output_config: object = None,
    metadata: object = None,
    backend_model: str | None = None,
) -> tuple[
    list[Message], list[ToolSchema] | None, str | dict | None, list[str] | None, str | None, dict[str, Any] | None
]:
    """Convert Anthropic Messages request fields to internal types.

    Returns (messages, tools, tool_choice, stop, reasoning_effort, text).
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if backend_model is not None and (
        not isinstance(backend_model, str) or not backend_model or backend_model != backend_model.strip()
    ):
        raise ValueError("backend_model must be a non-empty string when provided")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty array")
    validate_anthropic_metadata(metadata)
    if stop_sequences is not None and (
        not isinstance(stop_sequences, list) or any(not isinstance(value, str) or not value for value in stop_sequences)
    ):
        raise ValueError("stop_sequences must be an array of non-empty strings")
    normalized_stop_sequences = stop_sequences if stop_sequences else None
    internal_messages: list[Message] = []

    # System prompt → SYSTEM message
    if system is not None:
        sys_text = _extract_system_text(system)
        if sys_text:
            internal_messages.append(Message(role=MessageRole.SYSTEM, content=sys_text))

    # Convert messages
    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"messages[{index}] must be an object")
        _reject_unknown_fields(msg, {"role", "content"}, f"messages[{index}]")
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            _convert_user_message(content, internal_messages)
        elif role == "assistant":
            _convert_assistant_message(content, internal_messages, backend_model=backend_model)
        else:
            raise ValueError(f"messages[{index}].role must be user or assistant")

    # Convert tools
    internal_tools = _convert_tools(tools) if tools is not None else None

    # Convert tool_choice
    internal_tool_choice = _convert_tool_choice(tool_choice)

    reasoning_effort = _convert_reasoning_effort(thinking, output_config, max_tokens)
    text = _convert_output_format(output_format, output_config)

    return internal_messages, internal_tools, internal_tool_choice, normalized_stop_sequences, reasoning_effort, text


def _extract_system_text(system: str | list[dict[str, Any]]) -> str:
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        raise ValueError("system must be a string or array")
    parts: list[str] = []
    for index, block in enumerate(system):
        if not isinstance(block, dict) or block.get("type") != "text":
            raise ValueError(f"system[{index}] must be a text block")
        _validate_nullable_unrepresentable_field(block, "citations", f"system[{index}]")
        _reject_unknown_fields(block, {"type", "text", "citations"}, f"system[{index}]")
        text = block.get("text")
        if not isinstance(text, str):
            raise ValueError(f"system[{index}].text must be a string")
        parts.append(text)
    return "\n\n".join(parts)


def _convert_user_message(content: object, out: list[Message]) -> None:
    if isinstance(content, str):
        out.append(Message(role=MessageRole.USER, content=content))
        return
    if not isinstance(content, list):
        raise ValueError("Anthropic user message content must be a string or array")
    if not content:
        raise ValueError("Anthropic user message content blocks must be non-empty")
    text_parts: list[str] = []
    image_urls: list[str] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            raise ValueError(f"Anthropic user content block {index} must be an object")
        block_type = block.get("type")
        if block_type == "text":
            _validate_nullable_unrepresentable_field(
                block,
                "citations",
                f"Anthropic user content block {index}",
            )
            _reject_unknown_fields(
                block,
                {"type", "text", "citations"},
                f"Anthropic user content block {index}",
            )
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Anthropic user text block {index} requires string text")
            text_parts.append(text)
        elif block_type == "tool_result":
            _reject_unknown_fields(
                block,
                {"type", "tool_use_id", "content", "is_error"},
                f"Anthropic tool_result block {index}",
            )
            if text_parts or image_urls:
                out.append(
                    Message(
                        role=MessageRole.USER,
                        content="".join(text_parts),
                        images=tuple(image_urls),
                    )
                )
                text_parts = []
                image_urls = []
            tool_use_id = block.get("tool_use_id")
            if not isinstance(tool_use_id, str):
                raise ValueError(f"Anthropic tool_result block {index} requires string tool_use_id")
            result_content = block.get("content", "")
            tool_result_images: list[str] = []
            if isinstance(result_content, list):
                text_pieces: list[str] = []
                for part_index, p in enumerate(result_content):
                    if not isinstance(p, dict):
                        raise ValueError(f"Anthropic tool_result block {index} content {part_index} must be an object")
                    if p.get("type") == "text":
                        _validate_nullable_unrepresentable_field(
                            p,
                            "citations",
                            f"Anthropic tool_result block {index} content {part_index}",
                        )
                        _reject_unknown_fields(
                            p,
                            {"type", "text", "citations"},
                            f"Anthropic tool_result block {index} content {part_index}",
                        )
                        text = p.get("text")
                        if not isinstance(text, str):
                            raise ValueError(f"Anthropic tool_result block {index} text content requires string text")
                        text_pieces.append(text)
                    elif p.get("type") == "image":
                        _reject_unknown_fields(
                            p,
                            {"type", "source"},
                            f"Anthropic tool_result block {index} content {part_index}",
                        )
                        source = p.get("source")
                        image_url = _anthropic_image_url(source)
                        tool_result_images.append(image_url)
                    else:
                        text_pieces.append(_render_anthropic_content_block(p))
                result_content = "".join(text_pieces)
            elif not isinstance(result_content, str):
                raise ValueError(f"Anthropic tool_result block {index} content must be a string or array")
            if "is_error" in block and not isinstance(block["is_error"], bool):
                raise ValueError(f"Anthropic tool_result block {index} is_error must be a boolean")
            if block.get("is_error") is True:
                result_content = f"[tool_error]\n{result_content}"
            out.append(
                Message(
                    role=MessageRole.TOOL,
                    content=result_content,
                    tool_call_id=tool_use_id,
                )
            )
            if tool_result_images:
                out.append(
                    Message(
                        role=MessageRole.USER,
                        content="",
                        images=tuple(tool_result_images),
                    )
                )
        elif block_type == "image":
            _reject_unknown_fields(
                block,
                {"type", "source"},
                f"Anthropic user content block {index}",
            )
            source = block.get("source")
            image_url = _anthropic_image_url(source)
            image_urls.append(image_url)
        else:
            text_parts.append(_render_anthropic_content_block(block))
    if text_parts or image_urls:
        out.append(
            Message(
                role=MessageRole.USER,
                content="".join(text_parts),
                images=tuple(image_urls),
            )
        )


def _convert_assistant_message(
    content: object,
    out: list[Message],
    *,
    backend_model: str | None = None,
) -> None:
    if isinstance(content, str):
        out.append(Message(role=MessageRole.ASSISTANT, content=content))
        return
    if not isinstance(content, list):
        raise ValueError("Anthropic assistant message content must be a string or array")
    if not content:
        out.append(Message(role=MessageRole.ASSISTANT, content=""))
        return
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_call_ids: set[str] = set()
    reasoning_items: list[dict[str, Any]] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            raise ValueError(f"Anthropic assistant content block {index} must be an object")
        block_type = block.get("type")
        if block_type == "text":
            _validate_nullable_unrepresentable_field(
                block,
                "citations",
                f"Anthropic assistant content block {index}",
            )
            _reject_unknown_fields(
                block,
                {"type", "text", "citations"},
                f"Anthropic assistant content block {index}",
            )
            text = block.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Anthropic assistant text block {index} requires string text")
            text_parts.append(text)
        elif block_type == "tool_use":
            _reject_unknown_fields(
                block,
                {"type", "id", "name", "input", "caller"},
                f"Anthropic tool_use block {index}",
            )
            _validate_direct_tool_caller(block, f"Anthropic tool_use block {index}")
            call_id = block.get("id")
            name = block.get("name")
            arguments = block.get("input")
            if not isinstance(call_id, str):
                raise ValueError(f"Anthropic tool_use block {index} requires string id")
            if not isinstance(name, str):
                raise ValueError(f"Anthropic tool_use block {index} requires string name")
            if not isinstance(arguments, dict):
                raise ValueError(f"Anthropic tool_use block {index} input must be an object")
            if call_id in tool_call_ids:
                raise ValueError(f"Anthropic assistant tool_use blocks contain duplicate id {call_id!r}")
            tool_call_ids.add(call_id)
            tool_calls.append(
                ToolCall(
                    id=call_id,
                    name=name,
                    arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                )
            )
        elif block_type == "thinking":
            _reject_unknown_fields(
                block,
                {"type", "thinking", "signature"},
                f"Anthropic assistant content block {index}",
            )
            thinking_text = block.get("thinking")
            if not isinstance(thinking_text, str):
                raise ValueError(f"Anthropic assistant thinking block {index} requires string thinking")
            signature = block.get("signature")
            if signature is not None and not isinstance(signature, str):
                raise ValueError(f"Anthropic assistant thinking block {index} signature must be a string")
            if is_recognized_reasoning_signature(signature):
                envelope = decode_reasoning_signature(cast(str, signature))
                if backend_model is None:
                    reasoning_items.append(envelope)
                else:
                    extracted = extract_reasoning_item_for_model(
                        envelope,
                        backend_model=backend_model,
                    )
                    if extracted is not None:
                        reasoning_items.append(extracted)
        elif block_type == "redacted_thinking":
            _reject_unknown_fields(
                block,
                {"type", "data"},
                f"Anthropic assistant content block {index}",
            )
            data = block.get("data")
            if not isinstance(data, str):
                raise ValueError(f"Anthropic assistant redacted_thinking block {index} requires string data")
        elif block_type == "server_tool_use":
            text_parts.append(_render_server_tool_use_block(block))
        elif block_type == "web_search_tool_result":
            text_parts.append(_render_web_search_tool_result_block(block))
        else:
            text_parts.append(_render_anthropic_content_block(block))
    out.append(
        Message(
            role=MessageRole.ASSISTANT,
            content="".join(text_parts),
            tool_calls=tuple(tool_calls) if tool_calls else (),
            reasoning_items=tuple(reasoning_items),
        )
    )


def _convert_tools(tools: object) -> list[ToolSchema]:
    if not isinstance(tools, list):
        raise ValueError("tools must be an array")
    result: list[ToolSchema] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"Anthropic tool {index} must be an object")
        if tool.get("type") == "programmatic_tool_calling":
            raise ValueError(
                "programmatic_tool_calling requires native Responses program/caller replay "
                "and is not supported by the Anthropic facade"
            )
        if "allowed_callers" in tool or "output_schema" in tool:
            raise ValueError(f"Anthropic tool {index} uses Programmatic Tool Calling fields that cannot be preserved")
        if "defer_loading" in tool:
            defer_loading = tool.get("defer_loading")
            if defer_loading is not None and not isinstance(defer_loading, bool):
                raise ValueError(f"Anthropic tool {index} defer_loading must be a boolean")
            if defer_loading is True:
                raise ValueError(
                    f"Anthropic tool {index} field defer_loading is not supported by the Codex OAuth backend"
                )
        for field in ("eager_input_streaming",):
            value = tool.get(field)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"Anthropic tool {index} {field} must be a boolean")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Anthropic tool {index} requires a non-empty name")
        web_search_tool = _convert_anthropic_web_search_tool(tool, index)
        if web_search_tool is not None:
            result.append(web_search_tool)
            continue
        tool_type = tool.get("type", "custom")
        if tool_type is not None and tool_type != "custom":
            raise ValueError(f"Anthropic tool {index} type must be 'custom' or null")
        _reject_unknown_fields(
            tool,
            {
                "type",
                "name",
                "description",
                "input_schema",
                "strict",
                "defer_loading",
                "eager_input_streaming",
                "allowed_callers",
                "output_schema",
            },
            f"Anthropic tool {index}",
        )
        result.append(
            ToolSchema(
                name=name,
                description=_optional_tool_description(tool, index),
                parameters=_required_tool_schema(tool, index),
                strict=_tool_strict(tool, index),
            )
        )
    return result


def _anthropic_image_url(source: object) -> str:
    if not isinstance(source, dict):
        raise ValueError("Anthropic image source must be an object")
    source_type = source.get("type")
    if source_type == "base64":
        _reject_unknown_fields(source, {"type", "media_type", "data"}, "Anthropic image source")
        media_type = source.get("media_type")
        data = source.get("data")
        if media_type in {"image/jpeg", "image/png", "image/gif", "image/webp"} and isinstance(data, str) and data:
            return f"data:{media_type};base64,{data}"
        raise ValueError(
            "Anthropic base64 image source media_type must be one of: "
            "image/jpeg, image/png, image/gif, image/webp, and data must be non-empty"
        )
    if source_type == "url":
        _reject_unknown_fields(source, {"type", "url"}, "Anthropic image source")
        url = source.get("url")
        if isinstance(url, str) and url:
            return url
        raise ValueError("Anthropic URL image source requires a non-empty url")
    raise ValueError("Anthropic image source type must be base64 or url")


def _optional_tool_description(tool: dict[str, Any], index: int) -> str | None:
    if "description" not in tool:
        return None
    description = tool["description"]
    if not isinstance(description, str):
        raise ValueError(f"Anthropic tool {index} description must be a string")
    return description


def _tool_strict(tool: dict[str, Any], index: int) -> bool:
    if "strict" not in tool:
        return False
    strict = tool["strict"]
    if not isinstance(strict, bool):
        raise ValueError(f"Anthropic tool {index} strict must be a boolean")
    return strict


def _required_tool_schema(tool: dict[str, Any], index: int) -> dict[str, Any]:
    schema = tool.get("input_schema")
    if not isinstance(schema, dict):
        raise ValueError(f"Anthropic tool {index} input_schema must be an object")
    return schema


def _convert_anthropic_web_search_tool(tool: dict[str, Any], index: int) -> ToolSchema | None:
    if (
        tool.get("name") != "web_search"
        or not isinstance(tool.get("type"), str)
        or tool["type"] not in {"web_search", "web_search_20250305", "web_search_20260209"}
    ):
        return None
    _reject_unknown_fields(
        tool,
        {
            "type",
            "name",
            "allowed_domains",
            "blocked_domains",
            "max_uses",
            "strict",
            "user_location",
            "defer_loading",
            "eager_input_streaming",
            "allowed_callers",
            "output_schema",
        },
        f"Anthropic tool {index}",
    )
    if "strict" in tool and tool["strict"] is not None and not isinstance(tool["strict"], bool):
        raise ValueError(f"Anthropic tool {index} strict must be a boolean or null")
    if tool.get("strict") is True:
        raise ValueError(
            f"Anthropic tool {index} web_search strict mode is not supported by the Codex OAuth transport"
        )
    if tool.get("blocked_domains") is not None:
        raise ValueError(
            f"Anthropic tool {index} web_search blocked_domains is not supported by the Codex OAuth transport"
        )
    max_uses = tool.get("max_uses")
    if max_uses is not None and (not isinstance(max_uses, int) or isinstance(max_uses, bool) or max_uses <= 0):
        raise ValueError(f"Anthropic tool {index} web_search max_uses must be a positive integer")
    openai_tool: dict[str, Any] = {"type": "web_search"}
    allowed_domains = tool.get("allowed_domains")
    if allowed_domains is not None:
        if (
            not isinstance(allowed_domains, list)
            or not allowed_domains
            or any(not isinstance(domain, str) or not domain.strip() for domain in allowed_domains)
        ):
            raise ValueError(
                f"Anthropic tool {index} web_search allowed_domains must be a non-empty array of strings"
            )
        openai_tool["filters"] = {"allowed_domains": list(allowed_domains)}
    user_location = tool.get("user_location")
    if user_location is not None:
        openai_tool["user_location"] = _normalize_web_search_user_location(
            user_location,
            f"Anthropic tool {index} web_search user_location",
        )
    return ToolSchema(
        name="web_search",
        description=None,
        parameters={
            "__codex_as_api_tool_type": "web_search",
            "openai_tool": openai_tool,
        },
    )


def _normalize_web_search_user_location(value: object, location: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    _reject_unknown_fields(
        value,
        {"type", "city", "country", "region", "timezone"},
        location,
    )
    if value.get("type") != "approximate":
        raise ValueError(f"{location}.type must be approximate")
    out: dict[str, str] = {"type": "approximate"}
    city = _optional_non_empty_string(value.get("city"), f"{location}.city")
    country = _optional_non_empty_string(value.get("country"), f"{location}.country")
    region = _optional_non_empty_string(value.get("region"), f"{location}.region")
    timezone = _optional_non_empty_string(value.get("timezone"), f"{location}.timezone")
    if city is not None:
        out["city"] = city
    if country is not None:
        if len(country) != 2 or not country.isascii() or not country.isalpha():
            raise ValueError(f"{location}.country must be a two-letter ISO country code")
        out["country"] = country
    if region is not None:
        out["region"] = region
    if timezone is not None:
        out["timezone"] = timezone
    return out


def _optional_non_empty_string(value: object, location: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a non-empty string")
    return value


def _convert_tool_choice(tc: dict[str, Any] | None) -> str | dict | None:
    if tc is None:
        return None
    if not isinstance(tc, dict):
        raise ValueError("tool_choice must be an object")
    tc_type = tc.get("type")
    if tc_type == "tool":
        allowed = {"type", "name", "disable_parallel_tool_use"}
    elif tc_type == "none":
        allowed = {"type"}
    else:
        allowed = {"type", "disable_parallel_tool_use"}
    _reject_unknown_fields(tc, allowed, "tool_choice")
    anthropic_parallel_tool_calls(tc)
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        name = tc.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_choice.name must be a non-empty string")
        if name == "web_search":
            return {"type": "web_search"}
        return {"type": "function", "name": name}
    if tc_type == "none":
        return "none"
    raise ValueError("tool_choice.type must be one of: auto, any, tool, none")


def anthropic_parallel_tool_calls(
    tool_choice: dict[str, Any] | None,
) -> bool | None:
    if tool_choice is None or "disable_parallel_tool_use" not in tool_choice:
        return None
    value = tool_choice["disable_parallel_tool_use"]
    if not isinstance(value, bool):
        raise ValueError("tool_choice.disable_parallel_tool_use must be a boolean")
    return not value


def validate_anthropic_metadata(metadata: object) -> None:
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    for key, value in metadata.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(f"metadata.{key} must be a string")


def resolve_anthropic_thinking_options(
    thinking: dict[str, Any] | None,
    max_tokens: int | None = None,
) -> tuple[str, str | None, bool]:
    """Return (thinking_display, reasoning_summary, thinking_disabled) for a thinking object."""
    if thinking is None:
        return "summarized", None, False
    _convert_thinking(thinking, max_tokens)
    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        return "omitted", "none", True
    display = "omitted" if thinking.get("display") == "omitted" else "summarized"
    reasoning_summary = "none" if display == "omitted" else "auto"
    return display, reasoning_summary, False


def _convert_thinking(thinking: dict[str, Any] | None, max_tokens: int | None) -> str | None:
    if thinking is None:
        return None
    if not isinstance(thinking, dict):
        raise ValueError("thinking must be an object")
    thinking_type = thinking.get("type")
    if thinking_type == "enabled":
        _reject_unknown_fields(thinking, {"type", "budget_tokens", "display"}, "thinking")
        display = thinking.get("display")
        if display is not None and (not isinstance(display, str) or display not in {"summarized", "omitted"}):
            raise ValueError("thinking.display must be 'summarized' or 'omitted' when provided")
        budget_tokens = thinking.get("budget_tokens")
        parsed_budget_tokens = as_js_safe_integer(budget_tokens)
        if parsed_budget_tokens is None or parsed_budget_tokens < 1024:
            raise ValueError("thinking.budget_tokens must be an integer greater than or equal to 1024")
        if max_tokens is not None and parsed_budget_tokens >= max_tokens:
            raise ValueError("thinking.budget_tokens must be less than max_tokens")
        return "high"
    if thinking_type == "adaptive":
        _reject_unknown_fields(thinking, {"type", "display"}, "thinking")
        display = thinking.get("display")
        if display is not None and (not isinstance(display, str) or display not in {"summarized", "omitted"}):
            raise ValueError("thinking.display must be 'summarized' or 'omitted' when provided")
        return "medium"
    if thinking_type == "disabled":
        _reject_unknown_fields(thinking, {"type"}, "thinking")
        return "none"
    raise ValueError("thinking.type must be one of: enabled, adaptive, disabled")


def _convert_reasoning_effort(
    thinking: dict[str, Any] | None,
    output_config: object,
    max_tokens: int | None,
) -> str | None:
    output_effort: str | None = None
    if output_config is not None:
        if not isinstance(output_config, dict):
            raise ValueError("output_config must be an object")
        unknown = sorted(set(output_config) - {"effort", "format", "task_budget"})
        if unknown:
            raise ValueError("output_config contains unsupported fields: " + ", ".join(unknown))
        task_budget = output_config.get("task_budget")
        if task_budget is not None:
            raise ValueError("output_config.task_budget is not supported by the Codex OAuth backend")
        raw_effort = output_config.get("effort")
        if raw_effort is not None:
            if not isinstance(raw_effort, str) or raw_effort not in {
                "low",
                "medium",
                "high",
                "xhigh",
                "max",
            }:
                raise ValueError("output_config.effort must be one of: low, medium, high, xhigh, max")
            output_effort = raw_effort

    thinking_effort = _convert_thinking(thinking, max_tokens)
    if thinking_effort == "none":
        return "none"
    return output_effort or thinking_effort


def _convert_output_format(
    output_format: object,
    output_config: object,
) -> dict[str, Any] | None:
    config_format: object = None
    if isinstance(output_config, dict):
        config_format = output_config.get("format")

    converted: list[tuple[object, dict[str, Any]]] = []
    for field, value in (("output_format", output_format), ("output_config.format", config_format)):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be an object")
        _validate_anthropic_output_format(value, field)
        text = anthropic_output_format_to_openai_text(value)
        if text is None:  # pragma: no cover - validation above guarantees conversion
            raise AssertionError("validated Anthropic output format did not convert")
        converted.append((value, text))

    if len(converted) == 2 and converted[0][0] != converted[1][0]:
        raise ValueError("output_format conflicts with output_config.format")
    return converted[0][1] if converted else None


def anthropic_output_format_to_openai_text(output_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if output_format is None:
        return None
    _validate_anthropic_output_format(output_format, "output_format")
    typ = output_format.get("type")
    if typ == "json_schema":
        schema = output_format.get("schema")
        name = output_format.get("name", "codex_output_schema")
        fmt: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": schema,
        }
        if isinstance(output_format.get("description"), str):
            fmt["description"] = output_format["description"]
        if isinstance(output_format.get("strict"), bool):
            fmt["strict"] = output_format["strict"]
        return {"format": fmt}
    if typ == "json_object":
        return {"format": {"type": "json_object"}}
    raise AssertionError("validated Anthropic output format has an unknown type")


def _validate_anthropic_output_format(output_format: dict[str, Any], field: str) -> None:
    typ = output_format.get("type")
    if not isinstance(typ, str):
        raise ValueError(f"{field}.type must be a string")
    if typ == "json_object":
        unknown = sorted(set(output_format) - {"type"})
        if unknown:
            raise ValueError(f"{field}.{unknown[0]} is not supported")
        return
    if typ != "json_schema":
        raise ValueError(f"{field}.type must be one of: json_object, json_schema")

    unknown = sorted(set(output_format) - {"type", "name", "description", "schema", "strict"})
    if unknown:
        raise ValueError(f"{field}.{unknown[0]} is not supported")
    if not isinstance(output_format.get("schema"), dict):
        raise ValueError(f"{field}.schema must be an object")
    name = output_format.get("name", "codex_output_schema")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError(
            f"{field}.name must contain only ASCII letters, digits, underscores, or hyphens "
            "and be at most 64 characters"
        )
    description = output_format.get("description")
    if "description" in output_format and not isinstance(description, str):
        raise ValueError(f"{field}.description must be a string when provided")
    strict = output_format.get("strict")
    if strict is not None and not isinstance(strict, bool):
        raise ValueError(f"{field}.strict must be a boolean when provided")


def _render_anthropic_content_block(block: dict[str, Any]) -> str:
    typ = block.get("type")
    if not isinstance(typ, str) or not typ:
        raise ValueError("Anthropic content block requires a non-empty type")
    if typ == "document":
        return _render_document_block(block)
    if typ == "search_result":
        return _render_search_result_block(block)
    raise ValueError(f"Anthropic content block type {typ!r} is not supported by this facade")


def _render_document_block(block: dict[str, Any]) -> str:
    _validate_nullable_unrepresentable_field(block, "citations", "Anthropic document block")
    _reject_unknown_fields(
        block,
        {"type", "source", "title", "context", "citations"},
        "Anthropic document block",
    )
    source = block.get("source")
    if not isinstance(source, dict):
        raise ValueError("Anthropic document block source must be an object")
    source_type = source.get("type")
    title = _optional_non_empty_string(block.get("title"), "Anthropic document block.title")
    context = _optional_non_empty_string(block.get("context"), "Anthropic document block.context")
    if source_type == "text":
        _reject_unknown_fields(
            source,
            {"type", "media_type", "data"},
            "Anthropic document block.source",
        )
        media_type = source.get("media_type")
        if media_type is not None and media_type != "text/plain":
            raise ValueError("Anthropic text document source media_type must be text/plain")
        data = source.get("data")
        if not isinstance(data, str):
            raise ValueError("Anthropic text document source requires string data")
        return _format_document_text(data, title, context)
    if source_type == "content":
        _reject_unknown_fields(
            source,
            {"type", "content"},
            "Anthropic document block.source",
        )
        raw_content = source.get("content")
        if isinstance(raw_content, str):
            return _format_document_text(raw_content, title, context)
        if not isinstance(raw_content, list) or not raw_content:
            raise ValueError(
                "Anthropic content document source requires a string or non-empty content array"
            )
        parts: list[str] = []
        for index, item in enumerate(raw_content):
            if not isinstance(item, dict):
                raise ValueError(f"Anthropic document content[{index}] must be an object")
            if item.get("type") != "text":
                raise ValueError(f"Anthropic document content[{index}] must be a text block")
            _validate_nullable_unrepresentable_field(
                item,
                "citations",
                f"Anthropic document content[{index}]",
            )
            _reject_unknown_fields(
                item,
                {"type", "text", "citations"},
                f"Anthropic document content[{index}]",
            )
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Anthropic document content[{index}] requires string text")
            parts.append(text)
        return _format_document_text("".join(parts), title, context)
    raise ValueError(f"Anthropic document source type {source_type!r} is not supported by this facade")


def _format_document_text(body: str, title: str | None, context: str | None) -> str:
    header_parts = ["[document"]
    if title is not None:
        header_parts.append(f" title={json.dumps(title, ensure_ascii=False)}")
    if context is not None:
        header_parts.append(f" context={json.dumps(context, ensure_ascii=False)}")
    header_parts.append("]")
    return f"{''.join(header_parts)}\n{body}\n[/document]"


def _render_search_result_block(block: dict[str, Any]) -> str:
    _validate_nullable_unrepresentable_field(block, "citations", "Anthropic search_result block")
    _reject_unknown_fields(
        block,
        {"type", "source", "url", "title", "content", "citations"},
        "Anthropic search_result block",
    )
    raw_source = block.get("source") if "source" in block else block.get("url")
    source = _optional_non_empty_string(raw_source, "Anthropic search_result block.source")
    title = _optional_non_empty_string(block.get("title"), "Anthropic search_result block.title")
    if source is None or title is None:
        raise ValueError("Anthropic search_result block requires non-empty source and title")
    content = block.get("content")
    if isinstance(content, str):
        body = content
    elif isinstance(content, list) and content:
        parts: list[str] = []
        for index, item in enumerate(content):
            if not isinstance(item, dict):
                raise ValueError(f"Anthropic search_result content[{index}] must be an object")
            if item.get("type") != "text":
                raise ValueError(f"Anthropic search_result content[{index}] must be a text block")
            _validate_nullable_unrepresentable_field(
                item,
                "citations",
                f"Anthropic search_result content[{index}]",
            )
            _reject_unknown_fields(
                item,
                {"type", "text", "citations"},
                f"Anthropic search_result content[{index}]",
            )
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Anthropic search_result content[{index}] requires string text")
            parts.append(text)
        body = "".join(parts)
    else:
        raise ValueError("Anthropic search_result block requires a string or non-empty content array")
    return (
        f"[search_result source={json.dumps(source, ensure_ascii=False)} "
        f"title={json.dumps(title, ensure_ascii=False)}]\n"
        f"{body}\n[/search_result]"
    )


def _render_server_tool_use_block(block: dict[str, Any]) -> str:
    _reject_unknown_fields(
        block,
        {"type", "id", "name", "input", "caller"},
        "Anthropic server_tool_use block",
    )
    _validate_direct_tool_caller(block, "Anthropic server_tool_use block")
    tool_id = _optional_non_empty_string(block.get("id"), "Anthropic server_tool_use block.id")
    name = _optional_non_empty_string(block.get("name"), "Anthropic server_tool_use block.name")
    if tool_id is None or name is None:
        raise ValueError("Anthropic server_tool_use block requires non-empty id and name")
    if name != "web_search":
        raise ValueError(f"Anthropic server_tool_use name {name!r} is not supported by this facade")
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        raise ValueError("Anthropic server_tool_use block.input must be an object")
    return (
        f"[server_tool_use id={json.dumps(tool_id, ensure_ascii=False)} "
        f"name={json.dumps(name, ensure_ascii=False)} "
        f"input={json.dumps(tool_input, ensure_ascii=False, separators=(',', ':'))}]"
    )


def _render_web_search_tool_result_block(block: dict[str, Any]) -> str:
    _reject_unknown_fields(
        block,
        {"type", "tool_use_id", "content", "caller"},
        "Anthropic web_search_tool_result block",
    )
    _validate_direct_tool_caller(block, "Anthropic web_search_tool_result block")
    tool_use_id = _optional_non_empty_string(
        block.get("tool_use_id"),
        "Anthropic web_search_tool_result block.tool_use_id",
    )
    if tool_use_id is None:
        raise ValueError("Anthropic web_search_tool_result block requires non-empty tool_use_id")
    content = block.get("content")
    if not isinstance(content, (list, dict)):
        raise ValueError(
            "Anthropic web_search_tool_result block requires a content array or error object"
        )
    return (
        f"[web_search_tool_result tool_use_id={json.dumps(tool_use_id, ensure_ascii=False)}]\n"
        f"{json.dumps(content, ensure_ascii=False, separators=(',', ':'))}\n[/web_search_tool_result]"
    )


def _validate_nullable_unrepresentable_field(
    value: dict[str, Any],
    field: str,
    location: str,
) -> None:
    if field not in value or value[field] is None:
        return
    if field == "citations":
        citations = value[field]
        if isinstance(citations, list) and not citations:
            return
        if (
            isinstance(citations, dict)
            and set(citations) == {"enabled"}
            and citations.get("enabled") is False
        ):
            return
    raise ValueError(f"{location}.{field} cannot be preserved by this facade")


def _validate_direct_tool_caller(block: dict[str, Any], location: str) -> None:
    if "caller" not in block:
        return
    caller = block["caller"]
    if not isinstance(caller, dict) or caller != {"type": "direct"}:
        raise ValueError(f"{location}.caller must be exactly {{'type': 'direct'}}")


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    location: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{location} contains unsupported fields: {', '.join(unknown)}")


def _parse_anthropic_tool_arguments(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise ChatGPTOAuthProtocolError("tool call arguments must be a JSON object string")
    try:
        parsed = strict_json_loads(value)
    except ValueError as exc:
        raise ChatGPTOAuthProtocolError("tool call arguments must contain valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ChatGPTOAuthProtocolError("tool call arguments JSON must be an object")
    return parsed


# ---------------------------------------------------------------------------
# Non-streaming response: internal → Anthropic
# ---------------------------------------------------------------------------


def _extract_response_reasoning_entry(
    entry: Mapping[str, Any],
    *,
    default_model: str | None,
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(entry, Mapping):
        raise ChatGPTOAuthProtocolError("response reasoning_items entry must be an object")
    raw_item = entry.get("item") if "item" in entry else entry
    if not isinstance(raw_item, dict) or raw_item.get("type") != "reasoning":
        raise ChatGPTOAuthProtocolError("response reasoning_items entry requires a reasoning item object")
    raw_model = entry.get("model")
    if raw_model is not None and (not isinstance(raw_model, str) or not raw_model or raw_model != raw_model.strip()):
        raise ChatGPTOAuthProtocolError("response reasoning_items model must be a non-empty string")
    return raw_item, raw_model or default_model


def internal_response_to_anthropic(
    response: AssistantResponse,
    model: str,
    request_id: str,
    *,
    thinking_display: str = "summarized",
    backend_model: str | None = None,
    thinking_disabled: bool = False,
) -> dict[str, Any]:
    if thinking_display not in {"summarized", "omitted"}:
        raise ValueError("thinking_display must be 'summarized' or 'omitted'")
    if backend_model is not None and (
        not isinstance(backend_model, str) or not backend_model or backend_model != backend_model.strip()
    ):
        raise ValueError("backend_model must be a non-empty string when provided")
    content: list[dict[str, Any]] = []

    raw_events = _raw_response_events(response.raw)

    visible_reasoning_parts: list[str] = []
    if not thinking_disabled:
        if response.reasoning_items:
            for entry in response.reasoning_items:
                raw_item, item_model = _extract_response_reasoning_entry(
                    entry,
                    default_model=backend_model or model,
                )
                try:
                    item_summary, item_raw = reasoning_parts_from_response_items([raw_item])
                except ValueError as exc:
                    raise ChatGPTOAuthProtocolError(str(exc)) from exc
                item_text = item_summary + item_raw
                if item_text:
                    visible_reasoning_parts.append(item_text)
                encrypted_content = raw_item.get("encrypted_content")
                if encrypted_content is not None and not isinstance(encrypted_content, str):
                    raise ChatGPTOAuthProtocolError("reasoning item encrypted_content must be a string or null")
                if isinstance(encrypted_content, str) and encrypted_content:
                    if not isinstance(item_model, str) or not item_model:
                        raise ChatGPTOAuthProtocolError("encrypted reasoning item requires a backend model")
                    try:
                        signature = encode_reasoning_signature(item_model, raw_item)
                    except ValueError as exc:
                        raise ChatGPTOAuthProtocolError(str(exc)) from exc
                    if signature is None:
                        raise ChatGPTOAuthProtocolError("encrypted reasoning item failed to produce a signature")
                    thinking_text = "" if thinking_display == "omitted" else item_text
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": thinking_text,
                            "signature": signature,
                        }
                    )
                elif thinking_display == "summarized" and item_text:
                    content.append({"type": "thinking", "thinking": item_text})
        elif thinking_display == "summarized" and response.reasoning_content:
            content.append({"type": "thinking", "thinking": response.reasoning_content})

    web_search_requests = 0
    for event in raw_events:
        if event.get("type") != "web_search_call":
            continue
        call_id = event.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ChatGPTOAuthProtocolError("provider web_search_call event requires a non-empty string id")
        raw_input = event.get("input")
        query = raw_input.get("query") if isinstance(raw_input, dict) else event.get("query")
        if not isinstance(query, str):
            raise ChatGPTOAuthProtocolError("provider web_search_call event requires a string query")
        sources = event.get("content") if isinstance(event.get("content"), list) else event.get("sources")
        if not isinstance(sources, list):
            raise ChatGPTOAuthProtocolError("provider web_search_call event requires a content/sources array")
        web_search_requests += 1
        content.append(
            {
                "type": "server_tool_use",
                "id": call_id,
                "name": "web_search",
                "input": {"query": query},
                "caller": {"type": "direct"},
            }
        )
        content.append(
            {
                "type": "web_search_tool_result",
                "tool_use_id": call_id,
                "content": sources,
                "caller": {"type": "direct"},
            }
        )

    if response.content:
        content.append({"type": "text", "text": response.content, "citations": None})

    for tc in response.tool_calls:
        arguments = _parse_anthropic_tool_arguments(tc.arguments)
        content.append(
            {
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": arguments,
                "caller": {"type": "direct"},
            }
        )

    stop_reason = _map_stop_reason(response.finish_reason, bool(response.tool_calls))
    if response.usage is None:
        raise ChatGPTOAuthProtocolError("provider response requires authoritative usage")
    usage_dict = _merge_usage_extensions(
        _anthropic_usage_from_internal(response.usage),
        response.raw,
    )
    if usage_dict.get("server_tool_use") is None and web_search_requests > 0:
        usage_dict["server_tool_use"] = {
            "web_search_requests": web_search_requests,
            "web_fetch_requests": 0,
        }

    result: dict[str, Any] = {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "container": None,
        "content": content,
        "context_management": None,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_dict,
    }
    effective_reasoning = response.reasoning_content or "".join(visible_reasoning_parts)
    if not thinking_disabled and thinking_display == "summarized" and effective_reasoning:
        result["codex_reasoning"] = effective_reasoning
    return result


def _raw_response_events(raw: dict | None) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ChatGPTOAuthProtocolError("provider raw response must be an object")
    events = raw.get("events")
    if not isinstance(events, list):
        raise ChatGPTOAuthProtocolError("provider raw response requires an events array")
    validated: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ChatGPTOAuthProtocolError(f"provider raw event {index} must be an object")
        if not isinstance(event.get("type"), str) or not event["type"]:
            raise ChatGPTOAuthProtocolError(f"provider raw event {index} requires a non-empty string type")
        validated.append(event)
    return validated


def _reject_unrepresentable_response_events(raw: dict | None) -> None:
    if any(event.get("type") == "web_search_call" for event in _raw_response_events(raw)):
        raise ChatGPTOAuthProtocolError(
            "provider web_search_call output cannot be represented losslessly by the Anthropic facade"
        )


def _uncached_input_tokens(total: int, cache_read: int | None, cache_write: int | None) -> int:
    # OpenAI totals include cache buckets; Anthropic exposes disjoint counters.
    cached = (cache_read or 0) + (cache_write or 0)
    if cached > total:
        raise ChatGPTOAuthProtocolError("usage cache tokens must not exceed input_tokens")
    return total - cached


def _anthropic_usage_from_internal(usage: Usage) -> dict[str, Any]:
    return {
        "cache_creation": None,
        "cache_creation_input_tokens": usage.cache_write_tokens,
        "cache_read_input_tokens": usage.cached_tokens,
        "inference_geo": None,
        "input_tokens": _uncached_input_tokens(
            usage.prompt_tokens, usage.cached_tokens, usage.cache_write_tokens
        ),
        "iterations": None,
        "output_tokens": usage.completion_tokens,
        "server_tool_use": None,
        "service_tier": None,
        "speed": None,
    }


def _merge_usage_extensions(
    usage: dict[str, Any],
    raw: dict | None,
) -> dict[str, Any]:
    for event in _raw_response_events(raw):
        if event.get("type") != "finish":
            continue
        raw_usage = event.get("usage")
        if raw_usage is None:
            continue
        if not isinstance(raw_usage, dict):
            raise ChatGPTOAuthProtocolError("provider finish event usage must be an object")
        for key, required_fields in (
            ("cache_creation", {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"}),
            ("server_tool_use", {"web_search_requests", "web_fetch_requests"}),
        ):
            value = raw_usage.get(key)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ChatGPTOAuthProtocolError(f"provider usage {key} must be an object or null")
            _validate_usage_counter_object(value, key, required_fields)
            usage[key] = value
        service_tier = raw_usage.get("service_tier")
        if service_tier is not None:
            if service_tier not in {"standard", "priority", "batch"}:
                raise ChatGPTOAuthProtocolError(
                    "provider usage service_tier must be standard, priority, batch, or null"
                )
            usage["service_tier"] = service_tier
        return usage
    return usage


def _map_stop_reason(finish_reason: str | None, has_tool_calls: bool) -> str:
    if finish_reason not in {None, "stop", "tool_calls"}:
        raise ChatGPTOAuthProtocolError(f"unsupported provider finish_reason {finish_reason!r}")
    if finish_reason is None:
        raise ChatGPTOAuthProtocolError("provider response requires a non-null finish_reason")
    if finish_reason == "tool_calls":
        if not has_tool_calls:
            raise ChatGPTOAuthProtocolError("provider finish_reason tool_calls requires at least one tool call")
        return "tool_use"
    if has_tool_calls:
        raise ChatGPTOAuthProtocolError("provider finish_reason stop conflicts with emitted tool calls")
    return "end_turn"


# ---------------------------------------------------------------------------
# Streaming adapter: provider events → Anthropic SSE
# ---------------------------------------------------------------------------


def anthropic_stream_adapter(
    event_stream: Iterator[dict[str, Any]],
    model: str,
    request_id: str,
    *,
    thinking_display: str = "summarized",
    backend_model: str | None = None,
    thinking_disabled: bool = False,
) -> Iterator[str]:
    """Convert provider chat_stream events into Anthropic SSE strings."""
    if thinking_display not in {"summarized", "omitted"}:
        raise ValueError("thinking_display must be 'summarized' or 'omitted'")
    if backend_model is not None and (
        not isinstance(backend_model, str) or not backend_model or backend_model != backend_model.strip()
    ):
        raise ValueError("backend_model must be a non-empty string when provided")
    yield _message_start_sse(model, request_id)
    yield from _render_anthropic_stream_events(
        event_stream,
        thinking_display=thinking_display,
        default_signature_model=backend_model or model,
        thinking_disabled=thinking_disabled,
    )


def _message_start_sse(model: str, request_id: str) -> str:
    return _sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "container": None,
                "content": [],
                "context_management": None,
                "stop_reason": None,
                "stop_sequence": None,
                # Provisional; message_delta carries authoritative cumulative usage.
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        },
    )


def _usage_int(value: Any, field: str, *, required: bool = False) -> int | None:
    if value is None and not required:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        requirement = "a non-negative integer"
        raise ChatGPTOAuthProtocolError(f"provider usage {field} must be {requirement}")
    return value


def _anthropic_usage_from_provider(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        raise ChatGPTOAuthProtocolError("provider finish usage must be an object")
    unsupported_aliases = sorted(
        set(usage)
        & {
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
            "cached_input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        }
    )
    if unsupported_aliases:
        raise ChatGPTOAuthProtocolError(
            "provider finish usage contains unsupported public aliases: " + ", ".join(unsupported_aliases)
        )
    token_details = usage.get("input_tokens_details")
    if token_details is not None and not isinstance(token_details, dict):
        raise ChatGPTOAuthProtocolError("provider usage input token details must be an object or null")
    cache_read: int | None = None
    cache_write: int | None = None
    if isinstance(token_details, dict):
        has_detail_cache_write = "cache_write_tokens" in token_details
        detail_cache_read = _usage_int(
            token_details.get("cached_tokens"),
            "cached_tokens",
            required=True,
        )
        detail_cache_write = _usage_int(
            token_details.get("cache_write_tokens"),
            "cache_write_tokens",
            required=has_detail_cache_write,
        )
        cache_read = detail_cache_read
        if has_detail_cache_write:
            cache_write = detail_cache_write
    input_tokens = _usage_int(
        usage.get("input_tokens"),
        "input_tokens",
        required=True,
    )
    output_tokens = _usage_int(
        usage.get("output_tokens"),
        "output_tokens",
        required=True,
    )
    total_tokens = _usage_int(
        usage.get("total_tokens"),
        "total_tokens",
        required=True,
    )
    if input_tokens is None or output_tokens is None or total_tokens is None:
        raise ChatGPTOAuthProtocolError("provider usage must include input_tokens, output_tokens, and total_tokens")
    if total_tokens != input_tokens + output_tokens:
        raise ChatGPTOAuthProtocolError("provider usage total_tokens must equal input_tokens plus output_tokens")
    server_tool_use = usage.get("server_tool_use")
    if server_tool_use is not None:
        if not isinstance(server_tool_use, dict):
            raise ChatGPTOAuthProtocolError("provider usage server_tool_use must be an object or null")
        _validate_usage_counter_object(
            server_tool_use,
            "server_tool_use",
            {"web_search_requests", "web_fetch_requests"},
        )
    cache_creation = usage.get("cache_creation")
    if cache_creation is not None:
        if not isinstance(cache_creation, dict):
            raise ChatGPTOAuthProtocolError("provider usage cache_creation must be an object or null")
        _validate_usage_counter_object(
            cache_creation,
            "cache_creation",
            {"ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"},
        )
    service_tier = usage.get("service_tier")
    if service_tier is not None and (
        not isinstance(service_tier, str) or service_tier not in {"standard", "priority", "batch"}
    ):
        raise ChatGPTOAuthProtocolError(
            "provider usage service_tier must be standard, priority, batch, or null"
        )
    out: dict[str, Any] = {
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "input_tokens": _uncached_input_tokens(input_tokens, cache_read, cache_write),
        "iterations": None,
        "output_tokens": output_tokens,
        "server_tool_use": server_tool_use,
    }
    return out


def _validate_usage_counter_object(
    value: dict[str, Any],
    field: str,
    required_fields: set[str],
) -> None:
    unknown = sorted(set(value) - required_fields)
    if unknown:
        raise ChatGPTOAuthProtocolError(f"provider usage {field} contains unsupported fields: " + ", ".join(unknown))
    missing = sorted(required_fields - set(value))
    if missing:
        raise ChatGPTOAuthProtocolError(f"provider usage {field} is missing required fields: " + ", ".join(missing))
    for name, counter in value.items():
        if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
            raise ChatGPTOAuthProtocolError(f"provider usage {field}.{name} must be a non-negative integer")


def _render_anthropic_stream_events(
    events: Iterator[dict[str, Any]],
    *,
    thinking_display: str = "summarized",
    default_signature_model: str | None = None,
    thinking_disabled: bool = False,
) -> Iterator[str]:
    block_index = 0
    current_block: str | None = None
    has_tool_calls = False
    web_search_requests = 0
    pending_thinking_item_id: str | None = None
    pending_thinking_parts: list[str] = []
    saw_pending_thinking_delta = False
    flushed_unsigned_thinking = False

    def _flush_open_unsigned_thinking() -> None:
        nonlocal pending_thinking_item_id, pending_thinking_parts, saw_pending_thinking_delta, flushed_unsigned_thinking
        if not saw_pending_thinking_delta:
            return
        if pending_thinking_item_id is not None:
            raise ChatGPTOAuthProtocolError(
                "provider reasoning delta with item_id ended without matching reasoning_item"
            )
        flushed_unsigned_thinking = True
        saw_pending_thinking_delta = False
        pending_thinking_parts = []
        pending_thinking_item_id = None

    for event in events:
        if not isinstance(event, dict):
            raise ChatGPTOAuthProtocolError("provider stream event must be an object")
        typ = event.get("type")

        if typ in ("reasoning_delta", "reasoning_raw_delta"):
            text = event.get("text")
            if not isinstance(text, str):
                raise ChatGPTOAuthProtocolError(f"provider {typ} event requires string text")
            event_item_id = event.get("item_id")
            if event_item_id is not None and (not isinstance(event_item_id, str) or not event_item_id):
                raise ChatGPTOAuthProtocolError(f"provider {typ} event item_id must be a non-empty string")
            if saw_pending_thinking_delta:
                if (pending_thinking_item_id is None) != (event_item_id is None):
                    raise ChatGPTOAuthProtocolError(
                        "provider reasoning delta item_id presence is inconsistent within one thinking block"
                    )
                if (
                    pending_thinking_item_id is not None
                    and event_item_id is not None
                    and event_item_id != pending_thinking_item_id
                ):
                    raise ChatGPTOAuthProtocolError(
                        "provider reasoning delta item_id changed before reasoning_item completion"
                    )
            else:
                if pending_thinking_item_id is not None and event_item_id != pending_thinking_item_id:
                    raise ChatGPTOAuthProtocolError(
                        "provider reasoning delta item_id does not match open thinking section"
                    )
                pending_thinking_item_id = event_item_id
            saw_pending_thinking_delta = True
            pending_thinking_parts.append(text)
            if thinking_disabled or thinking_display == "omitted":
                continue
            if current_block != "thinking":
                if current_block is not None:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
                current_block = "thinking"
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "thinking_delta", "thinking": text},
                },
            )
            yield _sse(
                "codex_reasoning_delta",
                {
                    "type": "codex_reasoning_delta",
                    "delta": text,
                },
            )

        elif typ == "reasoning_section_break":
            event_item_id = event.get("item_id")
            if event_item_id is not None and (not isinstance(event_item_id, str) or not event_item_id):
                raise ChatGPTOAuthProtocolError(
                    "provider reasoning_section_break event item_id must be a non-empty string"
                )
            if saw_pending_thinking_delta:
                if (pending_thinking_item_id is None) != (event_item_id is None):
                    raise ChatGPTOAuthProtocolError(
                        "provider reasoning_section_break item_id presence is inconsistent with open thinking block"
                    )
                if (
                    pending_thinking_item_id is not None
                    and event_item_id is not None
                    and event_item_id != pending_thinking_item_id
                ):
                    raise ChatGPTOAuthProtocolError(
                        "provider reasoning_section_break item_id does not match open thinking block"
                    )
            elif event_item_id is not None:
                if pending_thinking_item_id is not None and event_item_id != pending_thinking_item_id:
                    raise ChatGPTOAuthProtocolError(
                        "provider reasoning_section_break item_id changed before reasoning_item completion"
                    )
                pending_thinking_item_id = event_item_id
            continue

        elif typ == "reasoning_item":
            if flushed_unsigned_thinking:
                raise ChatGPTOAuthProtocolError("provider reasoning_item arrived after thinking block was closed")
            raw_item = event.get("item")
            if not isinstance(raw_item, dict) or raw_item.get("type") != "reasoning":
                raise ChatGPTOAuthProtocolError("provider reasoning_item event requires a reasoning item object")
            item_id = raw_item.get("id")
            if item_id is not None and (not isinstance(item_id, str) or not item_id):
                raise ChatGPTOAuthProtocolError("provider reasoning_item id must be a non-empty string when provided")
            encrypted_content = raw_item.get("encrypted_content")
            if encrypted_content is not None and not isinstance(encrypted_content, str):
                raise ChatGPTOAuthProtocolError("provider reasoning_item encrypted_content must be a string or null")
            try:
                item_summary, item_raw = reasoning_parts_from_response_items([raw_item])
            except ValueError as exc:
                raise ChatGPTOAuthProtocolError(str(exc)) from exc
            expected_text = item_summary + item_raw
            if pending_thinking_item_id is not None and item_id != pending_thinking_item_id:
                raise ChatGPTOAuthProtocolError("provider reasoning_item id does not match streamed reasoning deltas")
            if saw_pending_thinking_delta and "".join(pending_thinking_parts) != expected_text:
                raise ChatGPTOAuthProtocolError(
                    "provider reasoning_item summary does not match streamed reasoning deltas"
                )
            event_model = event.get("model")
            if event_model is not None and (
                not isinstance(event_model, str) or not event_model or event_model != event_model.strip()
            ):
                raise ChatGPTOAuthProtocolError("provider reasoning_item model must be a non-empty string")
            sig_model = event_model or default_signature_model

            if not thinking_disabled:
                if isinstance(encrypted_content, str) and encrypted_content:
                    if not isinstance(sig_model, str) or not sig_model:
                        raise ChatGPTOAuthProtocolError("provider reasoning_item requires a model for signature")
                    try:
                        signature = encode_reasoning_signature(sig_model, raw_item)
                    except ValueError as exc:
                        raise ChatGPTOAuthProtocolError(str(exc)) from exc
                    if signature is None:
                        raise ChatGPTOAuthProtocolError("provider reasoning_item failed to produce a signature")
                    if thinking_display == "omitted":
                        if current_block is not None:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            current_block = None
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": block_index,
                                "content_block": {"type": "thinking", "thinking": ""},
                            },
                        )
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "signature_delta", "signature": signature},
                            },
                        )
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1
                    else:
                        if current_block != "thinking":
                            if current_block is not None:
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                                block_index += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                },
                            )
                            current_block = "thinking"
                        if not saw_pending_thinking_delta and expected_text:
                            yield _sse(
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": block_index,
                                    "delta": {"type": "thinking_delta", "thinking": expected_text},
                                },
                            )
                            yield _sse(
                                "codex_reasoning_delta",
                                {
                                    "type": "codex_reasoning_delta",
                                    "delta": expected_text,
                                },
                            )
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "signature_delta", "signature": signature},
                            },
                        )
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1
                        current_block = None
                elif thinking_display == "summarized":
                    if not saw_pending_thinking_delta and expected_text:
                        if current_block != "thinking":
                            if current_block is not None:
                                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                                block_index += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": block_index,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                },
                            )
                            current_block = "thinking"
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "thinking_delta", "thinking": expected_text},
                            },
                        )
                        yield _sse(
                            "codex_reasoning_delta",
                            {
                                "type": "codex_reasoning_delta",
                                "delta": expected_text,
                            },
                        )
                    if current_block == "thinking":
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1
                        current_block = None

            saw_pending_thinking_delta = False
            pending_thinking_parts = []
            pending_thinking_item_id = None

        elif typ == "content":
            _flush_open_unsigned_thinking()
            text = event.get("text")
            if not isinstance(text, str):
                raise ChatGPTOAuthProtocolError("provider content event requires string text")
            if current_block != "text":
                if current_block is not None:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                    block_index += 1
                yield _sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": "", "citations": None},
                    },
                )
                current_block = "text"
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": text},
                },
            )

        elif typ == "tool_call":
            _flush_open_unsigned_thinking()
            has_tool_calls = True
            if current_block is not None:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
            tool_id = event.get("id")
            tool_name = event.get("name")
            tool_args = event.get("arguments")
            if not isinstance(tool_id, str):
                raise ChatGPTOAuthProtocolError("provider tool_call event requires a string id")
            if not isinstance(tool_name, str):
                raise ChatGPTOAuthProtocolError("provider tool_call event requires a string name")
            _parse_anthropic_tool_arguments(tool_args)
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": {},
                        "caller": {"type": "direct"},
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": tool_args},
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            current_block = None

        elif typ == "web_search_call":
            _flush_open_unsigned_thinking()
            if current_block is not None:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
            call_id = event.get("id")
            if not isinstance(call_id, str) or not call_id:
                raise ChatGPTOAuthProtocolError("provider web_search_call event requires a non-empty string id")
            raw_input = event.get("input")
            query = raw_input.get("query") if isinstance(raw_input, dict) else event.get("query")
            if not isinstance(query, str):
                raise ChatGPTOAuthProtocolError("provider web_search_call event requires a string query")
            sources = event.get("content") if isinstance(event.get("content"), list) else event.get("sources")
            if not isinstance(sources, list):
                raise ChatGPTOAuthProtocolError("provider web_search_call event requires a content/sources array")
            web_search_requests += 1
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": call_id,
                        "name": "web_search",
                        "input": {},
                        "caller": {"type": "direct"},
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps({"query": query}, ensure_ascii=False, separators=(",", ":")),
                    },
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": call_id,
                        "content": sources,
                        "caller": {"type": "direct"},
                    },
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
            block_index += 1
            current_block = None

        elif typ == "finish":
            _flush_open_unsigned_thinking()
            if current_block is not None:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                current_block = None

            finish_reason = event.get("finish_reason")
            if finish_reason is not None and (not isinstance(finish_reason, str) or not finish_reason):
                raise ChatGPTOAuthProtocolError("provider finish event finish_reason must be non-empty or null")
            stop_reason = _map_stop_reason(finish_reason, has_tool_calls)
            usage = _anthropic_usage_from_provider(event.get("usage"))
            if usage.get("server_tool_use") is None and web_search_requests > 0:
                usage["server_tool_use"] = {
                    "web_search_requests": web_search_requests,
                    "web_fetch_requests": 0,
                }

            message_delta: dict[str, Any] = {
                "type": "message_delta",
                "context_management": None,
                "delta": {"container": None, "stop_reason": stop_reason, "stop_sequence": None},
                "usage": usage,
            }
            yield _sse("message_delta", message_delta)
            yield _sse("message_stop", {"type": "message_stop"})
            return

        else:
            raise ChatGPTOAuthProtocolError(f"unsupported normalized response event type {typ!r}")

    raise ChatGPTOAuthProtocolError("provider stream ended without a finish event")


def _sse(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------


def format_anthropic_error(status: int, message: str) -> dict[str, Any]:
    type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "api_error",
        529: "overloaded_error",
    }
    return {
        "type": "error",
        "error": {
            "type": type_map.get(status, "api_error"),
            "message": message,
        },
    }
