"""
Antigravity Router - Handles OpenAI format requests and converts to Antigravity API
处理 OpenAI 格式请求并转换为 Antigravity API 格式
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from log import log

from .antigravity_api import (
    build_antigravity_request_body,
    send_antigravity_request_no_stream,
    send_antigravity_request_stream,
)
from .credential_manager import CredentialManager
from .models import ChatCompletionRequest, Model, ModelList

# 创建路由器
router = APIRouter()
security = HTTPBearer()

# 全局凭证管理器实例
credential_manager = None


@asynccontextmanager
async def get_credential_manager():
    """获取全局凭证管理器实例"""
    global credential_manager
    if not credential_manager:
        credential_manager = CredentialManager()
        await credential_manager.initialize()
    yield credential_manager


async def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """验证用户密码"""
    from config import get_api_password

    password = await get_api_password()
    token = credentials.credentials
    if token != password:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="密码错误")
    return token


# 模型名称映射
def model_mapping(model_name: str) -> str:
    """
    OpenAI 模型名映射到 Antigravity 实际模型名

    参考文档:
    - claude-sonnet-4-5-thinking -> claude-sonnet-4-5
    - claude-opus-4-5 -> claude-opus-4-5-thinking
    - gemini-2.5-flash-thinking -> gemini-2.5-flash
    """
    mapping = {
        "claude-sonnet-4-5-thinking": "claude-sonnet-4-5",
        "claude-opus-4-5": "claude-opus-4-5-thinking",
        "gemini-2.5-flash-thinking": "gemini-2.5-flash",
    }
    return mapping.get(model_name, model_name)


def is_thinking_model(model_name: str) -> bool:
    """检测是否是思考模型"""
    thinking_models = [
        "gemini-2.5-pro",
        "gemini-3-pro",
        "claude-sonnet-4-5-thinking",
        "claude-opus-4-5",  # 会被映射为 claude-opus-4-5-thinking
    ]

    # 检查是否包含 -thinking 后缀
    if "-thinking" in model_name:
        return True

    # 检查是否匹配特定模型
    for thinking_model in thinking_models:
        if model_name.startswith(thinking_model):
            return True

    return False


def extract_images_from_content(content: Any) -> Dict[str, Any]:
    """
    从 OpenAI content 中提取文本和图片
    """
    result = {"text": "", "images": []}

    if isinstance(content, str):
        result["text"] = content
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    result["text"] += item.get("text", "")
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    # 解析 data:image/png;base64,xxx 格式
                    if image_url.startswith("data:image/"):
                        import re
                        match = re.match(r"^data:image/(\w+);base64,(.+)$", image_url)
                        if match:
                            mime_type = match.group(1)
                            base64_data = match.group(2)
                            result["images"].append({
                                "inlineData": {
                                    "mimeType": f"image/{mime_type}",
                                    "data": base64_data
                                }
                            })

    return result


def extract_images_from_anthropic_content(content: Any) -> Dict[str, Any]:
    """
    从 Anthropic Messages 的 content 中提取文本和图片。

    Anthropic 常见形态：
    - content 是字符串
    - content 是数组，元素可能为：
      - {"type":"text","text":"..."}
      - {"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}
    """
    result = {"text": "", "images": []}

    if isinstance(content, str):
        result["text"] = content
        return result

    if not isinstance(content, list):
        return result

    for item in content:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "text":
            result["text"] += item.get("text", "")
            continue

        if item_type == "image":
            source = item.get("source", {})
            if not isinstance(source, dict):
                continue
            if source.get("type") != "base64":
                continue

            media_type = source.get("media_type") or source.get("mediaType") or "image/png"
            base64_data = source.get("data", "")
            if base64_data:
                result["images"].append(
                    {"inlineData": {"mimeType": media_type, "data": base64_data}}
                )

    return result


def anthropic_system_to_text(system: Any) -> str:
    """
    将 Anthropic 的 system 字段归一化为纯文本。

    Anthropic 允许：
    - system: "..."
    - system: [{"type":"text","text":"..."}]
    """
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        text_parts: List[str] = []
        for item in system:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "".join(text_parts)
    return ""


def anthropic_messages_to_antigravity_contents(
    messages: List[Any], system: Any = None
) -> List[Dict[str, Any]]:
    """
    将 Anthropic Messages 格式转换为 Antigravity contents 格式。
    """
    contents: List[Dict[str, Any]] = []

    system_text = anthropic_system_to_text(system)
    system_injected = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            parts: List[Dict[str, Any]] = []

            # 兼容策略：将 system 合并到第一条 user 消息
            if system_text and not system_injected:
                parts.append({"text": system_text})
                system_injected = True

            extracted = extract_images_from_anthropic_content(content)
            if extracted["text"]:
                parts.append({"text": extracted["text"]})
            parts.extend(extracted["images"])

            if parts:
                contents.append({"role": "user", "parts": parts})

        elif role == "assistant":
            parts = []
            extracted = extract_images_from_anthropic_content(content)
            if extracted["text"]:
                parts.append({"text": extracted["text"]})
            parts.extend(extracted["images"])

            if parts:
                contents.append({"role": "model", "parts": parts})

    return contents


def convert_anthropic_tools_to_openai_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    """
    将 Anthropic tools 转为 OpenAI tools 形态，复用现有的 schema 清理与转换逻辑。

    Anthropic 常见形态：
    - {"name":"...","description":"...","input_schema":{...}}
    """
    if not isinstance(tools, list) or not tools:
        return None

    openai_tools: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name:
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", "") or "",
                    "parameters": tool.get("input_schema", {}) or {},
                },
            }
        )
    return openai_tools if openai_tools else None


def openai_messages_to_antigravity_contents(messages: List[Any]) -> List[Dict[str, Any]]:
    """
    将 OpenAI 消息格式转换为 Antigravity contents 格式
    """
    contents = []
    system_messages = []

    for msg in messages:
        role = getattr(msg, "role", "user")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", None)
        tool_call_id = getattr(msg, "tool_call_id", None)

        # 处理 system 消息 - 合并到第一条用户消息
        if role == "system":
            system_messages.append(content)
            continue

        # 处理 user 消息
        elif role == "user":
            parts = []

            # 如果有系统消息，添加到第一条用户消息
            if system_messages:
                for sys_msg in system_messages:
                    parts.append({"text": sys_msg})
                system_messages = []

            # 提取文本和图片
            extracted = extract_images_from_content(content)
            if extracted["text"]:
                parts.append({"text": extracted["text"]})
            parts.extend(extracted["images"])

            if parts:
                contents.append({"role": "user", "parts": parts})

        # 处理 assistant 消息
        elif role == "assistant":
            parts = []

            # 添加文本内容
            if content:
                extracted = extract_images_from_content(content)
                if extracted["text"]:
                    parts.append({"text": extracted["text"]})

            # 添加工具调用
            if tool_calls:
                for tool_call in tool_calls:
                    tc_id = getattr(tool_call, "id", None)
                    tc_type = getattr(tool_call, "type", "function")
                    tc_function = getattr(tool_call, "function", None)

                    if tc_function:
                        func_name = getattr(tc_function, "name", "")
                        func_args = getattr(tc_function, "arguments", "{}")

                        # 解析 arguments（可能是字符串）
                        if isinstance(func_args, str):
                            try:
                                args_dict = json.loads(func_args)
                            except:
                                args_dict = {"query": func_args}
                        else:
                            args_dict = func_args

                        parts.append({
                            "functionCall": {
                                "id": tc_id,
                                "name": func_name,
                                "args": args_dict
                            }
                        })

            if parts:
                contents.append({"role": "model", "parts": parts})

        # 处理 tool 消息
        elif role == "tool":
            parts = [{
                "functionResponse": {
                    "id": tool_call_id,
                    "name": getattr(msg, "name", "unknown"),
                    "response": {"output": content}
                }
            }]
            contents.append({"role": "user", "parts": parts})

    return contents


def convert_openai_tools_to_antigravity(tools: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    将 OpenAI 工具定义转换为 Antigravity 格式
    """
    if not tools:
        return None

    # Antigravity/Gemini 对 functionDeclarations.parameters 的 Schema 支持较严格：
    # - 不支持许多 JSON Schema/OpenAPI 关键字（会触发 400: Unknown name）
    # - "type"、"items" 等字段在某些生成器里可能是数组形式（会触发 400: cannot start list）
    # 这里采用“兼容优先”的最小降级策略：尽量清理/归一化，保证请求可发送。
    EXCLUDED_KEYS = {
        # 项目原有排除项
        "$schema",
        "additionalProperties",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        # Gemini 侧已知不支持项（参考 src/openai_transfer.py 的清理逻辑）
        "$id",
        "$ref",
        "$defs",
        "definitions",
        "title",
        "example",
        "examples",
        "readOnly",
        "writeOnly",
        "default",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "oneOf",
        "anyOf",
        "allOf",
        "const",
        "additionalItems",
        "contains",
        "patternProperties",
        "dependencies",
        "propertyNames",
        "if",
        "then",
        "else",
        "contentEncoding",
        "contentMediaType",
    }

    def _normalize_type_field(cleaned: Dict[str, Any], path: str) -> None:
        schema_type = cleaned.get("type")
        if not isinstance(schema_type, list):
            return

        # 兼容 ["string", "null"] 这类可空写法
        nullable = "null" in schema_type
        non_null_types = [t for t in schema_type if t != "null"]
        chosen_type = non_null_types[0] if non_null_types else (schema_type[0] if schema_type else None)

        if chosen_type is None:
            cleaned.pop("type", None)
        else:
            cleaned["type"] = chosen_type

        if nullable and "nullable" not in cleaned:
            cleaned["nullable"] = True

        log.warning(f"[ANTIGRAVITY][TOOLS] Schema 字段 type 为数组，已降级处理: {path}.type -> {cleaned.get('type')}, nullable={cleaned.get('nullable')}")

    def _normalize_items_field(cleaned: Dict[str, Any], path: str) -> None:
        items = cleaned.get("items")
        if not isinstance(items, list):
            return

        # 兼容 tuple validation: items 是数组（多 schema），Antigravity/Gemini 通常不支持
        if items:
            cleaned["items"] = items[0]
            log.warning(f"[ANTIGRAVITY][TOOLS] Schema 字段 items 为数组，已取第一个元素降级: {path}.items[0]")
        else:
            cleaned.pop("items", None)
            log.warning(f"[ANTIGRAVITY][TOOLS] Schema 字段 items 为空数组，已移除: {path}.items")

    def clean_parameters(obj: Any, path: str = "parameters") -> Any:
        """递归清理并归一化参数 schema（兼容优先的最小降级策略）"""
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for key, value in obj.items():
                if key in EXCLUDED_KEYS:
                    continue
                cleaned[key] = clean_parameters(value, f"{path}.{key}")

            # 归一化：修复 type/items 出现数组导致的 proto 解析错误
            _normalize_type_field(cleaned, path)
            _normalize_items_field(cleaned, path)

            # 归一化：properties 必须是对象
            if "properties" in cleaned and not isinstance(cleaned["properties"], dict):
                log.warning(f"[ANTIGRAVITY][TOOLS] Schema 字段 properties 非对象，已移除: {path}.properties")
                cleaned.pop("properties", None)

            # 归一化：required 必须是字符串数组
            if "required" in cleaned:
                if isinstance(cleaned["required"], list):
                    cleaned["required"] = [r for r in cleaned["required"] if isinstance(r, str)]
                else:
                    log.warning(f"[ANTIGRAVITY][TOOLS] Schema 字段 required 非数组，已移除: {path}.required")
                    cleaned.pop("required", None)

            # 兜底：如果有 properties/items 但没有 type，补齐 type
            if "properties" in cleaned and "type" not in cleaned:
                cleaned["type"] = "object"
            if "items" in cleaned and "type" not in cleaned:
                cleaned["type"] = "array"

            return cleaned

        if isinstance(obj, list):
            return [clean_parameters(item, f"{path}[]") for item in obj]

        return obj

    function_declarations = []

    for tool in tools:
        # 兼容：tool 可能是 Pydantic 模型或 dict
        if hasattr(tool, "model_dump"):
            tool_dict = tool.model_dump()
        elif hasattr(tool, "dict"):
            tool_dict = tool.dict()
        else:
            tool_dict = tool

        if not isinstance(tool_dict, dict):
            log.warning(f"[ANTIGRAVITY][TOOLS] 无法解析 tool 定义（非对象），已跳过: {type(tool_dict)}")
            continue

        tool_type = tool_dict.get("type", "function")
        if tool_type != "function":
            continue

        function = tool_dict.get("function")
        if not isinstance(function, dict):
            log.warning("[ANTIGRAVITY][TOOLS] tool.function 缺失或非对象，已跳过")
            continue

        func_name = function.get("name")
        assert func_name is not None, "Function name is required"
        func_desc = function.get("description", "")
        func_params = function.get("parameters", {})

        # 转换为字典（如果是 Pydantic 模型）
        if hasattr(func_params, "dict"):
            func_params = func_params.dict()
        elif hasattr(func_params, "model_dump"):
            func_params = func_params.model_dump()

        # 清理参数
        cleaned_params = clean_parameters(func_params, f"parameters({func_name})")

        function_declarations.append({
            "name": func_name,
            "description": func_desc,
            "parameters": cleaned_params
        })

    if function_declarations:
        return [{"functionDeclarations": function_declarations}]

    return None


def generate_generation_config(
    parameters: Dict[str, Any],
    enable_thinking: bool,
    model_name: str
) -> Dict[str, Any]:
    """
    生成 Antigravity generationConfig
    """
    config = {
        "candidateCount": 1,
        "stopSequences": [
            "<|user|>",
            "<|bot|>",
            "<|context_request|>",
            "<|endoftext|>",
            "<|end_of_turn|>"
        ]
    }

    # 添加温度参数
    if "temperature" in parameters:
        config["temperature"] = parameters["temperature"]

    # 添加 topP
    if "top_p" in parameters:
        config["topP"] = parameters["top_p"]

    # 添加 topK
    if "top_k" in parameters:
        config["topK"] = parameters["top_k"]
    else:
        config["topK"] = 50  # 默认值

    # 添加 maxOutputTokens
    if "max_tokens" in parameters:
        config["maxOutputTokens"] = parameters["max_tokens"]

    # 思考模型配置
    if enable_thinking:
        config["thinkingConfig"] = {
            "includeThoughts": True,
            "thinkingBudget": 1024
        }

        # Claude 思考模型：删除 topP 参数
        if "claude" in model_name.lower():
            config.pop("topP", None)

    return config


def convert_to_openai_tool_call(function_call: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 Antigravity functionCall 转换为 OpenAI tool_call
    """
    return {
        "id": function_call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
        "type": "function",
        "function": {
            "name": function_call.get("name", ""),
            "arguments": json.dumps(function_call.get("args", {}))
        }
    }


async def convert_antigravity_stream_to_openai(
    response: Any,
    stream_ctx: Any,
    client: Any,
    model: str,
    request_id: str,
    credential_manager: Any,
    credential_name: str
):
    """
    将 Antigravity 流式响应转换为 OpenAI 格式的 SSE 流

    """
    state = {
        "tool_calls": [],
        "emitted_content": False,
        "emitted_reasoning": False,
        "sent_role": False,
        "success_recorded": False,
        "final_sent": False,
        "last_usage_metadata": None,
        "last_finish_reason": None,
    }

    created = int(time.time())

    try:
        def _extract_sse_data_payload(line: str) -> str:
            """
            从 SSE 单行中提取 data payload。

            兼容上游两种常见格式：
            - "data: {...}"
            - "data:{...}"
            """
            if not line or not line.startswith("data:"):
                return ""
            payload = line[len("data:") :]
            if payload.startswith(" "):
                payload = payload[1:]
            return payload

        def build_delta_chunk(delta: Dict[str, Any]) -> str:
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": None
                }]
            }
            return f"data: {json.dumps(chunk)}\n\n"

        def emit_delta(delta: Dict[str, Any]) -> str:
            """
            兼容性输出：
            - 一些 OpenAI→Anthropic 转换器/客户端会严格要求 delta 中存在 content 字段（可为空字符串）
            - 也可能要求首个“有效 chunk”携带 role=assistant 才会创建消息容器
            """
            if not state["sent_role"]:
                delta = {"role": "assistant", **delta}
                state["sent_role"] = True
            if "content" not in delta:
                delta = {**delta, "content": ""}
            return build_delta_chunk(delta)

        def emit_final_chunk(
            finish_reason_raw: Any,
            usage_metadata: Optional[Dict[str, Any]] = None,
        ) -> str:
            """
            发送最终 chunk（包含 finish_reason）。

            重要：有些中转平台在做 OpenAI→Anthropic 时，会依赖 OpenAI 的
            `finish_reason` 来触发 message_stop；如果只发 [DONE] 可能导致前端不渲染。
            """
            # 确定 finish_reason
            openai_finish_reason = "stop"
            if state["tool_calls"]:
                openai_finish_reason = "tool_calls"
            elif finish_reason_raw == "MAX_TOKENS":
                openai_finish_reason = "length"

            # 兼容性：部分 OpenAI→Anthropic 中转会依赖 delta.content（哪怕是空字符串）
            # 来触发 block/message 的收尾；同时若此前从未发送过 role，也在最终块补上。
            final_delta: Dict[str, Any] = {"content": ""}
            if not state["sent_role"]:
                final_delta = {"role": "assistant", **final_delta}
                state["sent_role"] = True

            chunk: Dict[str, Any] = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": final_delta,
                        "finish_reason": openai_finish_reason,
                    }
                ],
            }

            if usage_metadata:
                chunk["usage"] = {
                    "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
                    "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
                    "total_tokens": usage_metadata.get("totalTokenCount", 0),
                }

            return f"data: {json.dumps(chunk)}\n\n"

        async for line in response.aiter_lines():
            if state["final_sent"]:
                break

            payload = _extract_sse_data_payload(line)
            if not payload:
                continue

            # 上游也可能发送 data: [DONE]
            if payload.strip() == "[DONE]":
                break

            # 解析 SSE 数据
            try:
                data = json.loads(payload)
            except:
                continue

            # 记录第一次成功响应（以成功解析到 JSON 为准）
            if not state["success_recorded"]:
                if credential_name and credential_manager:
                    await credential_manager.record_api_call_result(
                        credential_name, True, is_antigravity=True
                    )
                state["success_recorded"] = True

            # 提取 parts
            response_obj = data.get("response", {}) if isinstance(data, dict) else {}
            candidates = response_obj.get("candidates", []) if isinstance(response_obj, dict) else []
            first_candidate = candidates[0] if candidates else {}
            parts = first_candidate.get("content", {}).get("parts", []) if isinstance(first_candidate, dict) else []

            # 记录 usage / finishReason（可能只在最后出现，也可能在中途就出现）
            usage_metadata = response_obj.get("usageMetadata") if isinstance(response_obj, dict) else None
            if isinstance(usage_metadata, dict):
                state["last_usage_metadata"] = usage_metadata

            finish_reason = first_candidate.get("finishReason") if isinstance(first_candidate, dict) else None
            if finish_reason:
                state["last_finish_reason"] = finish_reason

            for part in parts:
                # 处理思考内容
                if part.get("thought") is True:
                    reasoning_text = part.get("text", "")
                    if reasoning_text:
                        state["emitted_reasoning"] = True
                        yield emit_delta({"reasoning_content": reasoning_text})

                # 处理图片数据 (inlineData)
                elif "inlineData" in part:
                    # 提取图片数据
                    inline_data = part["inlineData"]
                    mime_type = inline_data.get("mimeType", "image/png")
                    base64_data = inline_data.get("data", "")

                    # 转换为 Markdown 格式的图片
                    image_markdown = f"\n\n![生成的图片](data:{mime_type};base64,{base64_data})\n\n"
                    state["emitted_content"] = True

                    # 发送图片块
                    yield emit_delta({"content": image_markdown})

                # 处理普通文本
                elif "text" in part:
                    # 添加文本内容
                    text = part.get("text", "")
                    if text:
                        state["emitted_content"] = True

                    # 发送文本块
                    if text:
                        yield emit_delta({"content": text})

                # 处理工具调用
                elif "functionCall" in part:
                    tool_call = convert_to_openai_tool_call(part["functionCall"])
                    state["tool_calls"].append(tool_call)

            # 检查是否结束
            if finish_reason and not state["final_sent"]:
                # 如果只有思考内容，没有任何可见 content，补一个占位，避免部分前端显示空消息
                if state["emitted_reasoning"] and not state["emitted_content"] and not state["tool_calls"]:
                    yield emit_delta({"content": "[模型正在思考中，请稍后再试或重新提问]"})

                # 发送工具调用
                if state["tool_calls"]:
                    yield emit_delta({"tool_calls": state["tool_calls"]})

                yield emit_final_chunk(finish_reason, state.get("last_usage_metadata"))
                state["final_sent"] = True
                break

        # 发送结束标记
        if not state["final_sent"]:
            # 上游没有给出 finishReason，也要补齐 final chunk，避免中转平台无法 message_stop
            if state["emitted_reasoning"] and not state["emitted_content"] and not state["tool_calls"]:
                yield emit_delta({"content": "[模型正在思考中，请稍后再试或重新提问]"})

            if state["tool_calls"]:
                yield emit_delta({"tool_calls": state["tool_calls"]})

            yield emit_final_chunk(state.get("last_finish_reason"), state.get("last_usage_metadata"))
            state["final_sent"] = True

        yield "data: [DONE]\n\n"

    except Exception as e:
        log.error(f"[ANTIGRAVITY] Streaming error: {e}")
        error_response = {
            "error": {
                "message": str(e),
                "type": "api_error",
                "code": 500
            }
        }
        yield f"data: {json.dumps(error_response)}\n\n"
    finally:
        # 确保清理所有资源
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception as e:
            log.debug(f"[ANTIGRAVITY] Error closing stream context: {e}")
        try:
            await client.aclose()
        except Exception as e:
            log.debug(f"[ANTIGRAVITY] Error closing client: {e}")


def _anthropic_sse_event(event: str, data: Dict[str, Any]) -> str:
    """构造 Anthropic/Claude Messages SSE 事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _map_antigravity_finish_reason_to_anthropic(finish_reason: Any, has_tool_use: bool) -> str:
    """将 Antigravity/Gemini 的 finishReason 映射到 Anthropic stop_reason。"""
    if has_tool_use:
        return "tool_use"
    if finish_reason == "MAX_TOKENS":
        return "max_tokens"
    # Gemini/Antigravity 还有 STOP/SAFETY 等，这里先按 end_turn 兜底
    return "end_turn"


def convert_antigravity_response_to_anthropic(
    response_data: Dict[str, Any],
    model: str,
    message_id: str,
) -> Dict[str, Any]:
    """
    将 Antigravity 非流式响应转换为 Anthropic Messages 格式。
    """
    parts = (
        response_data.get("response", {})
        .get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )

    thinking_text = ""
    text_content = ""
    tool_uses: List[Dict[str, Any]] = []

    for part in parts:
        if part.get("thought") is True:
            thinking_text += part.get("text", "")
            continue

        if "inlineData" in part:
            inline_data = part["inlineData"]
            mime_type = inline_data.get("mimeType", "image/png")
            base64_data = inline_data.get("data", "")
            if base64_data:
                text_content += f"\n\n![生成的图片](data:{mime_type};base64,{base64_data})\n\n"
            continue

        if "text" in part:
            text_content += part.get("text", "")
            continue

        if "functionCall" in part:
            function_call = part.get("functionCall", {})
            if isinstance(function_call, dict):
                tool_uses.append(
                    {
                        "type": "tool_use",
                        "id": function_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name": function_call.get("name", "") or "",
                        "input": function_call.get("args", {}) or {},
                    }
                )
            continue

    content_blocks: List[Dict[str, Any]] = []
    if thinking_text:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": thinking_text,
                "signature": "",
            }
        )

    if tool_uses:
        content_blocks.extend(tool_uses)

    if text_content:
        content_blocks.append({"type": "text", "text": text_content})

    finish_reason_raw = (
        response_data.get("response", {})
        .get("candidates", [{}])[0]
        .get("finishReason")
    )

    usage_metadata = response_data.get("response", {}).get("usageMetadata", {}) or {}
    input_tokens = usage_metadata.get("promptTokenCount", 0) or 0
    output_tokens = usage_metadata.get("candidatesTokenCount", 0) or 0

    stop_reason = _map_antigravity_finish_reason_to_anthropic(
        finish_reason_raw, has_tool_use=bool(tool_uses)
    )

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": output_tokens,
        },
    }


async def convert_antigravity_stream_to_anthropic(
    response: Any,
    stream_ctx: Any,
    client: Any,
    model: str,
    message_id: str,
    credential_manager: Any,
    credential_name: str,
):
    """
    将 Antigravity 流式响应转换为 Anthropic Messages SSE。

    目标：输出完整的 message_start/content_block_*/message_delta/message_stop，
    避免依赖中转平台的 OpenAI→Anthropic 转换导致的渲染/收尾缺失问题。
    """
    state = {
        "success_recorded": False,
        "thinking_index": None,
        "text_index": None,
        "next_index": 0,
        "has_tool_use": False,
        "last_finish_reason": None,
        "last_usage_metadata": None,
        "thinking_open": False,
        "text_open": False,
    }

    # message_start
    yield _anthropic_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            },
        },
    )

    def _extract_sse_data_payload(line: str) -> str:
        if not line or not line.startswith("data:"):
            return ""
        payload = line[len("data:") :]
        if payload.startswith(" "):
            payload = payload[1:]
        return payload

    def _ensure_thinking_block_started():
        if state["thinking_index"] is None:
            state["thinking_index"] = state["next_index"]
            state["next_index"] += 1
        idx = state["thinking_index"]
        if not state["thinking_open"]:
            state["thinking_open"] = True
            yield _anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                },
            )

    def _ensure_text_block_started():
        if state["text_index"] is None:
            state["text_index"] = state["next_index"]
            state["next_index"] += 1
        idx = state["text_index"]
        if not state["text_open"]:
            state["text_open"] = True
            yield _anthropic_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {"type": "text", "text": ""},
                },
            )
        # 不返回值：调用侧直接从 state["text_index"] 读取 index

    try:
        async for line in response.aiter_lines():
            payload = _extract_sse_data_payload(line)
            if not payload:
                continue

            if payload.strip() == "[DONE]":
                break

            try:
                data = json.loads(payload)
            except Exception:
                continue

            # 记录第一次成功响应（以成功解析到 JSON 为准）
            if not state["success_recorded"]:
                if credential_name and credential_manager:
                    await credential_manager.record_api_call_result(
                        credential_name, True, is_antigravity=True
                    )
                state["success_recorded"] = True

            response_obj = data.get("response", {}) if isinstance(data, dict) else {}
            candidates = response_obj.get("candidates", []) if isinstance(response_obj, dict) else []
            first_candidate = candidates[0] if candidates else {}
            parts = (
                first_candidate.get("content", {}).get("parts", [])
                if isinstance(first_candidate, dict)
                else []
            )

            usage_metadata = response_obj.get("usageMetadata") if isinstance(response_obj, dict) else None
            if isinstance(usage_metadata, dict):
                state["last_usage_metadata"] = usage_metadata

            finish_reason = first_candidate.get("finishReason") if isinstance(first_candidate, dict) else None
            if finish_reason:
                state["last_finish_reason"] = finish_reason

            for part in parts:
                if part.get("thought") is True:
                    reasoning_text = part.get("text", "")
                    if reasoning_text:
                        # 开启 thinking block
                        for evt in _ensure_thinking_block_started():
                            yield evt
                        idx = state["thinking_index"]
                        yield _anthropic_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "thinking_delta", "thinking": reasoning_text},
                            },
                        )
                    continue

                if "inlineData" in part:
                    inline_data = part["inlineData"]
                    mime_type = inline_data.get("mimeType", "image/png")
                    base64_data = inline_data.get("data", "")
                    if base64_data:
                        image_markdown = f"\n\n![生成的图片](data:{mime_type};base64,{base64_data})\n\n"
                        if state["thinking_open"] and state["thinking_index"] is not None:
                            yield _anthropic_sse_event(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": state["thinking_index"]},
                            )
                            state["thinking_open"] = False
                        for evt in _ensure_text_block_started():
                            yield evt
                        idx = state["text_index"]
                        yield _anthropic_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": image_markdown},
                            },
                        )
                    continue

                if "text" in part:
                    text = part.get("text", "")
                    if text:
                        # 兼容：在开始输出 text 之前，先结束 thinking block（若存在）
                        if state["thinking_open"] and state["thinking_index"] is not None:
                            yield _anthropic_sse_event(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": state["thinking_index"]},
                            )
                            state["thinking_open"] = False

                        for evt in _ensure_text_block_started():
                            yield evt
                        idx = state["text_index"]
                        yield _anthropic_sse_event(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )
                    continue

                if "functionCall" in part:
                    # 以完整 tool_use block 形式一次性输出（简化流式 tool delta）
                    function_call = part.get("functionCall", {})
                    if isinstance(function_call, dict):
                        if state["thinking_open"] and state["thinking_index"] is not None:
                            yield _anthropic_sse_event(
                                "content_block_stop",
                                {"type": "content_block_stop", "index": state["thinking_index"]},
                            )
                            state["thinking_open"] = False
                        state["has_tool_use"] = True
                        tool_index = state["next_index"]
                        state["next_index"] += 1

                        tool_use_block = {
                            "type": "tool_use",
                            "id": function_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                            "name": function_call.get("name", "") or "",
                            "input": function_call.get("args", {}) or {},
                        }
                        yield _anthropic_sse_event(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": tool_index,
                                "content_block": tool_use_block,
                            },
                        )
                        yield _anthropic_sse_event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": tool_index},
                        )
                    continue

            if finish_reason:
                break

        # 收尾：停止已开启的 blocks
        if state["thinking_open"] and state["thinking_index"] is not None:
            yield _anthropic_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": state["thinking_index"]},
            )
            state["thinking_open"] = False

        if state["text_open"] and state["text_index"] is not None:
            yield _anthropic_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": state["text_index"]},
            )
            state["text_open"] = False

        usage_metadata = state.get("last_usage_metadata") or {}
        output_tokens = usage_metadata.get("candidatesTokenCount", 0) if isinstance(usage_metadata, dict) else 0
        stop_reason = _map_antigravity_finish_reason_to_anthropic(
            state.get("last_finish_reason"), has_tool_use=bool(state.get("has_tool_use"))
        )

        yield _anthropic_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens or 0},
            },
        )

        yield _anthropic_sse_event("message_stop", {"type": "message_stop"})

    except Exception as e:
        log.error(f"[ANTIGRAVITY] Anthropic streaming error: {e}")
        yield _anthropic_sse_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(e)},
            },
        )
    finally:
        try:
            await stream_ctx.__aexit__(None, None, None)
        except Exception as e:
            log.debug(f"[ANTIGRAVITY] Error closing stream context: {e}")
        try:
            await client.aclose()
        except Exception as e:
            log.debug(f"[ANTIGRAVITY] Error closing client: {e}")


def convert_antigravity_response_to_openai(
    response_data: Dict[str, Any],
    model: str,
    request_id: str
) -> Dict[str, Any]:
    """
    将 Antigravity 非流式响应转换为 OpenAI 格式
    """
    # 提取 parts
    parts = response_data.get("response", {}).get("candidates", [{}])[0].get("content", {}).get("parts", [])

    content = ""
    thinking_content = ""
    tool_calls = []

    for part in parts:
        # 处理思考内容
        if part.get("thought") is True:
            thinking_content += part.get("text", "")

        # 处理图片数据 (inlineData)
        elif "inlineData" in part:
            inline_data = part["inlineData"]
            mime_type = inline_data.get("mimeType", "image/png")
            base64_data = inline_data.get("data", "")
            # 转换为 Markdown 格式的图片
            content += f"\n\n![生成的图片](data:{mime_type};base64,{base64_data})\n\n"

        # 处理普通文本
        elif "text" in part:
            content += part.get("text", "")

        # 处理工具调用
        elif "functionCall" in part:
            tool_calls.append(convert_to_openai_tool_call(part["functionCall"]))

    # 仅当没有正常内容但有思考内容时，补一个占位，避免部分前端显示空消息
    if not content and thinking_content and not tool_calls:
        content = "[模型正在思考中，请稍后再试或重新提问]"

    # 构建 OpenAI 响应
    message = {
        "role": "assistant",
        "content": content
    }

    if thinking_content:
        message["reasoning_content"] = thinking_content

    if tool_calls:
        message["tool_calls"] = tool_calls

    # 确定 finish_reason
    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"

    finish_reason_raw = response_data.get("response", {}).get("candidates", [{}])[0].get("finishReason")
    if finish_reason_raw == "MAX_TOKENS":
        finish_reason = "length"

    # 提取使用统计
    usage_metadata = response_data.get("response", {}).get("usageMetadata", {})
    usage = {
        "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
        "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
        "total_tokens": usage_metadata.get("totalTokenCount", 0)
    }

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": usage
    }


@router.get("/antigravity/v1/models", response_model=ModelList)
async def list_models():
    """返回 OpenAI 格式的模型列表 - 动态从 Antigravity API 获取"""
    from src.credential_manager import get_credential_manager
    from .antigravity_api import fetch_available_models

    try:
        # 获取凭证管理器
        cred_mgr = await get_credential_manager()

        # 从 Antigravity API 获取模型列表（返回 OpenAI 格式的字典列表）
        models = await fetch_available_models(cred_mgr)

        if not models:
            # 如果获取失败，直接返回空列表
            log.warning("[ANTIGRAVITY] Failed to fetch models from API, returning empty list")
            return ModelList(data=[])

        # models 已经是 OpenAI 格式的字典列表，直接转换为 Model 对象
        return ModelList(data=[Model(**m) for m in models])

    except Exception as e:
        log.error(f"[ANTIGRAVITY] Error fetching models: {e}")
        # 返回空列表
        return ModelList(data=[])


@router.post("/antigravity/v1/chat/completions")
async def chat_completions(request: Request, token: str = Depends(authenticate)):
    """
    处理 OpenAI 格式的聊天完成请求，转换为 Antigravity API
    """
    # 获取原始请求数据
    try:
        raw_data = await request.json()
    except Exception as e:
        log.error(f"Failed to parse JSON request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    # 创建请求对象
    try:
        request_data = ChatCompletionRequest(**raw_data)
    except Exception as e:
        log.error(f"Request validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Request validation error: {str(e)}")

    # 健康检查
    if (
        len(request_data.messages) == 1
        and getattr(request_data.messages[0], "role", None) == "user"
        and getattr(request_data.messages[0], "content", None) == "Hi"
    ):
        return JSONResponse(
            content={
                "choices": [{"message": {"role": "assistant", "content": "antigravity API 正常工作中"}}]
            }
        )

    # 获取凭证管理器
    from src.credential_manager import get_credential_manager
    cred_mgr = await get_credential_manager()

    # 提取参数
    model = request_data.model
    messages = request_data.messages
    stream = getattr(request_data, "stream", False)
    tools = getattr(request_data, "tools", None)

    # 模型名称映射
    actual_model = model_mapping(model)
    enable_thinking = is_thinking_model(model)

    log.info(f"[ANTIGRAVITY] Request: model={model} -> {actual_model}, stream={stream}, thinking={enable_thinking}")

    # 转换消息格式
    try:
        contents = openai_messages_to_antigravity_contents(messages)
    except Exception as e:
        log.error(f"Failed to convert messages: {e}")
        raise HTTPException(status_code=500, detail=f"Message conversion failed: {str(e)}")

    # 转换工具定义
    antigravity_tools = convert_openai_tools_to_antigravity(tools)

    # 生成配置参数
    parameters = {
        "temperature": getattr(request_data, "temperature", None),
        "top_p": getattr(request_data, "top_p", None),
        "max_tokens": getattr(request_data, "max_tokens", None),
    }
    # 过滤 None 值
    parameters = {k: v for k, v in parameters.items() if v is not None}

    generation_config = generate_generation_config(parameters, enable_thinking, actual_model)

    # 获取凭证信息（用于 projectId 和 sessionId）
    cred_result = await cred_mgr.get_valid_credential(is_antigravity=True)
    if not cred_result:
        log.error("当前无可用 antigravity 凭证")
        raise HTTPException(status_code=500, detail="当前无可用 antigravity 凭证")

    _, credential_data = cred_result
    project_id = credential_data.get("projectId", "default-project")
    session_id = credential_data.get("sessionId", f"session-{uuid.uuid4().hex}")

    # 构建 Antigravity 请求体
    request_body = build_antigravity_request_body(
        contents=contents,
        model=actual_model,
        project_id=project_id,
        session_id=session_id,
        tools=antigravity_tools,
        generation_config=generation_config,
    )

    # 生成请求 ID
    request_id = f"chatcmpl-{int(time.time() * 1000)}"

    # 发送请求
    try:
        if stream:
            # 流式请求
            resources, cred_name, cred_data = await send_antigravity_request_stream(
                request_body, cred_mgr
            )
            # resources 是一个元组: (response, stream_ctx, client)
            response, stream_ctx, client = resources

            # 转换并返回流式响应,传递资源管理对象
            return StreamingResponse(
                convert_antigravity_stream_to_openai(
                    response, stream_ctx, client, model, request_id, cred_mgr, cred_name
                ),
                media_type="text/event-stream"
            )
        else:
            # 非流式请求
            response_data, cred_name, cred_data = await send_antigravity_request_no_stream(
                request_body, cred_mgr
            )

            # 转换并返回响应
            openai_response = convert_antigravity_response_to_openai(response_data, model, request_id)
            return JSONResponse(content=openai_response)

    except Exception as e:
        log.error(f"[ANTIGRAVITY] Request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Antigravity API request failed: {str(e)}")


@router.post("/antigravity/v1/messages")
async def messages(request: Request, token: str = Depends(authenticate)):
    """
    Anthropic/Claude Messages 兼容端点（直接输出 Claude SSE 协议）。

    用途：避免依赖外部中转平台进行 OpenAI→Anthropic 转换时出现的收尾缺失问题。
    """
    try:
        raw_data = await request.json()
    except Exception as e:
        log.error(f"Failed to parse JSON request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    if not isinstance(raw_data, dict):
        raise HTTPException(status_code=400, detail="Invalid request body")

    model = raw_data.get("model")
    messages_payload = raw_data.get("messages", [])
    system = raw_data.get("system", None)
    stream = bool(raw_data.get("stream", False))
    tools = raw_data.get("tools", None)

    if not model or not isinstance(model, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'model'")
    if not isinstance(messages_payload, list):
        raise HTTPException(status_code=400, detail="Missing or invalid 'messages'")

    # 健康检查：messages=[{"role":"user","content":"Hi"}]
    if (
        len(messages_payload) == 1
        and isinstance(messages_payload[0], dict)
        and messages_payload[0].get("role") == "user"
        and messages_payload[0].get("content") == "Hi"
    ):
        message_id = f"msg_ag_{uuid.uuid4().hex[:24]}"
        return JSONResponse(
            content={
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "antigravity messages API 正常工作中"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                },
            }
        )

    # 获取凭证管理器
    from src.credential_manager import get_credential_manager

    cred_mgr = await get_credential_manager()

    # 模型映射与 thinking 判定
    actual_model = model_mapping(model)
    enable_thinking = is_thinking_model(model)
    log.info(
        f"[ANTIGRAVITY][MESSAGES] Request: model={model} -> {actual_model}, stream={stream}, thinking={enable_thinking}"
    )

    # 转换 Anthropic messages -> Antigravity contents
    try:
        contents = anthropic_messages_to_antigravity_contents(messages_payload, system=system)
    except Exception as e:
        log.error(f"Failed to convert anthropic messages: {e}")
        raise HTTPException(status_code=500, detail=f"Message conversion failed: {str(e)}")

    # tools: Anthropic -> OpenAI -> Antigravity
    openai_tools = convert_anthropic_tools_to_openai_tools(tools)
    antigravity_tools = convert_openai_tools_to_antigravity(openai_tools)

    # generationConfig
    parameters = {
        "temperature": raw_data.get("temperature", None),
        "top_p": raw_data.get("top_p", None),
        "max_tokens": raw_data.get("max_tokens", None),
    }
    parameters = {k: v for k, v in parameters.items() if v is not None}
    generation_config = generate_generation_config(parameters, enable_thinking, actual_model)

    # 获取 projectId / sessionId
    cred_result = await cred_mgr.get_valid_credential(is_antigravity=True)
    if not cred_result:
        log.error("当前无可用 antigravity 凭证")
        raise HTTPException(status_code=500, detail="当前无可用 antigravity 凭证")
    _, credential_data = cred_result
    project_id = credential_data.get("projectId", "default-project")
    session_id = credential_data.get("sessionId", f"session-{uuid.uuid4().hex}")

    request_body = build_antigravity_request_body(
        contents=contents,
        model=actual_model,
        project_id=project_id,
        session_id=session_id,
        tools=antigravity_tools,
        generation_config=generation_config,
    )

    message_id = f"msg_ag_{uuid.uuid4().hex[:24]}"

    try:
        if stream:
            resources, cred_name, _cred_data = await send_antigravity_request_stream(
                request_body, cred_mgr
            )
            response, stream_ctx, client = resources
            return StreamingResponse(
                convert_antigravity_stream_to_anthropic(
                    response, stream_ctx, client, model, message_id, cred_mgr, cred_name
                ),
                media_type="text/event-stream",
            )

        response_data, _cred_name, _cred_data = await send_antigravity_request_no_stream(
            request_body, cred_mgr
        )
        anthropic_response = convert_antigravity_response_to_anthropic(
            response_data, model, message_id
        )
        return JSONResponse(content=anthropic_response)

    except Exception as e:
        log.error(f"[ANTIGRAVITY][MESSAGES] Request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Antigravity API request failed: {str(e)}")
