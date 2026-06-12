from __future__ import annotations

import json
import time
from typing import Any


class DeepSeekClient:
    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 120) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Model response does not contain a JSON object.")
        return json.loads(stripped[start : end + 1])

    def create_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.4,
        *,
        max_tokens: int | None = None,
        disable_thinking: bool = True,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        import httpx

        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens
        if disable_thinking:
            request_payload["thinking"] = {"type": "disabled"}

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=request_timeout or self.timeout,
                    trust_env=False,
                )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                return self._extract_json(content)
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                detail = self._extract_error_detail(exc.response)
                if status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"DeepSeek API 请求失败（HTTP {status_code}）：{detail}"
                ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise RuntimeError(f"DeepSeek API 网络请求失败：{exc}") from exc
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"DeepSeek 返回结果无法解析为 JSON：{exc}") from exc

        raise RuntimeError(f"DeepSeek 请求失败：{last_error}")

    def create_json_with_tool_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tool_name: str,
        tool_description: str,
        parameters_schema: dict[str, Any],
        temperature: float = 0.35,
        max_tokens: int | None = None,
        disable_thinking: bool = True,
        allow_fallback_json_mode: bool = True,
        request_timeout: float | None = None,
    ) -> dict[str, Any]:
        import httpx

        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "strict": True,
                        "description": tool_description,
                        "parameters": parameters_schema,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": tool_name}},
        }
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens
        if disable_thinking:
            request_payload["thinking"] = {"type": "disabled"}

        strict_base_url = (
            self.base_url if self.base_url.endswith("/beta") else f"{self.base_url}/beta"
        )
        try:
            payload = self._post_chat_completion(
                strict_base_url,
                request_payload,
                timeout=request_timeout or self.timeout,
            )
            return self._extract_tool_call_json(payload, tool_name)
        except Exception:
            if not allow_fallback_json_mode:
                raise
            return self.create_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                disable_thinking=disable_thinking,
                request_timeout=request_timeout,
            )

    def _post_chat_completion(
        self,
        base_url: str,
        request_payload: dict[str, Any],
        *,
        timeout: float,
    ) -> dict[str, Any]:
        import httpx

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = httpx.post(
                    f"{base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    timeout=timeout,
                    trust_env=False,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                detail = self._extract_error_detail(exc.response)
                if status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"DeepSeek API 请求失败（HTTP {status_code}）：{detail}"
                ) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.2 * (attempt + 1))
                    continue
                raise RuntimeError(f"DeepSeek API 网络请求失败：{exc}") from exc
        raise RuntimeError(f"DeepSeek 请求失败：{last_error}")

    @staticmethod
    def _extract_tool_call_json(payload: dict[str, Any], tool_name: str) -> dict[str, Any]:
        try:
            tool_calls = payload["choices"][0]["message"]["tool_calls"]
            for item in tool_calls:
                function = item.get("function", {})
                if function.get("name") == tool_name:
                    arguments = function.get("arguments", "")
                    return json.loads(arguments)
        except Exception as exc:
            raise RuntimeError(f"DeepSeek tool call 结果无法解析：{exc}") from exc
        raise RuntimeError("DeepSeek 未返回预期的 tool call 结构。")

    @staticmethod
    def _extract_error_detail(response: Any) -> str:
        try:
            payload = response.json()
        except Exception:
            text = response.text.strip()
            return text or "unknown error"
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
            message = payload.get("message")
            if message:
                return str(message)
        text = response.text.strip()
        return text or "unknown error"
