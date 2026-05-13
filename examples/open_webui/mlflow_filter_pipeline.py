# ruff: noqa
"""
title: MLflow Filter Pipeline
author: open-webui
date: 2026-04-20
version: 0.0.1
license: MIT
description: A filter pipeline that uses MLflow for tracing multi-turn chat sessions.
requirements: mlflow>=2.14.0
"""

from typing import List, Optional
import os
import re
import uuid

from utils.pipelines.main import get_last_assistant_message, get_last_user_message
from pydantic import BaseModel
import mlflow
from mlflow.entities import SpanType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey


def extract_latest_user_input(text: str) -> str:
    """If text contains a <chat_history> block, return only the last USER: segment inside it."""
    match = re.search(r"<chat_history>(.*?)</chat_history>", text, re.DOTALL)
    if match:
        history = match.group(1)
        user_messages = re.findall(r"USER:\s*(.*?)(?=\s*ASSISTANT:|\s*$)", history, re.DOTALL)
        if user_messages:
            return user_messages[-1].strip()
    return text


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        mlflow_tracking_uri: str = "http://localhost:5000"
        mlflow_experiment_name: str = "open-webui"
        debug: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "MLflow Filter"

        self.valves = self.Valves(**{
            "pipelines": ["*"],
            "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
            "mlflow_experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "open-webui"),
            "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
        })

        self.pending_inlets: dict = {}

    def log(self, message: str):
        if self.valves.debug:
            print(f"[DEBUG] {message}", flush=True)

    async def on_startup(self):
        self.log(f"on_startup triggered for {__name__}")
        self._setup_mlflow()

    async def on_shutdown(self):
        self.log(f"on_shutdown triggered for {__name__}")

    async def on_valves_updated(self):
        self.log("Valves updated, resetting MLflow config.")
        self._setup_mlflow()

    def _setup_mlflow(self):
        mlflow.set_tracking_uri(self.valves.mlflow_tracking_uri)
        mlflow.set_experiment(self.valves.mlflow_experiment_name)
        self.log(
            f"MLflow configured — uri: {self.valves.mlflow_tracking_uri}, "
            f"experiment: {self.valves.mlflow_experiment_name}"
        )

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("MLflow Filter INLET called")

        metadata = body.get("metadata", {})
        chat_id = body.get("chat_id") or metadata.get("chat_id") or str(uuid.uuid4())

        if chat_id == "local":
            session_id = metadata.get("session_id") or str(uuid.uuid4())
            metadata["session_id"] = session_id
            chat_id = f"temporary-session-{session_id}"

        metadata["chat_id"] = chat_id
        body["metadata"] = metadata

        self.pending_inlets[chat_id] = {
            "chat_id": chat_id,
            "input": extract_latest_user_input(get_last_user_message(body["messages"])),
            "model": body.get("model"),
            "user_email": user.get("email") if user else None,
        }

        self.log(f"Stored inlet snapshot for chat_id: {chat_id}")
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("MLflow Filter OUTLET called")

        chat_id = body.get("chat_id") or body.get("metadata", {}).get("chat_id")
        if not chat_id:
            self.log("[WARNING] No chat_id in outlet body — skipping trace")
            return body

        inlet_data = self.pending_inlets.pop(chat_id, None)
        if inlet_data is None:
            self.log(f"[WARNING] No inlet snapshot found for chat_id: {chat_id} — skipping trace")
            return body
        user_email = inlet_data["user_email"] or (user.get("email") if user else "unknown")
        model = inlet_data["model"] or body.get("model", "unknown")
        user_input = inlet_data["input"]

        assistant_message = get_last_assistant_message(body["messages"])
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])

        # Extract token usage if available
        token_usage = {}
        if assistant_message_obj:
            info = assistant_message_obj.get("usage") or {}
            input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
            output_tokens = info.get("eval_count") or info.get("completion_tokens")
            if input_tokens is not None:
                token_usage[TokenUsageKey.INPUT_TOKENS] = input_tokens
            if output_tokens is not None:
                token_usage[TokenUsageKey.OUTPUT_TOKENS] = output_tokens
            if input_tokens is not None and output_tokens is not None:
                token_usage[TokenUsageKey.TOTAL_TOKENS] = input_tokens + output_tokens

        try:
            with mlflow.start_span(name="chat_turn", span_type=SpanType.AGENT) as span:
                span.set_inputs({"user": user_input})
                span.set_outputs({"response": assistant_message})
                span.set_attribute(SpanAttributeKey.MODEL, model)
                if token_usage:
                    span.set_attribute(SpanAttributeKey.CHAT_USAGE, token_usage)

                # Groups all turns of this chat under one session in the MLflow UI
                mlflow.update_current_trace(
                    session_id=chat_id,
                    user=user_email,
                )

            self.log(f"MLflow trace logged for chat_id: {chat_id}")
        except Exception as e:
            warning = f"[WARNING] Failed to log MLflow trace ({type(e).__name__}) for chat_id: {chat_id}: {e}"
            self.log(warning)

        return body
