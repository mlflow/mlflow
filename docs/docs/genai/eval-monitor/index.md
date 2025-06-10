---
description: >
  Learn how to evaluate and monitor your GenAI applications with MLflow for quality, cost, and latency throughout the development lifecycle.
last_update:
  date: 2025-06-03
---

# Evaluate & Monitor

MLflow's evaluation and monitoring capabilities help you systematically measure, improve, and maintain the quality of your GenAI applications throughout their lifecycle. From development through production, use the same [quality scorers](/genai/eval-monitor/concepts/scorers) to ensure your applications deliver accurate, reliable responses while managing cost and latency.

This page gives an overview of core evaluation and monitoring workflows and concepts, and provides links for more information.

:::note
Agent Evaluation is integrated with Managed MLflow 3. The Agent Evaluation SDK methods are now exposed through the `mlflow[databricks]>=3.1` SDK. See the [migration guide](/mlflow3/genai/agent-eval-migration) to update your [legacy Agent Evaluation](/generative-ai/agent-evaluation/index) code to MLflow 3 SDKs.
:::

## Evaluation during development

Test and improve your GenAI app iteratively by running evaluations against curated [evaluation datasets](/genai/eval-monitor/concepts/eval-datasets) using pre-built and custom [scorers](/genai/eval-monitor/concepts/scorers). MLflow's [evaluation harness](/genai/eval-monitor/concepts/eval-harness) helps you test new versions of your app and prompts in order to:

- Determine if your changes improved quality
- Identify root causes of quality issues
- Compare different versions of your app side-by-side
- Verify that changes did not cause regressions

## Monitoring in production

:::info[Beta]
Monitoring is in [Beta](/release-notes/release-types).
:::

Continuously track your deployed app's performance and quality. MLflow's monitoring capabilities enable you to:

- Automatically assess quality using the same [scorers](/genai/eval-monitor/concepts/scorers) as development
- Track operational metrics (latency, cost, errors)
- Identify underperforming queries to create [evaluation datasets](/genai/eval-monitor/concepts/eval-datasets)

## Getting started

Start with the [Evaluation Quickstart](/mlflow3/genai/getting-started/eval) to evaluate your first GenAI app in minutes.

## Next steps

Continue your journey with these recommended actions and tutorials.

- [Evaluate your app](/genai/eval-monitor/evaluate-app) - Learn the complete evaluation workflow
- [Use predefined LLM scorers](/genai/eval-monitor/predefined-judge-scorers) - Start with research-backed quality metrics
- [Create evaluation datasets](/genai/eval-monitor/build-eval-dataset) - Build comprehensive test sets from production data

## Reference guides

Explore detailed documentation for concepts and features mentioned in this guide.

- [Scorers](/genai/eval-monitor/concepts/scorers) - Understand how scorers assess GenAI applications
- [LLM judges](/genai/eval-monitor/concepts/judges/index) - Learn about using LLMs as evaluators
- [Evaluation Harness](/genai/eval-monitor/concepts/eval-harness) - Explore how MLflow orchestrates evaluations
