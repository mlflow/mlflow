import TabItem from "@theme/TabItem";
import { APILink } from "@site/src/components/APILink";
import { Card, LogoCard, CardGroup, PageCard, SmallLogoCard } from "@site/src/components/Card";

# Develop, Evaluate, and Deploy an Agentic Customer Service Chatbot with MLflow

GenAI application development is complex. It can involve multiple models, multimodal data, and complicated application logic. There are several unique challenges associated with GenAI applications:

- Different combinations of models, prompts, and inference parameters can have huge impacts on a GenAI application's performance. Keeping track of tests and experiments comparing performance across different combinations of these elements is challenging.
- It can be difficult to define what makes a GenAI model's responses "good" or "bad," making it challenging to evaluate GenAI application performance.
- Identifying sources of errors or breakdowns in application logic is very difficult when dealing with complex GenAI applications that include a mix of AI models calls and deterministic functions.

MLflow helps to solve these problems by providing a suite of tools for tracing and visualizing all of your GenAI model calls, evaluating your models and applications, building application logic into custom models, tracking and versioning your models, and deploying your models to production.

To demonstrate these capabilities, this multi-part guide will walk you through the process of building and deploying a customer service chatbot for a food delivery service, covering the entire lifecycle from early experimentation through structured evaluation.

<CardGroup>
<PageCard
link="/genai/tutorials/customer-service-chatbot/debug-tracing"
headerText="Part 1: Autologging and Tracing"
text="Covers informal experimentation with MLflow tracing."
/>
<PageCard
headerText="Part 2: Structured Evaluation"
text="Coming Soon"
/>
</CardGroup>

## How to use this guide

The four parts of this guide are meant to be read in sequence. Later sections build on the concepts and code introduced in earlier sections. Get started with [part 1](/genai/tutorials/customer-service-chatbot/debug-tracing).

## Prerequisites

This guide is designed to be accessible, with explanations of Python and MLflow concepts provided. However, readers without prior programming or software development experience may find it challenging. Familiarity with basic programming principles, Python syntax, and application development concepts will be beneficial.
