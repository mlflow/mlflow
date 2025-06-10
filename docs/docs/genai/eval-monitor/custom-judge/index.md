---
description: >
  Learn how to create custom LLM judges in MLflow for tailored GenAI quality assessments.
last_update:
  date: 2025-05-18
---

# Creating custom LLM scorers

While MLflow's [predefined LLM judge scorers](/genai/eval-monitor/predefined-judge-scorers) offer excellent starting points for common quality dimensions in simpler applications, you'll need to create custom LLM judges as your application becomes more complex and to tune your evaluation criteria to meet the specific, nuanced business requirements of your use case and align with your domain expert's judgement. MLflow provides robust and flexible ways to create **custom LLM judges** tailored to these unique requirements.

## Approaches for creating custom judges

MLflow offers 2 approaches to building custom judges. We reccomend starting with guidelines-based judges and only using prompt-based judges if you need more control or can't write your evaluation criteria as pass/fail guidelines. Guidelines-based judges have the distinct advantage of being easy to explain to business stakeholders and can often be directly written by domain experts.

### Guidelines-based scorers _(we suggest starting here)_

- **Best for:** Evaluations based on a clear set of specific, natural language criteria, framed as pass/fail conditions. Ideal for checking compliance with rules, style guides, or information inclusion/exclusion.
- **How it works:** You provide a set of plain-language rules that refer to specific inputs to or outputs from your app, for example `The response must be polite`. An LLM then determines if the guideline passes or fails and provides a rationale for why.

[Get started with guidelines &raquo;](/genai/eval-monitor/custom-judge/meets-guidelines)

### Prompt-based scorers

- **Best for:** Complex, nuanced evaluations where you need full control over the scorer's prompt or need to have the scorer specify multiple output values, for example, "great", "ok", "bad".
- **How it works:** You provide a prompt template that defines your evaluation criteria and has placeholders for specific fields in your app's trace. You define the output choices the scorer can select. An LLM then selects the appropiate output choice and provides a rationale for its selection.

[Get started with prompt-based judges &raquo;](/genai/eval-monitor/custom-judge/create-prompt-judge)

## Next steps

Continue your journey with these recommended actions and tutorials.

- [Create guidelines-based scorers](/genai/eval-monitor/custom-judge/meets-guidelines) - Define evaluation criteria using natural language rules (recommended)
- [Create prompt-based scorers](/genai/eval-monitor/custom-judge/create-prompt-judge) - Build complex judges with custom prompts and output choices
- [Run scorers in production](/genai/eval-monitor/run-scorer-in-prod) - Deploy your custom judges for continuous monitoring

## Reference guides

Explore detailed documentation for concepts and features mentioned in this guide.

- [LLM judges](/genai/eval-monitor/concepts/judges/index) - Understand how LLM judges work and their architecture
- [Custom judges: guidelines-based](/genai/eval-monitor/concepts/judges/guidelines) - Deep dive into guidelines-based evaluation
- [Custom judges: prompt-based](/genai/eval-monitor/concepts/judges/prompt-based-judge) - Technical details on prompt-based judges
