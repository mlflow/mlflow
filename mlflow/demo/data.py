from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlflow.demo.base import DEMO_PROMPT_PREFIX
from mlflow.entities.model_registry import PromptVersion

# =============================================================================
# Prompt Data Definitions
# =============================================================================

_CUSTOMER_SUPPORT_NAME = f"{DEMO_PROMPT_PREFIX}.prompts.customer-support"
_DOCUMENT_SUMMARIZER_NAME = f"{DEMO_PROMPT_PREFIX}.prompts.document-summarizer"
_CODE_REVIEWER_NAME = f"{DEMO_PROMPT_PREFIX}.prompts.code-reviewer"


@dataclass
class DemoPromptDef:
    name: str
    versions: list[PromptVersion]


CUSTOMER_SUPPORT_PROMPT = DemoPromptDef(
    name=_CUSTOMER_SUPPORT_NAME,
    versions=[
        PromptVersion(
            name=_CUSTOMER_SUPPORT_NAME,
            version=1,
            template="You are a customer support agent. Help the user with: {{query}}",
            commit_message="Initial customer support prompt",
            aliases=["baseline"],
        ),
        PromptVersion(
            name=_CUSTOMER_SUPPORT_NAME,
            version=2,
            template=(
                "You are a friendly and professional customer support agent. "
                "Respond in a helpful, empathetic tone.\n\n"
                "User query: {{query}}"
            ),
            commit_message="Add tone and style guidance",
            aliases=["tone-guidance"],
        ),
        PromptVersion(
            name=_CUSTOMER_SUPPORT_NAME,
            version=3,
            template=(
                "You are a friendly and professional customer support agent for {{company_name}}. "
                "Respond in a helpful, empathetic tone.\n\n"
                "Context: {{context}}\n\n"
                "User query: {{query}}"
            ),
            commit_message="Add company context and conversation history",
            aliases=["with-context"],
        ),
        PromptVersion(
            name=_CUSTOMER_SUPPORT_NAME,
            version=4,
            template=[
                {
                    "role": "system",
                    "content": (
                        "You are a friendly and professional customer support agent "
                        "for {{company_name}}. Follow these guidelines:\n"
                        "- Be empathetic and patient\n"
                        "- Provide clear, actionable solutions\n"
                        "- Escalate complex issues appropriately\n"
                        "- Always verify customer satisfaction before closing"
                    ),
                },
                {"role": "user", "content": "Context: {{context}}\n\nQuery: {{query}}"},
            ],
            commit_message="Convert to chat format with detailed guidelines",
            aliases=["production"],
        ),
    ],
)

DOCUMENT_SUMMARIZER_PROMPT = DemoPromptDef(
    name=_DOCUMENT_SUMMARIZER_NAME,
    versions=[
        PromptVersion(
            name=_DOCUMENT_SUMMARIZER_NAME,
            version=1,
            template="Summarize the following document:\n\n{{document}}",
            commit_message="Initial summarization prompt",
            aliases=["baseline"],
        ),
        PromptVersion(
            name=_DOCUMENT_SUMMARIZER_NAME,
            version=2,
            template=(
                "Summarize the following document in {{max_words}} words or less:\n\n{{document}}"
            ),
            commit_message="Add length constraint parameter",
            aliases=["length-constraint"],
        ),
        PromptVersion(
            name=_DOCUMENT_SUMMARIZER_NAME,
            version=3,
            template=(
                "Summarize the following document for a {{audience}} audience. "
                "Keep the summary under {{max_words}} words.\n\n"
                "Document:\n{{document}}"
            ),
            commit_message="Add audience targeting",
            aliases=["audience-targeting"],
        ),
        PromptVersion(
            name=_DOCUMENT_SUMMARIZER_NAME,
            version=4,
            template=[
                {
                    "role": "system",
                    "content": (
                        "You are a document summarization expert. Create concise, accurate "
                        "summaries that capture the essential information while maintaining "
                        "the original meaning."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize this document for a {{audience}} audience.\n"
                        "Maximum length: {{max_words}} words.\n\n"
                        "Include:\n"
                        "1. Main topic/thesis\n"
                        "2. Key points (3-5 bullets)\n"
                        "3. Conclusion or main takeaway\n\n"
                        "Document:\n{{document}}"
                    ),
                },
            ],
            commit_message="Add structured output format with key points",
            aliases=["production"],
        ),
    ],
)

CODE_REVIEWER_PROMPT = DemoPromptDef(
    name=_CODE_REVIEWER_NAME,
    versions=[
        PromptVersion(
            name=_CODE_REVIEWER_NAME,
            version=1,
            template=(
                "Review the following code and provide feedback:\n\n```{{language}}\n{{code}}\n```"
            ),
            commit_message="Initial code review prompt",
            aliases=["baseline"],
        ),
        PromptVersion(
            name=_CODE_REVIEWER_NAME,
            version=2,
            template=(
                "Review the following {{language}} code for:\n"
                "- Bugs and errors\n"
                "- Performance issues\n"
                "- Code style\n\n"
                "```{{language}}\n{{code}}\n```"
            ),
            commit_message="Add specific review categories",
            aliases=["review-categories"],
        ),
        PromptVersion(
            name=_CODE_REVIEWER_NAME,
            version=3,
            template=(
                "Review the following {{language}} code. For each issue found, specify:\n"
                "- Severity: Critical, Major, Minor, or Suggestion\n"
                "- Category: Bug, Performance, Security, Style, or Maintainability\n"
                "- Line number (if applicable)\n"
                "- Recommended fix\n\n"
                "```{{language}}\n{{code}}\n```"
            ),
            commit_message="Add severity levels and structured feedback format",
            aliases=["severity-levels"],
        ),
        PromptVersion(
            name=_CODE_REVIEWER_NAME,
            version=4,
            template=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert code reviewer. Analyze code for bugs, security "
                        "vulnerabilities, performance issues, and maintainability concerns. "
                        "Provide actionable feedback with clear explanations and suggested fixes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Review this {{language}} code:\n\n"
                        "```{{language}}\n{{code}}\n```\n\n"
                        "Provide feedback in this format:\n"
                        "## Summary\n"
                        "Brief overview of code quality.\n\n"
                        "## Issues Found\n"
                        "For each issue:\n"
                        "- **[Severity]** Category: Description\n"
                        "  - Line: X\n"
                        "  - Fix: Recommendation\n\n"
                        "## Positive Aspects\n"
                        "What the code does well."
                    ),
                },
            ],
            commit_message="Production-ready with structured markdown output",
            aliases=["production"],
        ),
    ],
)

DEMO_PROMPTS: list[DemoPromptDef] = [
    CUSTOMER_SUPPORT_PROMPT,
    DOCUMENT_SUMMARIZER_PROMPT,
    CODE_REVIEWER_PROMPT,
]


# =============================================================================
# Trace Data Definitions
# =============================================================================


@dataclass
class LinkedPromptRef:
    """Reference to a prompt version for linking to traces."""

    prompt_name: str
    version: int


@dataclass
class ToolCall:
    """Tool call with input/output for agent traces."""

    name: str
    input: dict[str, Any]
    output: dict[str, Any]


@dataclass
class PromptTemplateValues:
    """Template values for prompt-based traces.

    Contains the prompt name, template, and variable values used to render the prompt.
    This allows traces to show the resolved prompt with interpolated values.
    """

    prompt_name: str
    template: str
    variables: dict[str, str]

    def render(self) -> str:
        """Render the template with the variable values."""
        result = self.template
        for key, value in self.variables.items():
            result = result.replace(f"{{{{{key}}}}}", value)
        return result


@dataclass
class DemoTrace:
    """Demo trace with query, two response versions, and expected ground truth.

    - v1_response: Initial/baseline agent output (less accurate, more verbose)
    - v2_response: Improved agent output (better quality, closer to expected)
    - expected_response: Ground truth for evaluation
    - prompt_template: Optional prompt template info for prompt-based traces
    """

    query: str
    v1_response: str
    v2_response: str
    expected_response: str
    trace_type: str
    tools: list[ToolCall] = field(default_factory=list)
    session_id: str | None = None
    session_user: str | None = None
    turn_index: int | None = None
    prompt_template: PromptTemplateValues | None = None


# =============================================================================
# RAG Traces (2 traces)
# =============================================================================

RAG_TRACES: list[DemoTrace] = [
    DemoTrace(
        query="What is MLflow Tracing and how does it help with LLM observability?",
        v1_response=(
            "MLflow Tracing is a feature that helps you understand what's happening "
            "in your LLM applications. It captures information about your app's execution "
            "and shows it in the UI somewhere."
        ),
        v2_response=(
            "MLflow Tracing provides comprehensive observability for LLM applications by "
            "capturing the execution flow as hierarchical spans. Each span records inputs, "
            "outputs, latency, and metadata, making it easy to debug and optimize your AI systems."
        ),
        expected_response=(
            "MLflow Tracing provides observability for LLM applications, capturing "
            "prompts, model calls, and tool invocations as hierarchical spans with "
            "inputs, outputs, and latency information."
        ),
        trace_type="rag",
    ),
    DemoTrace(
        query="How do I use mlflow.evaluate() to assess my LLM's output quality?",
        v1_response=(
            "MLflow has an evaluate() function. You pass it some data and scorers "
            "and it gives you back metrics. The results are logged automatically I think."
        ),
        v2_response=(
            "Use mlflow.evaluate() by passing your model/data and a list of scorers like "
            "relevance() or faithfulness(). It returns per-row scores and aggregate metrics, "
            "all automatically logged to your MLflow experiment for easy comparison."
        ),
        expected_response=(
            "Use mlflow.evaluate() with your model and scorers (e.g., relevance, faithfulness). "
            "Results include per-row scores and aggregate metrics, logged to MLflow."
        ),
        trace_type="rag",
    ),
]

# =============================================================================
# Agent Traces (2 traces)
# =============================================================================

AGENT_TRACES: list[DemoTrace] = [
    DemoTrace(
        query="What's the weather in San Francisco and should I bring an umbrella today?",
        v1_response=(
            "The weather in San Francisco is currently 62 degrees with partly cloudy skies. "
            "There's some chance of rain today, but I'm not sure exactly how much."
        ),
        v2_response=(
            "It's currently 62F and partly cloudy in San Francisco with only a 15% chance "
            "of rain. You probably don't need an umbrella today, but a light jacket might "
            "be nice for the evening fog!"
        ),
        expected_response=(
            "San Francisco is 62F and partly cloudy with 15% rain chance. "
            "No umbrella needed, but consider a light jacket for evening fog."
        ),
        trace_type="agent",
        tools=[
            ToolCall(
                name="get_weather",
                input={"city": "San Francisco", "units": "fahrenheit"},
                output={
                    "temperature": 62,
                    "condition": "partly cloudy",
                    "rain_chance": 15,
                    "humidity": 68,
                },
            ),
        ],
    ),
    DemoTrace(
        query="Calculate the compound interest on $10,000 at 5% annual rate for 10 years",
        v1_response=(
            "Based on my calculation, $10,000 invested at 5% annual interest "
            "compounded yearly for 10 years would grow to around $16,289 or so."
        ),
        v2_response=(
            "With annual compounding, $10,000 at 5% interest for 10 years grows to "
            "$16,288.95. The formula is: Principal x (1 + rate)^years = 10000 x 1.05^10. "
            "You'll earn $6,288.95 in interest over the period."
        ),
        expected_response=(
            "$10,000 at 5% annual compound interest for 10 years = $16,288.95. "
            "Total interest earned: $6,288.95."
        ),
        trace_type="agent",
        tools=[
            ToolCall(
                name="calculator",
                input={"expression": "10000 * (1.05 ** 10)", "precision": 2},
                output={"result": 16288.95, "formula": "10000 * 1.05^10"},
            ),
        ],
    ),
]

# =============================================================================
# Prompt-Based Traces (6 traces - 2 per prompt type)
# =============================================================================

PROMPT_TRACES: list[DemoTrace] = [
    # Customer Support Prompts (2 traces)
    DemoTrace(
        query=(
            "I can't log into my account. It says my password is wrong but I'm sure it's correct."
        ),
        v1_response=(
            "Try resetting your password using the forgot password link. "
            "If that doesn't work, contact support."
        ),
        v2_response=(
            "I'm sorry you're having trouble logging in! This is frustrating, I know. "
            "Let's fix this: First, try the 'Forgot Password' link on the login page. "
            "If you still can't get in after resetting, I can manually verify your account "
            "and send you a secure reset link. Which would you prefer?"
        ),
        expected_response=(
            "I understand login issues are frustrating. Try 'Forgot Password' first, "
            "or I can manually reset your account access."
        ),
        trace_type="prompt",
        prompt_template=PromptTemplateValues(
            prompt_name="customer-support",
            template=(
                "You are a customer support agent for {{company_name}}. "
                "Help the user with their issue.\n\n"
                "Context: {{context}}\n\n"
                "User query: {{query}}"
            ),
            variables={
                "company_name": "TechCorp",
                "context": "User has been a customer for 2 years with no previous issues",
                "query": "I can't log into my account. Password seems wrong.",
            },
        ),
    ),
    DemoTrace(
        query="I was charged twice for my subscription this month. Can you help?",
        v1_response=(
            "I see the duplicate charge. I'll process a refund which should "
            "appear in 3-5 business days."
        ),
        v2_response=(
            "I sincerely apologize for the duplicate charge - that shouldn't have happened! "
            "I can see both charges on your account. I've already initiated a refund for "
            "$29.99, which will appear in your account within 2-3 business days. I've also "
            "added a $10 credit to your account for the inconvenience. Is there anything "
            "else I can help you with?"
        ),
        expected_response=(
            "I apologize for the duplicate charge. Refund of $29.99 initiated, "
            "arriving in 2-3 business days, plus $10 credit for the inconvenience."
        ),
        trace_type="prompt",
        prompt_template=PromptTemplateValues(
            prompt_name="customer-support",
            template=(
                "You are a customer support agent for {{company_name}}. "
                "Help the user with their issue.\n\n"
                "Context: {{context}}\n\n"
                "User query: {{query}}"
            ),
            variables={
                "company_name": "TechCorp",
                "context": "Premium subscriber, $29.99/month plan, charged on 1st and 3rd",
                "query": "I was charged twice for my subscription this month.",
            },
        ),
    ),
    # Document Summarizer Prompts (2 traces)
    DemoTrace(
        query="Summarize the Q3 2024 earnings report for investors.",
        v1_response="Q3 2024: Revenue up, profits increased, outlook is positive.",
        v2_response=(
            "**Q3 2024 Earnings Summary for Investors**\n\n"
            "- Revenue: $2.3B (+15% YoY)\n"
            "- Net Income: $450M (+23% YoY)\n"
            "- EPS: $1.82 (beat estimates by $0.12)\n\n"
            "Key Drivers: Cloud services growth (32%), cost optimization savings ($45M)\n"
            "Outlook: Full-year guidance raised to $9.1B revenue"
        ),
        expected_response=(
            "Q3 2024: Revenue $2.3B (+15%), Net Income $450M (+23%), EPS $1.82. "
            "Cloud growth and cost savings drove results. FY guidance raised to $9.1B."
        ),
        trace_type="prompt",
        prompt_template=PromptTemplateValues(
            prompt_name="document-summarizer",
            template=(
                "Summarize the following document for a {{audience}} audience. "
                "Keep the summary under {{max_words}} words.\n\n"
                "Document:\n{{document}}"
            ),
            variables={
                "audience": "investor",
                "max_words": "150",
                "document": (
                    "Q3 2024 Financial Results: Total revenue reached $2.3 billion, "
                    "representing a 15% increase year-over-year. Net income was $450 million, "
                    "up 23% from Q3 2023. Earnings per share of $1.82 exceeded analyst "
                    "estimates of $1.70. Cloud services segment grew 32%..."
                ),
            },
        ),
    ),
    DemoTrace(
        query="Create a summary of the machine learning research paper for a technical blog.",
        v1_response="The paper proposes a new transformer architecture that is more efficient.",
        v2_response=(
            "**Novel Sparse Transformer Architecture**\n\n"
            "This paper introduces SparseFormer, achieving 12% better accuracy on standard "
            "benchmarks while reducing memory usage by 40%. Key innovations:\n\n"
            "1. **Sparse Attention**: O(n log n) complexity vs O(n^2) for vanilla transformers\n"
            "2. **Dynamic Positional Encoding**: Adapts to sequence length automatically\n"
            "3. **Gradient Checkpointing**: Enables training on longer sequences\n\n"
            "Validated on translation (WMT) and summarization (CNN/DailyMail) tasks."
        ),
        expected_response=(
            "SparseFormer: New transformer with 12% accuracy gain and 40% memory reduction. "
            "Uses O(n log n) sparse attention and dynamic positional encoding."
        ),
        trace_type="prompt",
        prompt_template=PromptTemplateValues(
            prompt_name="document-summarizer",
            template=(
                "Summarize the following document for a {{audience}} audience. "
                "Keep the summary under {{max_words}} words.\n\n"
                "Document:\n{{document}}"
            ),
            variables={
                "audience": "technical",
                "max_words": "200",
                "document": (
                    "Abstract: We present SparseFormer, a novel transformer architecture "
                    "that achieves state-of-the-art results while significantly reducing "
                    "computational requirements. Our sparse attention mechanism reduces "
                    "complexity from O(n^2) to O(n log n)..."
                ),
            },
        ),
    ),
    # Code Reviewer Prompts (2 traces)
    DemoTrace(
        query="Review this Python function that calculates factorial.",
        v1_response="The code works but could be improved for performance.",
        v2_response=(
            "## Code Review: factorial function\n\n"
            "### Issues Found\n"
            "1. **[Critical] Bug - Line 3**: No handling for negative numbers\n"
            "   - Fix: Add `if n < 0: raise ValueError('n must be non-negative')`\n\n"
            "2. **[Major] Performance - Line 5**: Recursive implementation has O(n) stack depth\n"
            "   - Fix: Use iterative approach or `math.factorial()` for large n\n\n"
            "3. **[Minor] Style**: Missing type hints and docstring\n\n"
            "### Positive Aspects\n"
            "- Clean, readable implementation\n"
            "- Correct base case handling"
        ),
        expected_response=(
            "Critical: No negative number handling. Major: Recursive approach risks stack "
            "overflow for large n. Minor: Add type hints and docstring."
        ),
        trace_type="prompt",
        prompt_template=PromptTemplateValues(
            prompt_name="code-reviewer",
            template=(
                "Review the following {{language}} code. Identify bugs, performance issues, "
                "and style problems.\n\n```{{language}}\n{{code}}\n```"
            ),
            variables={
                "language": "python",
                "code": (
                    "def factorial(n):\n"
                    "    if n == 0:\n"
                    "        return 1\n"
                    "    return n * factorial(n - 1)"
                ),
            },
        ),
    ),
    DemoTrace(
        query="Review this JavaScript async function for API calls.",
        v1_response="There are some issues with error handling in the async code.",
        v2_response=(
            "## Code Review: fetchUserData async function\n\n"
            "### Issues Found\n"
            "1. **[Critical] Security - Line 4**: API key exposed in client-side code\n"
            "   - Fix: Move to environment variable or backend proxy\n\n"
            "2. **[Critical] Bug - Line 7**: Missing try/catch around await\n"
            "   - Fix: Wrap in try/catch or add .catch() handler\n\n"
            "3. **[Major] Performance - Lines 8-10**: Sequential awaits should be parallel\n"
            "   - Fix: `const [user, posts] = await Promise.all([...]);`\n\n"
            "4. **[Minor] Style**: Inconsistent error message format\n\n"
            "### Positive Aspects\n"
            "- Good use of async/await syntax\n"
            "- Clear function naming"
        ),
        expected_response=(
            "Critical: API key exposure, missing error handling. Major: Use Promise.all() "
            "for parallel requests. Minor: Inconsistent error formatting."
        ),
        trace_type="prompt",
        prompt_template=PromptTemplateValues(
            prompt_name="code-reviewer",
            template=(
                "Review the following {{language}} code. Identify bugs, performance issues, "
                "and style problems.\n\n```{{language}}\n{{code}}\n```"
            ),
            variables={
                "language": "javascript",
                "code": (
                    "async function fetchUserData(userId) {\n"
                    "  const apiKey = 'sk-1234567890';\n"
                    "  const user = await fetch(`/api/users/${userId}`);\n"
                    "  const posts = await fetch(`/api/users/${userId}/posts`);\n"
                    "  return { user: user.json(), posts: posts.json() };\n"
                    "}"
                ),
            },
        ),
    ),
]

# =============================================================================
# Session Traces (3 sessions with varying turns: 2, 3, 2 = 7 traces total)
# =============================================================================

SESSION_TRACES: list[DemoTrace] = [
    # Session 1: MLflow Setup (2 turns)
    DemoTrace(
        query="I'm new to MLflow. How do I get started with experiment tracking?",
        v1_response=(
            "To get started with MLflow, first install it with pip. Then you can "
            "start using the tracking API to log your experiments."
        ),
        v2_response=(
            "Welcome to MLflow! Here's your quickstart:\n\n"
            "1. Install: `pip install mlflow`\n"
            "2. Start the UI: `mlflow server --port 5000`\n"
            "3. In your code:\n"
            "```python\n"
            "import mlflow\n"
            "with mlflow.start_run():\n"
            "    mlflow.log_param('learning_rate', 0.01)\n"
            "    mlflow.log_metric('accuracy', 0.95)\n"
            "```\n"
            "Open http://localhost:5000 to see your experiments!"
        ),
        expected_response=(
            "Install with `pip install mlflow`, run `mlflow server`, then use "
            "mlflow.start_run() with log_param() and log_metric() to track experiments."
        ),
        trace_type="session",
        session_id="session-mlflow-setup",
        session_user="alice@example.com",
        turn_index=1,
    ),
    DemoTrace(
        query="Great! How do I log a trained model so I can deploy it later?",
        v1_response=(
            "You can use mlflow.sklearn.log_model() or similar functions to log models. "
            "They get saved with the run."
        ),
        v2_response=(
            "After training, log your model with the appropriate flavor:\n\n"
            "```python\n"
            "# For scikit-learn\n"
            "mlflow.sklearn.log_model(model, 'model')\n\n"
            "# For PyTorch\n"
            "mlflow.pytorch.log_model(model, 'model')\n"
            "```\n\n"
            "The model is saved with your run and can be loaded later with:\n"
            "`mlflow.sklearn.load_model('runs:/<run_id>/model')`\n\n"
            "Ready to register it for deployment?"
        ),
        expected_response=(
            "Use mlflow.<flavor>.log_model() (e.g., sklearn, pytorch) to save models. "
            "Load later with mlflow.<flavor>.load_model('runs:/<run_id>/model')."
        ),
        trace_type="session",
        session_id="session-mlflow-setup",
        session_user="alice@example.com",
        turn_index=2,
    ),
    # Session 2: Debugging a RAG Application (3 turns)
    DemoTrace(
        query="My RAG application is giving inconsistent answers. How can I debug it?",
        v1_response=(
            "You should look at your prompts and retrieved documents to see what's happening. "
            "MLflow has some tools for this."
        ),
        v2_response=(
            "Let's debug this systematically with MLflow Tracing!\n\n"
            "1. **Enable tracing** to capture every step:\n"
            "```python\n"
            "mlflow.tracing.enable()\n"
            "```\n\n"
            "2. **Check the Traces tab** in MLflow UI to see:\n"
            "   - What documents were retrieved\n"
            "   - The actual prompts sent to the LLM\n"
            "   - Token usage and latency per step\n\n"
            "What kind of inconsistency are you seeing - retrieval or generation issues?"
        ),
        expected_response=(
            "Enable mlflow.tracing.enable() to capture execution flow. "
            "Check the Traces tab to inspect retrieved docs, prompts, and latency."
        ),
        trace_type="session",
        session_id="session-rag-debug",
        session_user="bob@example.com",
        turn_index=1,
    ),
    DemoTrace(
        query="I think it's a retrieval issue - sometimes it finds irrelevant documents.",
        v1_response=(
            "You should look at the retrieval scores and maybe adjust your embedding model "
            "or chunk size."
        ),
        v2_response=(
            "Retrieval issues are common! Here's how to diagnose:\n\n"
            "1. **In MLflow Traces**, click on the RETRIEVER span to see:\n"
            "   - Retrieved document IDs and relevance scores\n"
            "   - The query embedding used\n\n"
            "2. **Common fixes**:\n"
            "   - Lower your similarity threshold (try 0.7 instead of 0.8)\n"
            "   - Experiment with chunk sizes (512 tokens often works well)\n"
            "   - Try a different embedding model\n\n"
            "3. **Track experiments**: Log retrieval params to compare:\n"
            "```python\n"
            "mlflow.log_params({'chunk_size': 512, 'threshold': 0.7})\n"
            "```"
        ),
        expected_response=(
            "Check RETRIEVER span in traces for scores. Try adjusting similarity threshold, "
            "chunk size, or embedding model. Log params to compare experiments."
        ),
        trace_type="session",
        session_id="session-rag-debug",
        session_user="bob@example.com",
        turn_index=2,
    ),
    DemoTrace(
        query="How do I compare different retrieval configurations systematically?",
        v1_response=(
            "You can run multiple experiments with different settings and compare them "
            "in the MLflow UI."
        ),
        v2_response=(
            "Use MLflow's evaluation framework for systematic comparison:\n\n"
            "```python\n"
            "import mlflow\n"
            "from mlflow.metrics.genai import relevance, faithfulness\n\n"
            "# Create evaluation dataset\n"
            "eval_data = pd.DataFrame({\n"
            "    'inputs': questions,\n"
            "    'ground_truth': expected_answers\n"
            "})\n\n"
            "# Run evaluation\n"
            "results = mlflow.evaluate(\n"
            "    model=your_rag_pipeline,\n"
            "    data=eval_data,\n"
            "    scorers=[relevance(), faithfulness()]\n"
            ")\n"
            "```\n\n"
            "Compare runs in the MLflow UI's comparison view to see which config wins!"
        ),
        expected_response=(
            "Use mlflow.evaluate() with relevance() and faithfulness() scorers. "
            "Create eval dataset with inputs and ground_truth, then compare runs in UI."
        ),
        trace_type="session",
        session_id="session-rag-debug",
        session_user="bob@example.com",
        turn_index=3,
    ),
    # Session 3: Model Deployment (2 turns)
    DemoTrace(
        query="I have a model logged in MLflow. How do I deploy it as an API?",
        v1_response=(
            "You can use MLflow's model serving. Register the model and then serve it "
            "with the mlflow models serve command."
        ),
        v2_response=(
            "Great! Here's the deployment path:\n\n"
            "1. **Register the model** (if not already):\n"
            "```python\n"
            "mlflow.register_model(\n"
            "    'runs:/<run_id>/model',\n"
            "    'my-classifier'\n"
            ")\n"
            "```\n\n"
            "2. **Serve it locally**:\n"
            "```bash\n"
            "mlflow models serve -m models:/my-classifier/1 -p 5001\n"
            "```\n\n"
            "3. **Test the endpoint**:\n"
            "```bash\n"
            "curl -X POST http://localhost:5001/invocations \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"inputs\": [[1, 2, 3, 4]]}'\n"
            "```\n\n"
            "Want to deploy to Kubernetes or a cloud platform instead?"
        ),
        expected_response=(
            "Register with mlflow.register_model(), then serve with "
            "`mlflow models serve -m models:/name/version`. Test with curl POST to /invocations."
        ),
        trace_type="session",
        session_id="session-deployment",
        session_user="carol@example.com",
        turn_index=1,
    ),
    DemoTrace(
        query="Yes, I want to deploy to Kubernetes. What's the process?",
        v1_response=(
            "You can build a Docker image with MLflow and deploy it to Kubernetes using "
            "standard kubectl commands."
        ),
        v2_response=(
            "Here's the K8s deployment workflow:\n\n"
            "1. **Build a Docker image**:\n"
            "```bash\n"
            "mlflow models build-docker \\\n"
            "  -m models:/my-classifier/1 \\\n"
            "  -n my-classifier:v1\n"
            "```\n\n"
            "2. **Push to your registry**:\n"
            "```bash\n"
            "docker push your-registry/my-classifier:v1\n"
            "```\n\n"
            "3. **Deploy to K8s** (example deployment.yaml):\n"
            "```yaml\n"
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "spec:\n"
            "  containers:\n"
            "  - name: model\n"
            "    image: your-registry/my-classifier:v1\n"
            "    ports:\n"
            "    - containerPort: 8080\n"
            "```\n\n"
            "The container exposes a `/invocations` endpoint compatible with MLflow's format."
        ),
        expected_response=(
            "Build image with `mlflow models build-docker`, push to registry, "
            "deploy with K8s manifests. Container exposes /invocations endpoint."
        ),
        trace_type="session",
        session_id="session-deployment",
        session_user="carol@example.com",
        turn_index=2,
    ),
]

# =============================================================================
# Combined Trace Data
# =============================================================================

ALL_DEMO_TRACES: list[DemoTrace] = RAG_TRACES + AGENT_TRACES + PROMPT_TRACES + SESSION_TRACES

# Mapping of queries (lowercased) to expected responses for evaluation
EXPECTED_ANSWERS: dict[str, str] = {
    trace.query.lower(): trace.expected_response for trace in ALL_DEMO_TRACES
}
