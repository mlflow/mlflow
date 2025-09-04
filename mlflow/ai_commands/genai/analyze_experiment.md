---
namespace: genai
description: Analyzes the traces logged in an MLflow experiment to find operational and quality issues automatically, generating a markdown report.
---

# Analyze Experiment

Analyzes traces in an MLflow experiment for quality issues, performance problems, and patterns.

## Step 1: Setup and Configuration

### 1.1 Collect Experiment Information

- **REQUIRED FIRST**: Ask user "How do you want to authenticate to MLflow?"

  **Option 1: Local/Self-hosted MLflow**

  - Ask for tracking URI (one of):
    - SQLite: `sqlite:////path/to/mlflow.db`
    - PostgreSQL: `postgresql://user:password@host:port/database`
    - MySQL: `mysql://user:password@host:port/database`
    - File Store: `file:///path/to/mlruns` or just `/path/to/mlruns`
  - Ask user to create an environment file (e.g., `mlflow.env`) containing:
    ```
    MLFLOW_TRACKING_URI=<provided_uri>
    ```

  **Option 2: Databricks**

  - Ask which authentication method:
    - **PAT Auth**: Request `DATABRICKS_HOST` and `DATABRICKS_TOKEN`
    - **Profile Auth**: Request `DATABRICKS_CONFIG_PROFILE` name
  - Ask user to create an environment file (e.g., `mlflow.env`) containing:

    ```
    # For PAT Auth:
    MLFLOW_TRACKING_URI=databricks
    DATABRICKS_HOST=<provided_host>
    DATABRICKS_TOKEN=<provided_token>

    # OR for Profile Auth:
    MLFLOW_TRACKING_URI=databricks
    DATABRICKS_CONFIG_PROFILE=<provided_profile>
    ```

  **Option 3: Environment Variables Already Set**

  - Ask user "Do you already have MLflow environment variables set in your shell (bashrc/zshrc)?"
  - If yes, test connection directly: `uv run python -m mlflow experiments search --max-results 10`
  - If this works, skip env file creation and use commands without `--env-file` flag
  - If not, fall back to Options 1 or 2

- Ask user for the path to their environment file (if using Options 1-2)
- Verify connection by listing experiments: `uv run --env-file <env_file_path> python -m mlflow experiments search --max-results 10`
- **Option to search by name**: If user knows the experiment name, use `--filter-string` parameter:
  - `uv run --env-file <env_file_path> python -m mlflow experiments search --filter-string "name LIKE '%experiment_name%'" --max-results 10`
- Ask user for experiment ID or let them choose from the list
- **WAIT for user response** - do not continue until they provide the experiment ID
- Add `MLFLOW_EXPERIMENT_ID=<experiment_id>` to their environment file
- Run `uv run --env-file <env_file_path> python -m mlflow traces --help` to understand the CLI commands and options

### 1.2 Test Trace Retrieval

- Call `uv run --env-file <env_file_path> python -m mlflow traces search --max-results 5` to verify:
  - Traces exist in the experiment
  - CLI is working properly (using local MLflow installation)
  - Database connection is valid
- Extract sample trace IDs for testing
- Get one full trace with `uv run --env-file <env_file_path> python -m mlflow traces get --trace-id <id>` to understand the data structure

## Step 2: Analysis Phase

### 2.1 Bulk Trace Collection

- Search for a larger sample using `--max-results` parameter (start with 20-50 traces for initial analysis)
- **IMPORTANT**: Use `--max-results` to limit results for users with hundreds of thousands of experiments/traces
- Extract key fields: trace_id, state, execution_duration_ms, request_preview, response_preview

### 2.1.5 Understand Agent Purpose and Capabilities

- Analyze trace inputs/outputs to understand the agent's task:
  - Extract trace inputs/outputs: `--extract-fields info.trace_metadata.\`mlflow.traceInputs\`,info.trace_metadata.\`mlflow.traceOutputs\``
  - Examine these fields to understand:
    - Types of questions users ask
    - Types of responses the agent provides
    - Common patterns in user interactions
  - Identify available tools by examining spans with type "TOOL":
    - What tools are available to the agent?
    - What data sources can the agent access?
    - What capabilities do these tools provide?
- Generate a 1-paragraph agent description covering:
  - **What the agent's job is** (e.g., "a boating agent that answers questions about weather and helps users plan trips")
  - **What data sources it has access to** (APIs, databases, etc.)
- **Present this description to the user** and ask for confirmation/corrections
- **WAIT for user response** - do not proceed until they confirm or provide corrections
- **Ask if they want to focus the analysis on anything specific** (or do a general report)
  - If they provide specific focus areas, use these as additional context for hypothesis formation
  - Don't overfit to their focus - still do comprehensive analysis, but prioritize their areas of interest
  - Their specific concerns should become hypotheses to validate/invalidate during analysis
- **WAIT for user response** before proceeding to section 2.2
- Use agent context + any specific focus areas for all subsequent hypothesis testing in sections 2.2+

### 2.2 Operational Issues Analysis (Hypothesis-Driven Approach)

**NOTE: Use MLflow CLI commands for trace exploration - DO NOT use inline Python scripts during this phase**

**Show your thinking as you go**: Always explain your hypothesis development process including:

- Current hypothesis being tested
- Evidence found: ALWAYS show BOTH trace input (user request) AND trace output (agent response), plus tools called
- Reasoning for supporting/refuting the hypothesis

Process traces in batches of 10, building and refining hypotheses with each batch:

1. Form initial hypotheses from first batch
2. With each new batch: validate, refute, or expand hypotheses
3. Continue until patterns stabilize

**After confirming ANY hypothesis (operational or quality)**: Track assessments for inclusion in final report:

- **1:1 Correspondence**: Each assessment must correspond to ONE specific issue/hypothesis
- Use snake_case names as assessment keys (e.g., `overly_verbose`, `tool_failure`, `rate_limited`, `slow_response`)
- Track which traces exhibit each issue with detailed rationales
- Document specifics like:

  - For quality issues: exact character counts, repetition counts, unnecessary sections
  - For operational issues: exact durations, error messages, timeout values

- **Error Analysis**

  - Filter for ERROR traces: `uv run --env-file <env_file_path> python -m mlflow traces search --filter "info.state = 'ERROR'" --max-results 10`
  - **Adjust --max-results as needed**: Start with 10-20, increase if you need more examples to identify patterns
  - **Pattern Analysis Focus**: Identify WHY errors occur by examining:
    - Tool/API failures in spans (look for spans with type "TOOL" that failed)
    - Rate limiting responses from external APIs
    - Authentication/permission errors
    - Timeout patterns (compare execution_duration_ms)
    - Input validation failures
    - Resource unavailability (databases, services down)
  - Example hypotheses to test:
    - Certain types of queries consistently trigger tool failures
    - Errors cluster around specific time ranges (service outages)
    - Fast failures (~2s) indicate input validation vs slower failures (~30s) indicate timeouts
    - Specific tools/APIs are unreliable and cause cascading failures
    - Rate limiting from external services causes batch failures
  - **Note**: You may discover other operational error patterns as you analyze the traces

- **Performance Problems (High Latency Analysis)**
  - Filter for OK traces with high latency: `uv run --env-file <env_file_path> python -m mlflow traces search --filter "info.state = 'OK'" --max-results 10`
  - **Adjust --max-results as needed**: Start with 10-20, increase if you need more examples to identify patterns
  - **Pattern Analysis Focus**: Identify WHY traces are slow by examining:
    - Tool call duration patterns in spans
    - Number of sequential vs parallel tool calls
    - Specific slow APIs/tools (database queries, web requests, etc.)
    - Cold start vs warm execution patterns
    - Resource contention indicators
  - Example hypotheses to test:
    - Complex queries with multiple sequential tool calls have multiplicative latency
    - Certain tools/APIs are consistent performance bottlenecks (>5s per call)
    - First queries in sessions are slower due to cold start overhead
    - Database queries without proper indexing cause delays
    - Network timeouts or retries inflate execution time
    - Parallel tool execution is not properly implemented
  - **Note**: You may discover other performance patterns as you analyze the traces

### 2.3 Quality Issues Analysis (Hypothesis-Driven Approach)

**NOTE: Use MLflow CLI commands for trace exploration - DO NOT use inline Python scripts during this phase**

Focus on response quality, not operational performance:

- **Content Quality Issues**
  - Sample both OK and ERROR traces
  - Example hypotheses to test:
    - Agent provides overly verbose responses for simple questions
    - Some text/information is repeated unnecessarily across responses
    - Conversation context carries over inappropriately
    - Agent asks follow-up questions instead of attempting tasks
    - Responses are inconsistent for similar queries
    - Agent provides incorrect or outdated information
    - Response format is inappropriate for the query type
  - **Note**: You may discover other quality issues as you analyze the traces

### 2.4 Strengths and Successes Analysis (Hypothesis-Driven Approach)

**NOTE: Use MLflow CLI commands for trace exploration - DO NOT use inline Python scripts during this phase**

Process successful traces to identify what's working well:

- **Successful Interactions**

  - Filter for OK traces with good outcomes
  - Example hypotheses to test:
    - Agent provides comprehensive, helpful responses for complex queries
    - Certain types of questions consistently get high-quality answers
    - Tool usage is appropriate and effective for specific scenarios
    - Response format is well-structured for particular use cases

- **Effective Tool Usage**

  - Examine traces where tools are used successfully
  - Example hypotheses to test:
    - Agent selects appropriate tools for different query types
    - Multi-step tool usage produces better outcomes
    - Certain tool combinations work particularly well together

- **Quality Responses**
  - Identify traces with good response quality
  - Example hypotheses to test:
    - Agent provides right level of detail for complex questions
    - Safety/important information is appropriately included
    - Agent successfully handles follow-up questions in context

### 2.5 Generate Final Report

- Ask user where to save the report (markdown file path, e.g., `experiment_analysis.md`)
- **ONLY NOW use uv inline Python scripts for statistical calculations** - never compute stats manually
- Inline Python scripts are ONLY for final math/statistics, NOT for trace exploration
- Use `uv run --env-file <env_file_path> python -c "..."` for any Python calculations that need MLflow access
- Generate a single comprehensive markdown report with:
  - **Summary statistics** (computed via `uv run --env-file <env_file_path> python -c "..."` with collected trace data):
    - Total traces analyzed
    - Success rate (OK vs ERROR percentage)
    - Average, median, p95 latency for successful traces
    - Error rate distribution by duration (fast fails vs timeouts)
  - **Operational Issues** (errors, latency, performance):
    - For each confirmed operational hypothesis:
      - Clear statement of the hypothesis
      - Example trace IDs that support the hypothesis
      - BOTH trace input (user request) AND trace output (agent response) excerpts from those traces
      - Tools called (spans of type "TOOL") and their durations/failures
      - Root cause analysis: WHY the issue occurs (rate limiting, API failures, timeouts, etc.)
      - **Trace assessments**: List specific trace IDs that exhibit this issue with detailed rationales explaining why each trace demonstrates the pattern
      - Quantitative evidence (frequency, timing patterns, etc.) - computed via Python
  - **Quality Issues** (content problems, user experience):
    - For each confirmed quality hypothesis:
      - Clear statement of the hypothesis
      - Example trace IDs that support the hypothesis
      - BOTH trace input (user request) AND trace output (agent response) excerpts from those traces
      - **Trace assessments**: List specific trace IDs that exhibit this issue with detailed rationales explaining why each trace demonstrates the pattern
      - Quantitative evidence (frequency, assessment patterns, etc.) - computed via Python
  - **Refuted Hypotheses** (briefly noted)
  - Recommendations for improvement based on confirmed issues
