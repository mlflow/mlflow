# CHANGELOG

## 3.10.0 (2026-02-20)

MLflow 3.10.0 includes several major features and improvements

Features:

- [UI] Add sliding animation to workflow switch component (#20831, @daniellok-db)
- [Tracing] Display cached tokens in trace UI (#20957, @TomeHirata)
- [Evaluation] Move select traces button to be next to Run judge (#20992, @PattaraS)
- [] Distributed tracing for gateway endpoints (#20864, @TomeHirata)
- [] Add user selector in the gateway usage page (#20944, @TomeHirata)
- [Docs] [MLflow Demo] Docs for GenAI Demo (#20240, @BenWilson2)
- [UI] Move Getting Started above experiments list and make collapsible (#20691, @B-Step62)
- [Model Registry / Tracking] Add mlflow `migrate-filestore` command (#20615, @harupy)
- [UI] Add visual indicator for demo experiment in experiment list (#20787, @B-Step62)
- [Scoring] Enable parquet content_type in the scoring server input for pyfunc (#20630, @TFK1410)
- [UI] feat(ui): Add workspace landing page, multi-workspace support, and qu… (#20702, @Gkrumbach07)
- [Tracking] Merge workspace feature branch into master (#20657, @B-Step62)
- [] Add Gateway Usage Page  (#20642, @TomeHirata)
- [] Add usage section in endpoint page (#20357, @TomeHirata)
- [UI] [ MLflow Demo ] UI updates for MLflow Demo interfaces (#20162, @BenWilson2)
- [Build] Support comma-separated rules in `# clint: disable=` comments (#20651, @copilot-swe-agent)
- [Build / Docs / Models / Projects / Scoring] Replace `virtualenv` with `python -m venv` in virtualenv env_manager path (#20640, @copilot-swe-agent)
- [Tracing] Add per-decorator `sampling_ratio_override` parameter to `@mlflow.trace` (#19784, @harupy)
- [Evaluation / Tracking] Add `mlflow datasets list` CLI command (#20167, @alkispoly-db)
- [] Add trace ingestion for Gateway endpoints (#20358, @TomeHirata)
- [Tracing] feat(typescript-anthropic): add streaming support (#20384, @rollyjoel)
- [Evaluation] Add delete dataset records API (#19690, @joelrobin18)
- [] Add tooltip link to navigate to traces tab with time range filter (#20466, @serena-ruan)
- [Tracking] [MLflow Demo] Add mlflow demo cli command (#20048, @BenWilson2)
- [Evaluation] Add an SDK for distillation from conversation to goal/persona (#20289, @smoorjani)
- [Tracing] Livekit Agents Integration in Mlflow (#20439, @joelrobin18)
- [Tracing / UI] Enable running scorers/judges from trace details drawer in UI (#20518, @danielseong1)
- [] link gateway and experiment (#20356, @TomeHirata)
- [Prompts] Add optimization backend APIs to auth control (#20392, @chenmoneygithub)
- [Tracing] Add an SDK for search sessions to get complete sessions (#20288, @smoorjani)
- [Tracing] Reasoning in Chat UI Mistral + Chat UI  (#19636, @joelrobin18)
- [Evaluation] Add TruLens third-party scorer integration (#19492, @debu-sinha)
- [Evaluation / Tracing] Add Guardrails AI scorer integration (#20038, @debu-sinha)
- [Tracking] [MLflow Demo] Add Prompt demo data (#20047, @BenWilson2)
- [Tracking] [MLflow Demo] Add Eval simulation data (#20046, @BenWilson2)
- [Tracking] [MLflow Demo] Add trace data for demo (#19995, @BenWilson2)
- [Tracking] Support get_dataset(name=...) in OSS environments (#20423, @alkispoly-db)
- [UI] Add session comparison UI with goal/persona matching (#20377, @smoorjani)
- [] [UI] Model and cost rendering for spans (#20330, @serena-ruan)
- [] [1/x] Support span model extraction and cost calculation (#20327, @serena-ruan)
- [Evaluation] Make conversation simulator public and easily subclassable (#20243, @smoorjani)
- [Prompts] Add progress tracking for prompt optimization job (#20374, @chenmoneygithub)
- [Prompts] Prompt Optimization backend PR 3: Add Get, Search, and Delete prompt optimization job APIs (#20197, @chenmoneygithub)
- [Prompts] Track intermediate candidates and evaluation scores in gepa optimizer (#20198, @chenmoneygithub)
- [Tracking] [MLflow Demo] Base implementation for demo framework (#19994, @BenWilson2)
- [Prompts] Prompt Optimization backend PR 2: Add CreatePromptOptimizationJob and CancelPromptOptimizationJob (#20115, @chenmoneygithub)
- [Tracing] Support shift+select for Traces (#20125, @B-Step62)
- [UI] Ml61127/remove experiment type selector inside experiment page (#20161, @ispoljari)
- [UI] Ml61126/remove nested sidebars within gateway and experiments tab (#20160, @ispoljari)
- [UI] [ML-61124]: add selector for workflow type in top level navbar (#20158, @ispoljari)
- [Prompts / UI] Feat/render md in prompt registry (#19615, @iyashk)
- [Prompts] [Prompt Optimization Backend PR #1] Wrap prompt optimize in mlflow job (#20001, @chenmoneygithub)
- [Tracking] Add --experiment-name option to mlflow experiments get command (#19929, @alkispoly-db)

Bug fixes:

- [Tracing / UI] Fix infinite fetch loop in trace detail view when num_spans metadata mismatches (#20596, @coldzero94)
- [UI] fix:implement dark mode in experiment correctly (#20974, @intelliking)
- [Evaluation] Fix 'Select traces' do not show new traces in Judge UI (#20991, @PattaraS)
- [Tracing / Tracking] Fix RecursionError in strands, semantic_kernel, and haystack autologgers with shared tracer provider (#20809, @cgrierson-smartsheet)
- [Tracking] fix(tracking): Fix IntegrityError in log_batch when duplicate metrics span multiple key batches (#20807, @aws-khatria)
- [Tracing] Support native tool calls in CrewAI 1.9.0+ autolog tests (#20742, @TomeHirata)
- [Evaluation] Fix retrieval_relevance assessments logged to wrong span with missing chunk index (#20998, @smoorjani)
- [Evaluation] Fix missing session metadata on failed session-level scorer assessments (#20988, @smoorjani)
- [Tracking] Enhance path validation in check_tarfile_security for windows (#20924, @TomeHirata)
- [Docs] Fix admonition link underlines not rendering (#20990, @copilot-swe-agent)
- [Tracking] Rebuild `SearchTraces` V2 request body on `ENDPOINT_NOT_FOUND` fallback (#20963, @brendanmaguire)
- [Build] Add model version search filtering based on user permissions (#20964, @TomeHirata)
- [Tracing] Display notebook trace viewer when workspace is on (#20947, @TomeHirata)
- [] Add `MLFLOW_GATEWAY_RESOLVE_API_KEY_FROM_FILE` flag to prevent local file inclusion in API gateway (#20965, @TomeHirata)
- [Tracking] Fix Claude Agent SDK tracing by capturing messages from receive_messages (#20778, @smoorjani)
- [Build / Tracking] Add missing authentication for fastapi routes (#20920, @TomeHirata)
- [Evaluation] Fix guardrails scorer compatibility with guardrails-ai 0.9.0 (#20934, @smoorjani)
- [UI] Fix duplicated title and add icons to Experiments/Prompts page headers (#20813, @B-Step62)
- [Tracing] Trace UI papercut: highlight searched text and change search box hint's wording. (#20841, @PattaraS)
- [Prompts] Fix arbitrary file read via prompt tag validation bypass in Model Registry (#20833, @TomeHirata)
- [Tracking] Fix `RestException` crash on null `error_code` and incorrect except clause (#20903, @copilot-swe-agent)
- [UI] Fix Disable action button in Traces Tab (#20883, @joelrobin18)
- [UI] Fix experiment rename modal not refreshing experiment details (#20882, @joelrobin18)
- [Build] Skip workspace header when workspace is disabled (#20904, @TomeHirata)
- [] Block CORS for ajax paths (#20832, @TomeHirata)
- [UI] [UI] Improve empty states across Experiments, Models, Prompts, and Gateway pages (#20044, @ridgupta26)
- [UI] UI: Improve empty states for Traces and Sessions tabs (#20034, @ridgupta26)
- [Build] Validate webhook url to fix SSRF vulnerability (#20747, @TomeHirata)
- [Scoring / Tracing] Fix TypeError in online scoring config endpoint when basic-auth is enabled (#20783, @copilot-swe-agent)
- [Tracing] Fix `experiment_id` type error in gateway config resolver (#20764, @copilot-swe-agent)
- [UI] Fix docs link to respect workflow type (GenAI vs ML) (#20752, @copilot-swe-agent)
- [Tracking] Fix: Do not emit pickle warning when user calls `mlflow.pyfunc.log_model` with `loader_module` param (#20727, @WeichenXu123)
- [Tracing] Change cache config to prevent seach bounce (#20688, @PattaraS)
- [Evaluation] Fix multiple align() calls on MemoryAugmentedJudge (#20708, @smoorjani)
- [Evaluation] Batch embedding calls for Databricks endpoints to avoid size limit errors (#20685, @smoorjani)
- [Evaluation] Fix the UI for MemAlign-ed scorers (#20632, @smoorjani)
- [Tracing] Fix type hints lost with @mlflow.trace decorator (#20648, @veeceey)
- [Evaluation] Use JSONAdapter for best-effort structured outputs in MemAlign predictions (#20679, @smoorjani)
- [Tracking] Fix `mlflow demo` URL to use experiment ID instead of name (#20678, @copilot-swe-agent)
- [Tracking] Fix circular import in FileStore caused by PromptVersion import (#20677, @copilot-swe-agent)
- [] Fix error handling for streaming request (#20610, @TomeHirata)
- [Models] Fix warning message: add space and documentation link for pickle security (#20656, @copilot-swe-agent)
- [Evaluation] Fix SHAP compatibility for shap >= 0.47 (#20623, @copilot-swe-agent)
- [Prompts] Fix the deadlock between run linking and trace linking (#20620, @TomeHirata)
- [Tracking] Fix FTP artifact path handling on Windows with Python 3.11+ (#20622, @copilot-swe-agent)
- [Evaluation] Fix failed judge call error propagation (#20601, @AveshCSingh)
- [Tracking] Fix off-by-one error in `_validate_max_retries` and `_validate_backoff_factor` (#20597, @vb-dbrks)
- [Prompts] Fix bug: linking prompt to experiments does not work for default experiments (#20588, @PattaraS)
- [Build] Fix Docker full image tags not being published for versioned releases (#20589, @copilot-swe-agent)
- [Prompts] Implement locking mechanism to prevent race conditions during prompt linking (#20586, @TomeHirata)
- [Prompts] Revert "Fix bug: linking prompt to experiments does not work for defa… (#20585, @PattaraS)
- [Prompts] Fix bug: linking prompt to experiments does not work for default experiments (#20562, @PattaraS)
- [Model Registry] Fix N+1 query issue in search_registered_models (#20493, @Karim-siala)
- [Tracking] Fix optimistic pagination in SQLAlchemy store `_search_runs` and handle `max_results=None` (#20547, @copilot-swe-agent)
- [UI] Add cancel button for LLM judge evaluations in trace details drawer (#20519, @danielseong1)
- [UI] Fix incorrect 'Trace level' label in session judges modal (#20520, @danielseong1)
- [Tracing] fix: allow overriding notebook trace iframe base URL (#20485, @TatsuyaHayashino)
- [Prompts] Include the prompt model config in the optimized prompt (#20431, @chenmoneygithub)
- [Tracing / UI] Fix Anthropic trace UI rendering for tool_result with image content (#20190, @joncarter1)
- [Tracking] Enforce authorization on AJAX proxy artifact APIs (#20035, @mprahl)
- [Tracking] Ensure server-provided artifact root is reused on MLflowClient calls (#19336, @mprahl)
- [UI] Fix trace selection not registering in SelectTracesModal (#20099, @joelrobin18)

Documentation updates:

- [Docs] Add documentation for `mlflow migrate-filestore` command (#20616, @harupy)
- [Docs] Document X-MLFLOW-WORKSPACE header for AI Gateway endpoints with workspace fallback behavior (#20984, @copilot-swe-agent)
- [Docs] Fix outdated server-features references to server-info (#20948, @copilot-swe-agent)
- [Docs / Tracing] Remove span attributes filtering from search traces documentation (#20858, @copilot-swe-agent)
- [Docs] Add Modal as a supported deployment target with full documentation (#20032, @debu-sinha)
- [Docs] Add gateway usage tracking doc page (#20748, @TomeHirata)
- [Docs / Evaluation] Fix MemAlign bug bash issues (#20712, @veronicalyu320)
- [] Fix docs: trace spans are stored in database, not artifact storage (#20668, @B-Step62)
- [Prompts] Change header level for "Automatic Prompt Linking" section in `use-prompts-in-apps.mdx`  (#20661, @PattaraS)
- [Docs] Multi-turn evaluation launch documentation (#20443, @smoorjani)
- [Prompts] Update `use-prompts-in-apps.mdx` with a section for prompt linking under traced method (#20593, @PattaraS)
- [Docs] docs: Add missing targets arg in huggingface dataset docs (#20637, @KarelZe)
- [Build] Display rule names instead of IDs in clint error output (#20592, @copilot-swe-agent)
- [Docs] Detailed guide for setting up SSO with mlflow-oidc-auth plugin (#20556, @WeichenXu123)
- [Prompts] Mark prompt registry APIs as stable. (#20507, @PattaraS)
- [Docs] code-based scorer examples (#20407, @SomtochiUmeh)
- [Docs] Custom judges section (#20393, @SomtochiUmeh)
- [Docs] (mostly) copy over eval datasets article from managed docs (#19787, @achen530)
- [Docs] Add the RAG built-in judges section (#20369, @SomtochiUmeh)
- [Docs] Fix `ToolAgent` name formatting in ag2 documentation and examples (#20470, @Umakanth555)
- [Docs] Add collection_name parameter to CrewAI knowledge configuration in docs and example (#20469, @Umakanth555)
- [Docs] Update index and predefined judges pages (#20368, @SomtochiUmeh)
- [Docs] docs: Clarify -full Docker image availability from v3.9.0 onwards (#20223, @copilot-swe-agent)
- [Docs] Generalize Knowledge Cutoff Note in CLAUDE.md beyond model names (#20165, @copilot-swe-agent)

Small bug fixes and documentation updates:

#20959, #20915, #20986, #20956, #20912, #20955, #20943, #20919, #20776, #20826, #20781, #20767, #20761, #20760, #20763, #20762, #20687, #20746, #20682, #20667, #20658, #20578, #20559, #20495, #20497, @TomeHirata; #21006, #20980, #20707, #20777, @bbqiu; #20950, #21008, #20877, #20822, #20817, #20813, #20816, #20796, #20815, #20765, #20716, #20689, #20744, #20690, #20451, #20502, #20252, #20314, #20210, @B-Step62; #21000, #20975, #20806, #20449, #20686, #20603, #20573, #20572, #20584, #20551, #20526, #20550, #20523, #20525, #20453, #20478, #20452, #20438, #20474, #20460, #20457, #20459, #20456, #20444, #20418, #20285, #20284, #20283, #20282, #20281, #20280, #20051, @smoorjani; #21005, #21007, #20880, #20857, #20802, #20779, #20717, #20713, #20714, #20692, #20693, #20683, #20675, #20665, #20674, #20673, #20663, #20662, #20659, #20652, #20649, #20650, #20647, #20646, #20641, #20638, #20635, #20634, #20633, #20626, #20625, #20621, #20619, #20618, #20617, #20606, #20564, #20581, #20570, #20568, #20566, #20558, #20560, #20543, #20554, #20537, #20536, #20532, #20530, #20528, #20512, #20505, #20501, #20498, #20496, #20491, #20490, #20489, #20487, #20486, #20484, #20483, #20482, #20441, #20436, #20427, #20417, #20400, #20399, #20397, #20395, #20396, #20391, #20342, #20341, #20332, #20326, #20316, #20315, #20305, #20300, #20299, #20297, #20293, #20268, #20262, #20260, #20251, #20250, #20244, #20235, #20228, #20227, #20226, #20220, #20202, #20186, #20172, #20152, #20150, #19984, #20102, #20098, #20095, #20093, #20094, #20091, #20090, #20089, #20088, #20087, #20086, #20085, #20084, #20083, #20082, #20081, #20080, #20077, #20076, #20075, #20070, #20067, #20069, #20020, #20026, @copilot-swe-agent; #20793, #20791, #20768, @WeichenXu123; #20979, #20701, #20609, #20608, #20569, #20535, #20481, #20318, #20224, #20149, #20119, #20068, #20014, #20016, #20019, @harupy; #20973, @Gkrumbach07; #21003, #20936, #20730, #20041, #20381, @xsh310; #20989, #20830, #20766, #20759, #20758, #20757, #20756, #20699, #20697, #20696, #20695, #20694, #20255, #20254, #20253, #20248, #20247, #20010, #20009, #19999, #19998, #19976, #19975, #19974, #19973, #19971, @daniellok-db; #20976, @aravind-segu; #20725, #20339, #20565, #20660, #20455, #20440, #20404, #20403, #20402, #20567, #20542, #20541, #20540, #20557, #20503, #20506, #20500, #20499, #20467, #20338, #20337, #20331, #20462, #20329, #20328, #20323, @serena-ruan; #20737, @jamesbxwu; #20862, #20861, @PattaraS; #20805, #20705, #20373, @mprahl; #20773, @etirelli; #20753, @etscript; #20629, #19758, @justinwei-db; #20711, @kevin-lyn; #20576, @nisha2003; #20553, #20521, @danielseong1; #20548, @bartosz-grabowski; #20504, @smivv; #20527, @BenWilson2; #20363, #20364, @rollyjoel; #20494, @dbczumar; #20360, #20340, #20313, #20312, #20276, #20275, #20261, #20233, #19484, @hubertzub-db; #20359, @LiberiFatali; #20386, @chenmoneygithub; #20159, @ispoljari

## 3.10.0rc0 (2026-02-11)

We're excited to announce MLflow 3.10.0rc0, which includes several notable updates:

**Major New Features**:

- 🏢 **Organization Support in MLflow Tracking Server**: MLflow now supports multi-workspace environments! You can organize your experiments and resources across different workspaces with a new landing page that lets you navigate between them seamlessly. (#20702, #20657, @Gkrumbach07, @B-Step62)
- 💬 **Multi-turn Conversation Simulation**: Building on the conversation simulator introduced in 3.9, we've made it fully public and easily subclassable. You can now create custom simulation scenarios, compare sessions with goal/persona matching, and distill conversations into reusable goal/persona pairs for comprehensive agent testing. (#20243, #20377, #20289, @smoorjani)
- 💰 **Trace Cost Tracking**: Gain visibility into your LLM spending! MLflow now automatically extracts model information from LLM spans and calculates costs, with a new UI that renders model and cost data directly in your trace views. (#20327, #20330, @serena-ruan)
- 🎯 **Top-level GenAI/Classical ML Split**: We've redesigned the navigation to provide a frictionless experience. A new workflow type selector in the top-level navbar lets you quickly switch between GenAI and Classical ML contexts, with streamlined sidebars that reduce visual clutter. (#20158, #20160, #20161, #20699, @ispoljari, @daniellok-db)
- 🎮 **MLflow Demo Experiment**: Get started with MLflow faster than ever! The new `mlflow demo` CLI command generates a fully-populated demo environment with sample traces, prompts, and evaluation data so you can explore MLflow's features hands-on without any setup. (#19994, #19995, #20046, #20047, #20048, #20162, @BenWilson2)
- 📊 **Gateway Usage Tracking**: Monitor your AI Gateway endpoints with detailed usage analytics. A new usage page shows request patterns and metrics, with trace ingestion that links gateway calls back to your experiments for end-to-end observability. (#20357, #20358, #20642, @TomeHirata)

Stay tuned for the full release, which will be packed with even more features and bugfixes.

To try out this release candidate, please run:

`pip install mlflow==3.10.0rc0`

## 3.9.0 (2026-01-28)

We're excited to announce MLflow 3.9.0, which includes several notable updates:

**Major New Features**:

- 🔮 **MLflow Assistant**: Figuring out the next steps to debug your apps and agents can be challenging. We're excited to introduce the MLflow Assistant, an in-product chatbot that can help you identify, diagnose, and fix issues. The assistant is backed by Claude Code, and directly passes context from the MLflow UI to Claude. Click on the floating "Assistant" button in the bottom right of the MLflow UI to get started!
- 📈 **Trace Overview Dashboard**: You can now get insights into your agent's performance at a glance with the new "Overview" tab in GenAI experiments. Many pre-built statistics are available out of the box, including performance metrics (e.g. latency, request count), quality metrics (based on assessments), and tool call summaries. If there are any additional charts you'd like to see, please feel free to raise an issue in the MLflow repository!
- ✨ **AI Gateway**: We're revamping our AI Gateway feature! AI Gateway provides a unified interface for your API requests, allowing you to route queries to your LLM provider(s) of choice. In MLflow 3.9.0, the Gateway server is now located directly in the tracking server, so you don't need to spin up a new process. Additional features such as passthrough endpoints, traffic splits, and fallback models are also available, with more to come soon! For more detailed information, please take a look at the [docs](https://mlflow.org/docs/latest/genai/governance/ai-gateway/).
- 🔎 **Online Monitoring with LLM Judges**: Configure LLM judges to automatically run on your traces, without having to write a line of code! You can either use one of our [pre-defined judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/), or provide your own prompt and instructions to create custom metrics. Head to the new "Judges" tab within the GenAI Experiment UI to get started.
- 🤖 **Judge Builder UI**: Define and iterate on custom LLM judge prompts directly from the UI! Within the new "Judges" tab, you can create your own prompt for an LLM judge, and test-run it on your traces to see what the output would be. Once you're happy with it, you can either use it for online monitoring (as mentioned above), or use it via the Python SDK for your evals.
- 🔗 **Distributed Tracing**: Trace context can now be propagated across different services and processes, allowing you to truly track request lifecycles from end to end. The related APIs are defined in the `mlflow.tracing.distributed` module (with more documentation to come soon).
- 📚 **MemAlign - a new judge optimizer algorithm**: We're excited to introduce `MemAlignOptimizer`, a new algorithm that makes your judges smarter over time. It learns general guidelines from past feedback while dynamically retrieving relevant examples at runtime, giving you more accurate evaluations.

Features:

- [Gateway] Add LiteLLM provider to support many other providers (#19394, @TomeHirata)
- [Gateway] Add passthrough support for Anthropic Messages API (#19423, @TomeHirata)
- [Gateway] Add passthrough support for Gemini `generateContent` and `streamGenerateContent` APIs (#19425, @TomeHirata)
- [Gateway] Add routing strategy and fallback configuration support for gateway endpoints (#19483, @TomeHirata)
- [Gateway] Deprecate Unity Catalog function integration in AI Gateway (#19457, @harupy)
- [Gateway / UI] Create List API Keys landing page (#19441, @BenWilson2)
- [Gateway / UI] Add Create API Keys functionality (#19442, @BenWilson2)
- [Gateway / UI] Add delete and update capabilities for API Keys (#19446, @BenWilson2)
- [Gateway / UI] Add endpoint listing page and tab layout (#19474, @BenWilson2)
- [Gateway / UI] Add Create endpoint page and enhance provider select (#19475, @BenWilson2)
- [Gateway / UI] Add Model select functionality for endpoint creation (#19477, @BenWilson2)
- [Gateway / UI] Add Auth config to endpoint creation (#19494, @BenWilson2)
- [Gateway / UI] Add the Endpoint Edit Page (#19502, @BenWilson2)
- [Gateway / UI] Refactor the provider display for better UX (#19503, @BenWilson2)
- [Gateway / UI] Create Endpoint details page (#19537, @BenWilson2)
- [Gateway / UI] Add security notice banner (#19538, @BenWilson2)
- [Gateway / UI] Create common editable combo box with extra modal select (#19546, @BenWilson2)
- [Evaluation] Introduce `MemAlign` as a new optimizer for judge alignment (#19598, @smoorjani)
- [Evaluation] Parallelize LLM calls in `MemAlign` guideline distillation (#20291, @veronicalyu320)
- [Evaluation] Add `GePaAlignmentOptimizer` for judge instruction optimization (#19882, @alkispoly-db)
- [Evaluation] Add `Fluency` scorer for evaluating text quality (#19414, @alkispoly-db)
- [Evaluation] Add `KnowledgeRetention` built-in scorer (#19436, @alkispoly-db)
- [Evaluation] Implement automatic discovery for builtin scorers (#19443, @alkispoly-db)
- [Evaluation] Add Phoenix (Arize) third-party scorer integration (#19473, @debu-sinha)
- [Evaluation] Add gateway provider support for scorers (#19470, @danielseong1)
- [Evaluation] Introduce a conversation simulator into `mlflow.genai` (#19614, @smoorjani)
- [Evaluation] Integrate conversation simulation into `mlflow.genai.evaluate` (#19760, @smoorjani)
- [Evaluation] Make conversation simulator work with datasets (#19845, @SomtochiUmeh)
- [Evaluation] Support for conversational datasets with persona, goal, and context (#19686, @SomtochiUmeh)
- [Evaluation] Introduce conversational guidelines scorer (#19729, @smoorjani)
- [Evaluation] Update tool call correctness judge to accept expected tool calls (#19613, @smoorjani)
- [Evaluation] Support trace parsing fallback using Databricks model (#19654, @AveshCSingh)
- [Evaluation] Documentation for online evaluation / scoring (#20103, @dbczumar)
- [Evaluation] Job backend: Update job backend to use static names rather than function full names (#19430, @WeichenXu123)
- [Evaluation] Job backend: support job cancellation (#19565, @WeichenXu123)
- [Tracing] Support distributed tracing (#19920, @WeichenXu123)
- [Tracing] Trace Metrics backend (#19271, @serena-ruan)
- [Tracing] Add `IS NULL` / `IS NOT NULL` comparator support for trace metadata filtering (#19720, @dbczumar)
- [Tracing] Auto-navigate to Events tab when clicking error spans (#20188, @anshuman-sahu)
- [Tracing] Support shift+select for Traces (#20125, @B-Step62)
- [Tracing] SpringAI Integration (#19949, @joelrobin18)
- [Tracing] Reasoning in Chat UI for OpenAI, Anthropic, Gemini, Langchain, and PydanticAI (#19535, #19541, #19627, #19651, #19657, @joelrobin18)
- [UI] Merge MLflow Assistant branch (#20011, @B-Step62)
- [UI] Current Page context to assistant (#20139, @joelrobin18)
- [UI] Assistant regenerate button (#20066, @joelrobin18)
- [UI] Copy button Assistant (#20063, @joelrobin18)
- [UI] Overview tab for GenAI experiments (#19521, @serena-ruan)
- [UI] Enable Scorers UI feature flags (#19842, @danielseong1)
- [UI] Improve LLM judge creation modal UX and variable ordering (#19963, @danielseong1)
- [UI] Hide instructions section for built-in LLM judges (#19883, @danielseong1)
- [UI] Change model provider and name to dropdown list (#19653, @chenmoneygithub)
- [Prompts] Support Jinja2 template in prompt registry (#19772, @B-Step62)
- [Prompts] Support metaprompting in `mlflow.genai.optimize_prompts()` (#19762, @chenmoneygithub)
- [Prompts] Add option to delegate saving dspy model to `dspy.module.save` API (#19704, @WeichenXu123)
- [Prompts / UI] Add traces mode to prompts details page and implement filtered traces (#19599, @TomeHirata)
- [Tracking] Support `mlflow.genai.to_predict_fn` for app invocation endpoints (#19779, @jennsun)
- [Tracking] Add `log_stream` API for logging binary streams as artifacts (#19104, @harupy)
- [Tracking] Add `import_checkpoints` API for databricks SGC Checkpointing with MLflow (#19839, @WeichenXu123)
- [Tracking] Support GC clean up for Historical Jobs (#19626, @joelrobin18)
- [Tracking] Add `JupyterNotebookRunContext` for Tracking local Jupyter notebook as the source (#19162, @iyashk)
- [Tracking] Full docker image support with db (#19979, @serena-ruan)
- [Tracking] Add react route handling to communicate with the tracking server (#19010, @BenWilson2)
- [Tracking] [TypeScript SDK] Simplify Databricks auth by delegating to Databricks SDK (#19434, @simonfaltum)
- [Models] Safe model serialization: Support saving pytorch model via `torch.export.save`, add `skops` serialization format, and deprecate unsafe pickle/cloudpickle formats (#18759, #18832, #19692, #20151, @WeichenXu123)

Bug fixes:

- [Gateway] Fix Anthropic and Gemini streaming for LiteLLM providers (#20398, @TomeHirata)
- [Build] Include git submodule contents in Python package build (#20394, @copilot-swe-agent)
- [Tracing] Fix duplicate traces in semantic kernel autolog (#20206, @harupy)
- [Tracing] Fix Claude autolog to prioritize settings.json over OS environment variables (#20376, @alkispoly-db)
- [Evaluation] Fix temperature/json issues with `ConversationSimulator` on managed (#20236, @xsh310)
- [Tracing / UI] Add support for OpenAI function calling inputs in chat UI parsing (#20058, @daniellok-db)
- [Tracking] Update checking code for pickle deserialization (#20267, @WeichenXu123)
- [Gateway] Fix Vertex AI model configuration (#20242, @TomeHirata)
- [UI] Store gateway<>scorer binding correctly (#20176, @TomeHirata)
- [Evaluation] Support `SparkDF` trace handling in eval (#20207, @BenWilson2)
- [Evaluation] Fix tool name extraction for tool call correctness (#20201, @smoorjani)
- [Prompts] Fix scorers issue in metaprompting (#20173, @chenmoneygithub)
- [UI] Propagate Run id context to Assistant (#20138, @joelrobin18)
- [Model Registry] Allow for model registration to use KMS auth from different workspace (#20156, @BenWilson2)
- [UI] Improve scorer trace picker UX and validation (#20178, @danielseong1)
- [Evaluation] Improve `MemAlign` optimizer for incremental judge alignment (#20049, @veronicalyu320)
- [Evaluation] Fix bug with max tokens using max output tokens (#20174, @smoorjani)
- [Evaluation] Fix a race condition bug when using DF inputs for genai eval (#20079, @BenWilson2)
- [Tracking] Fix `DATABRICKS_CONFIG_PROFILE` env var detection when fetching databricks credentials (#20112, @daniellok-db)
- [Gateway] Move gateway invocation validation to fastapi middleware (#20111, @TomeHirata)
- [Prompts] Fix the length check in `mlflow.genai.optimize_prompts()` (#19993, @chenmoneygithub)
- [UI] Fix trace selection not registering in SelectTracesModal (#20099, @joelrobin18)
- [UI] Fix LimitOverrunError in Assistant streaming (#20078, @joelrobin18)
- [Tracing] CC Token usage (#20022, @joelrobin18)
- [Gateway] Remove MLflow-specific `auth_mode` from `LiteLLMConfig` (#20059, @TomeHirata)
- [UI] Assistant UI fix for dark theme (#20056, @joelrobin18)
- [Tracing] Isolate runtime context between opentelemetry and mlflow (#19797, @B-Step62)
- [UI] Prevent spurious 404 requests for relative image URLs in markdown (#20003, @harupy)
- [Tracing] Support MLflow tracing with OpenTelemetry auto-instrumentation (#19501, @serena-ruan)
- [UI] [UI] Fix session selector table column resizing and link behavior (#19927, @danielseong1)
- [Gateway] Add Azure provider support in gateway configuration (#19933, @TomeHirata)
- [Gateway] Propagate extra auth config to LiteLLM provider (#19931, @TomeHirata)
- [Evaluation / UI] Add missing retrieval context error for retrieval scorers (#19895, @danielseong1)
- [Evaluation / UI] Improve trace selection UX in scorer/judge UI (#19913, @danielseong1)
- [Model Registry / Models] Fix `infer_code_paths` to capture transitive imports of functions/classes (#19814, @copilot-swe-agent)
- [Tracking] fix for addressing rest api call latency in databricks job run (#19886, @WeichenXu123)
- [UI] Enable {{trace}} variable support in sample judge evaluation (#19851, @danielseong1)
- [Scoring] Check security before extracting tar file (#19557, @WeichenXu123)
- [Gateway] Fix authorization header duplication (#19853, @TomeHirata)
- [Gateway] Fix Gateway error handling to translate `MlflowException` to `HTTPException` (#19728, @danielseong1)
- [Gateway] Remove `gateway_deprecated` decorator - AI Gateway is not deprecated (#19821, @copilot-swe-agent)
- [Tracking] Make local artifact location creation lazy to support read-only proxy environments (#19678, @BenWilson2)
- [Evaluation] fixed databricks hosted llm failure due to `response_schema` injection (#19741, @sinanshamsudheen)
- [Evaluation] Add `@overload` annotations to `@scorer` decorator for proper type inference (#19570, @mr-brobot)
- [Tracking] Add debug logging for 500 errors in `catch_mlflow_exception` (#19781, @harupy)
- [Tracing] [Bug fix] Support search traces by string feedback / expectation values (#19719, @dbczumar)
- [Tracing / UI] Fix scorer creation UX issues (#19756, @danielseong1)
- [Evaluation] Fix `KnowledgeRetention` model parameter not propagating to inner scorer (#19753, @danielseong1)
- [Tracking] [BUG] `serve-artifacts` is not enabled in docker-compose #19700 (#19701, @zjffdu)
- [Tracing] Fix type signature loss in `@trace_disabled` decorator (#19569, @mr-brobot)
- [Tracking] Fix: Return 400 instead of 500 for invalid experiment_id (#19655, @copilot-swe-agent)
- [Models] Fix schema enforcement for pandas `StringDtype` (#19518, @harupy)
- [Tracing] Fix Python 3.12 `DeprecationWarning` from `generator.throw()` in tracing (#19629, @mr-brobot)
- [Evaluation] Fix structured outputs for databricks serving endpoints (#19572, @smoorjani)
- [Models / Scoring] Add dict to `PyFuncOutput` type alias for `ResponsesAgent`/`ChatAgent`/`ChatModel` (#19560, @copilot-swe-agent)
- [Tracking] Fix `enable_git_model_versioning` to work from subdirectories (#19529, @copilot-swe-agent)

Documentation updates:

- [Docs] fix: Remove `multi_class` argument from scikit-learn's `LogisticRegression` in docs (#20266, @SOORAJTS2001)
- [Docs] Add doc for distributed tracing (#20027, @WeichenXu123)
- [Docs] Add Judge Builder UI documentation (#20163, @danielseong1)
- [Docs] Add framework integration examples for AI Gateway query-endpoint page (#20137, @TomeHirata)
- [Docs] Add "Evaluation Examples" article (#19722, @achen530)
- [Docs] [1/3] Add gateway tracing guide for LiteLLM, OpenRouter, and Vercel AI Gateway (#20031, @B-Step62)
- [Docs] Update prompt optimization doc to include metaprompting (#19966, @chenmoneygithub)
- [Docs] Reorganize gateway page structure (#19968, @TomeHirata)
- [Build / Docs] Fix broken auth REST API documentation links (#19872, @copilot-swe-agent)
- [Docs] Add setup and query documentation for new AI Gateway (#19804, @TomeHirata)
- [Docs] Add additional eval dataset serialization examples (#19697, @BenWilson2)
- [Docs] ML-60766: Add dataset schema from managed content to SDK reference page (#19676, @achen530)
- [Docs / Prompts] Fix duplicate tags argument in `register_prompt` documentation example (#19591, @copilot-swe-agent)
- [Docs] Fix ML-59546 eval quickstart links to wrong place, add notebook version of eval quickstart (#19511, @achen530)
- [Docs] Add documentation for `KnowledgeRetention` scorer (#19478, @alkispoly-db)

Small bug fixes and documentation updates:

#20406, #20122, #20317, #20333, #20361, #20274, #20362, #20249, #20169, #20345, #20252, #20314, #20214, #20215, #20210, #20212, #20142, #20183, #20121, #20141, #20140, #20124, #20073, #20062, #20065, #19893, #19912, #19464, #19857, #19401, #19600, #19555, #19400, #19392, #19393, @B-Step62; #20323, #20263, #19982, #20218, #20143, #20146, #20145, #20064, #20117, #20144, #20110, #20050, #20017, #20116, #20118, #19989, #19953, #19836, #19915, #19955, #19952, #19940, #19939, #19938, #19937, #19877, #19874, #19869, #19867, #19865, #19837, #19835, #19834, #19864, #19873, #19833, #19825, #19876, #19799, #19798, #19793, #19771, #19770, #19635, #19634, #19633, #19632, #19624, #19622, #19621, #19620, #19631, #19619, #19747, #19609, #19608, #19607, #19606, #19604, #19603, #19602, #19601, #19588, #19587, #19581, #19585, #19610, #19590, #19580, #19579, #19578, #19577, #19576, #19234, @serena-ruan; #20378, #20385, #20205, #20237, #20193, #20171, #20155, #20170, #20132, #20097, #20100, #20101, #19736, #19717, #19716, #19759, #19718, #19714, #19713, #19712, #19711, #19840, #19710, #19709, #19708, #19777, #19707, @dbczumar; #20387, #19981, #19964, @bbqiu; #20390, #20334, #20208, #19978, #19980, #19875, #19854, #19816, #19815, #19796, #19806, #19785, #19789, #19769, #19748, #19773, #19782, #19706, #19523, #19505, #19450, #19482, #19458, #19433, #19431, #19455, #19417, #19426, #19424, @harupy; #20355, #20245, #20120, #20229, #20114, #20053, #20012, #19972, #20002, #19991, #19990, #19977, #19986, #19985, #19967, #19957, #19960, #19954, #19945, #19941, #19934, #19917, #19916, #19905, #19904, #19903, #19900, #19899, #19897, #19894, #19892, #19890, #19888, #19887, #19861, #19828, #19818, #19803, #19802, #19791, #19788, #19795, #19790, #19786, #19783, #19767, #19768, #19746, #19735, #19733, #19732, #19726, #19561, #19549, #19544, #19543, #19510, #19486, #19487, #19463, #19871, @copilot-swe-agent; #20308, #20264, #20109, #20181, #20180, #20177, #20134, #20107, #20015, #20007, #20008, #19930, #20006, #20005, #19965, #19942, #19944, #19950, #19936, #19947, #19948, #19946, #19870, #19824, #19823, #19856, #19863, #19858, #19860, #19849, #19822, #19765, #19792, #19764, #19763, #19618, #19453, #19452, #19404, #19390, #19290, @TomeHirata; #20350, #20203, #19675, #19677, #19674, #19476, #19447, @BenWilson2; #20286, #20157, #20051, #20216, #20200, #20213, #20194, #20072, #20195, #20175, #20039, #19844, #19935, #19696, #19451, #19409, @smoorjani; #20209, #20131, #19742, #19969, #19734, #19480, #19351, @daniellok-db; #20204, #20164, #20192, #19997, #19925, #19850, #19914, #19774, #19721, #19673, #19623, #19668, #19496, #19554, #19471, @danielseong1; #20037, #19884, #19846, #19843, #19813, #19454, #19391, #19322, #19388, #19307, #19382, @xsh310; #20130, @iyashk; #20147, #20030, #19962, #19826, @kevin-lyn; #20108, #20071, #19743, #20045, #20042, #19959, #19880, @SomtochiUmeh; #20025, #19662, #19749, #19738, #19419, @WeichenXu123; #19847, @jaceklaskowski; #19820, @Abhiii47; #19800, @shreenidhi2205; #19703, #19693, #19689, #19688, #19664, #19663, #19660, #19534, #19533, #19532, #19531, @hubertzub-db; #19652, @AMRUTH-ASHOK; #19493, #19495, @alkispoly-db; #16372, @mohammadsubhani; #19522, @pmeier

## 3.9.0rc0 (2026-01-15)

We're excited to announce MLflow 3.9.0rc0, a pre-release including several notable updates:

**Major New Features**:

- 🔮 **MLflow Assistant**: Figuring out the next steps to debug your apps and agents can be challenging. We're excited to introduce the MLflow Assistant, an in-product chatbot that can help you identify, diagnose, and fix issues. The assistant is backed by Claude Code, and directly passes context from the MLflow UI to Claude. Click on the floating "Assistant" button in the bottom right of the MLflow UI to get started!
- 📈 **Trace Overview Dashboard**: You can now get insights into your agent's performance at a glance with the new "Overview" tab in GenAI experiments. Many pre-built statistics are available out of the box, including performance metrics (e.g. latency, request count), quality metrics (based on assessments), and tool call summaries. If there are any additional charts you'd like to see, please feel free to raise an issue in the MLflow repository!
- ✨ **AI Gateway**: We're revamping our AI Gateway feature! AI Gateway provides a unified interface for your API requests, allowing you to route queries to your LLM provider(s) of choice. In MLflow 3.9.0rc0, the Gateway server is now located directly in the tracking server, so you don't need to spin up a new process. Additional features such as passthrough endpoints, traffic splits, and fallback models are also available, with more to come soon! For more detailed information, please take a look at the [docs](https://mlflow.org/docs/latest/genai/governance/ai-gateway/).
- 🔎 **Online Monitoring with LLM Judges**: Configure LLM judges to automatically run on your traces, without having to write a line of code! You can either use one of our [pre-defined judges](https://mlflow.org/docs/latest/genai/eval-monitor/scorers/llm-judge/predefined/), or provide your own prompt and instructions to create custom metrics. Head to the new "Judges" tab within the GenAI Experiment UI to get started.
- 🤖 **Judge Builder UI**: Define and iterate on custom LLM judge prompts directly from the UI! Within the new "Judges" tab, you can create your own prompt for an LLM judge, and test-run it on your traces to see what the output would be. Once you're happy with it, you can either use it for online monitoring (as mentioned above), or use it via the Python SDK for your evals.
- 🔗 **Distributed Tracing**: Trace context can now be propagated across different services and processes, allowing you to truly track request lifecycles from end to end. The related APIs are defined in the `mlflow.tracing.distributed` module (with more documentation to come soon).
- 📚 **MemAlign - a new judge optimizer algorithm**: We're excited to introduce `MemAlignOptimizer`, a new algorithm that makes your judges smarter over time. It learns general guidelines from past feedback while dynamically retrieving relevant examples at runtime, giving you more accurate evaluations.

Stay tuned for the full release, which will be packed with even more features and bugfixes.

To try out this release candidate, please run:

`pip install mlflow==3.9.0rc0`

## 3.8.1 (2025-12-26)

MLflow 3.8.1 includes several bug fixes and documentation updates.

Bug fixes:

- [Tracking] Skip registering sqlalchemy store when sqlalchemy lib is not installed (#19563, @WeichenXu123)
- [Models / Scoring] fix(security): prevent command injection via malicious model artifacts (#19583, @ColeMurray)
- [Prompts] Fix prompt registration with model_config on Databricks (#19617, @TomeHirata)
- [UI] Fix UI blank page on plain HTTP by replacing crypto.randomUUID with uuid library (#19644, @copilot-swe-agent)

Small bug fixes and documentation updates:

#19539, #19451, #19409, @smoorjani; #19493, @alkispoly-db

## 3.8.0 (2025-12-19)

MLflow 3.8.0 includes several major features and improvements

### Major Features

- ⚙️ **Prompt Model Configuration**: Prompts can now include model configuration, allowing you to associate specific model settings with prompt templates for more reproducible LLM workflows. (#18963, #19174, #19279, @chenmoneygithub)
- ⏳ **In-Progress Trace Display**: The Traces UI now supports displaying spans from in-progress traces with auto-polling, enabling real-time debugging and monitoring of long-running LLM applications. (#19265, @B-Step62)
- ⚖️ **DeepEval Judges Integration**: New `get_judge` API enables using DeepEval's evaluation metrics as MLflow scorers, providing access to 20+ evaluation metrics including answer relevancy, faithfulness, and hallucination detection. (#18988, @smoorjani)
- 🛡️ **Conversational Safety Scorer**: New built-in scorer for evaluating safety of multi-turn conversations, analyzing entire conversation histories for hate speech, harassment, violence, and other safety concerns. (#19106, @joelrobin18)
- ⚡ **Conversational Tool Call Efficiency Scorer**: New built-in scorer for evaluating tool call efficiency in multi-turn agent interactions, detecting redundant calls, missing batching opportunities, and poor tool selections. (#19245, @joelrobin18)

### Important Notice

- **Collection of UI Telemetry**. From MLflow 3.8.0 onwards, MLflow will collect anonymized data about UI interactions, similar to the telemetry we collect for the Python SDK. If you manage your own server, UI telemetry is automatically disabled by setting the existing environment variables: `MLFLOW_DISABLE_TELEMETRY=true` or `DO_NOT_TRACK=true`. If you do not manage your own server (e.g. you use a managed service or are not the admin), you can still opt out personally via the new "Settings" tab in the MLflow UI. For more information, please read the documentation on [usage tracking](https://mlflow.org/docs/latest/community/usage-tracking/).

### Features:

- [Tracking] Add default passphrase support (#19360, @BenWilson2)
- [Tracing] Pydantic AI Stream support (#19118, @joelrobin18)
- [Docs] Deprecate Unity Catalog function integration in AI Gateway (#19457, @harupy)
- [Evaluation] Add basic RAGAS judges wrapping for MLflow (#19345, @SomtochiUmeh)
- [Tracking] Add `--max-results` option to mlflow experiments search (#19359, @alkispoly-db)
- [Tracking] Enhance encryption security (#19253, @BenWilson2)
- [Tracking] Fix and simplify Gateway store interfaces (#19346, @BenWilson2)
- [Evaluation] Add inference_params support for LLM Judges (#19152, @debu-sinha)
- [Tracing] Support batch span export to UC Table (#19324, @B-Step62)
- [Tracking] Add endpoint tags (#19308, @BenWilson2)
- [Docs / Evaluation] Add MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS to limit concurrent scorer execution (#19248, @debu-sinha)
- [Evaluation / Tracking] Enable search_datasets in Databricks managed MLflow (#19254, @alkispoly-db)
- [Prompts] render text prompt previews in markdown (#19200, @ispoljari)
- [UI] Add linked prompts filter for trace search tab (#19192, @TomeHirata)
- [Evaluation] Automatically wrap async functions when passed to predict_fn (#19249, @smoorjani)
- [Evaluation] [3/6][builtin judges] Conversational Role Adherence (#19247, @joelrobin18)
- [Tracking] [Endpoints] [1/x] Add backend DB tables for Endpoints (#19002, @BenWilson2)
- [Tracking] [Endpoints] [3/x] Entities base definitions (#19004, @BenWilson2)
- [Tracking] [Endpoints] [4/x] Abstract store interface (#19005, @BenWilson2)
- [Tracking] [Endpoints] [5/x] SQL Store backend for Endpoints (#19006, @BenWilson2)
- [Tracking] [Endpoints] [6/x] Protos and entities interfaces (#19007, @BenWilson2)
- [Tracking] [Endpoints] [7/x] Add rest store implementation (#19008, @BenWilson2)
- [Tracking] [Endpoints] [8/x] Add credential cache (#19014, @BenWilson2)
- [Tracking] [Endpoints] [9/x] Add provider, model, and configuration handling (#19009, @BenWilson2)
- [Evaluation / UI] Add show/hide visibility control for Evaluation runs chart view (#18797) (#18852, @pradpalnis)
- [Tracking] Add mlflow experiments get command (#19097, @alkispoly-db)
- [Server-infra] [ Gateway 1/10 ] Simplify secrets and masked secrets with map types (#19440, @BenWilson2)

### Bug fixes:

- [Tracing / UI] Branch 3.8 patch: Fix GraphQL SearchRuns filter using invalid attribute key in trace comparison (#19526, @WeichenXu123)
- [Scoring / Tracking] Fix artifact download performance regression (#19520, @copilot-swe-agent)
- [Tracking] Fix SQLAlchemy alias conflict in `_search_runs` for dataset filters (#19498, @fredericosantos)
- [Tracking] Add auth support for GraphQL routes (#19278, @BenWilson2)
- [] Fix SQL injection vulnerability in UC function execution (#19381, @harupy)
- [UI] Fix MultiIndex column search crash in dataset schema table (#19461, @copilot-swe-agent)
- [Tracking] Make datasource failures fail gracefully (#19469, @BenWilson2)
- [Tracing / Tracking] Fix litellm autolog for versions >= 1.78 (#19459, @harupy)
- [Model Registry / Tracking] Fix SQLAlchemy engine connection pool leak in model registry and job stores (#19386, @harupy)
- [UI] [Bug fix] Traces UI: Support filtering on assessments with multiple values (e.g. error and boolean) (#19262, @dbczumar)
- [Evaluation / Tracing] Fix error initialization in Feedback (#19340, @alkispoly-db)
- [Models] Switch container build to subprocess for Sagemaker (#19277, @BenWilson2)
- [Scoring] Fix scorers issue on Strands traces (#18835, @joelrobin18)
- [Tracking] Stop initializing backend stores in artifacts only mode (#19167, @mprahl)
- [Evaluation] Parallelize multi-turn session evaluation (#19222, @AveshCSingh)
- [Tracing] Add safe attribute capture for pydantic_ai (#19219, @BenWilson2)
- [Model Registry] Fix UC to UC copying regression (#19280, @BenWilson2)
- [Tracking] Fix artifact path traversal vector (#19260, @BenWilson2)
- [UI] Fix issue with auth controls on system metrics (#19283, @BenWilson2)
- [Models] Add context loading for ChatModel (#19250, @BenWilson2)
- [Tracing] Fix trace decorators usage for LangGraph async callers (#19228, @BenWilson2)
- [Tracking] Update docker compose to use --artifacts-destination not --default-artifact-root (#19215, @B-Step62)
- [Build] Reduce clint error message verbosity by consolidating README instructions (#19155, @copilot-swe-agent)

### Documentation updates:

- [Docs] Add specific references for correctness scorers (#19472, @BenWilson2)
- [Docs] Add documentation for Fluency scorer (#19481, @alkispoly-db)
- [Docs] Update eval quickstart to put all code into a script (#19444, @achen530)
- [Docs] Add documentation for `KnowledgeRetention` scorer (#19478, @alkispoly-db)
- [Evaluation] Fix non-reproducible code examples in deep-learning.mdx (#19376, @saumilyagupta)
- [Docs / Evaluation] fix: Confusing documentation for `mlflow.genai.evaluate()` (#19380, @brandonhawi)
- [Docs] Deprecate model logging of OpenAI flavor (#19325, @TomeHirata)
- [Docs] Add rounded corners to video elements in documentation (#19231, @copilot-swe-agent)
- [Docs] Sync Python/TypeScript tab selections in tracing quickstart docs (#19184, @copilot-swe-agent)

Small bug fixes and documentation updates:

#19497, #19358, #19322, #19383, #19288, #19287, #19230, #19225, @xsh310; #19504, @WeichenXu123; #19499, #19465, #19241, @B-Step62; #19479, #19385, #19297, #19347, #19314, #19286, #19269, @TomeHirata; #18894, @BnnaFish; #19480, #19427, #19351, #19312, #19292, #19303, #19291, #19418, #19395, #19240, #19267, #19102, #19082, #19076, @daniellok-db; #19463, #19370, #19369, #19368, #19367, #19366, #19363, #19354, #19302, #19272, #19266, #19258, #19255, #19242, #19236, #19235, #19203, #19214, #19212, #19210, #19204, #19197, #19196, #19194, #19190, #19182, #19178, #19179, #19163, #19157, #19150, #19137, #19132, #19114, #19115, #19113, #19112, #19111, #19110, #19107, #19091, #19090, #19078, @copilot-swe-agent; #19437, @SomtochiUmeh; #19420, #19329, #19317, #19207, #19086, @kevin-lyn; #19339, #19263, #19438, #19412, #19411, #19355, #19341, #19034, #19029, #19252, @smoorjani; #19416, #19399, #19402, #19353, #19313, #19296, #19294, #19264, #19202, #19206, #19165, #19161, #19158, #19126, #19147, #19099, @harupy; #19357, #19343, #19342, #19335, #19261, #19226, #19227, @BenWilson2; #19344, #19331, #19270, #19239, #19211, @serena-ruan; #19323, @bbqiu; #19373, @alkispoly-db; #19320, #19311, @kriscon-db; #19309, @stefanwayon; #19063, @cyficowley; #19160, @Killian-fal; #19142, #19141, @dbczumar; #19089, @hubertzub-db; #19098, @achen530

## 3.7.0 (2025-12-05)

MLflow 3.7.0 includes several major features and improvements for GenAI Observability, Evaluation, and Prompt Management.

### Major Features

- 📝 **Experiment Prompts UI**: New prompts functionality in the experiment UI allows you to manage and search prompts directly within experiments, with support for filter strings and prompt version search in traces. (#19156, #18919, #18906, @TomeHirata)
- 💬 **Multi-turn Evaluation Support**: Enhanced `mlflow.genai.evaluate` now supports multi-turn conversations, enabling comprehensive assessment of conversational AI applications with DataFrame and list inputs. (#18971, @AveshCSingh)
- ⚖️ **Trace Comparison**: New side-by-side comparison view in the Traces UI allows you to analyze and debug LLM application behavior across different runs, making it easier to identify regressions and improvements. (#17138, @joelrobin18)
- 🌐 **Gemini TypeScript SDK**: Auto-tracing support for Google's Gemini in TypeScript, expanding MLflow's observability capabilities for JavaScript/TypeScript AI applications. (#18207, @joelrobin18)
- 🎯 **Structured Outputs in Judges**: The `make_judge` API now supports structured outputs, enabling more precise and programmatically consumable evaluation results. (#18529, @TomeHirata)
- 🔗 **VoltAgent Tracing**: Added auto-tracing support for VoltAgent, extending MLflow's observability to this AI agent framework. (#19041, @joelrobin18)

### Breaking Changes

- [Tracking] SQLite is now the default backend for the MLflow Tracking server. (#18497, @harupy)
- [Models] Remove deprecated `diviner` flavor (#18808, @copilot-swe-agent)
- [Models] Remove deprecated `promptflow` flavor (#18805, @copilot-swe-agent)

### Features

- [Tracking] Create parent directories for SQLite database files (#19205, @harupy)
- [Prompts] Link Prompts and Experiments when prompts are loaded/registered (#18883, @TomeHirata)
- [Tracking] Include environment variable fallback for SGC run resumption (#19143, @artjen)
- [Tracking] Add support for SGC run resumption from Databricks Jobs (#19015, @artjen)
- [Evaluation] Add `--builtin/-b` flag to `mlflow scorers list` command (#19095, @alkispoly-db)
- [Tracing] Pydantic AI Chat UI support (#18777, @joelrobin18)
- [Tracking] Add auth support for scorers (#18699, @BenWilson2)
- [Evaluation] Remove experimental flags from scorers (#18122, @BenWilson2)
- [Evaluation] Add description field to all built-in scorers (#18547, @alkispoly-db)

### Bug Fixes

- [Tracing] Handle traces with third-party generic root span (#19217, @B-Step62)
- [Tracing] Fix OTLP endpoint path handling per OpenTelemetry spec (#19154, @harupy)
- [Tracing] Add gzip/deflate Content-Encoding support to OTLP traces endpoint (#19024, @Miaoxiang-philips)
- [Tracing] Add missing `_delete_trace_tag_v3` API (#18813, @Tian-Sky-Lan)
- [Tracing] Fix bug in chat sessions view where new sessions created after UI launch are not visible due to incorrect timestamp filtering (#18928, @dbczumar)
- [Tracing] Fix OTLP proto conversion for empty list/dict (#18958, @B-Step62)
- [Tracing] Agno V2 fixes (#18345, @joelrobin18)
- [Tracing] Fix `/v1/traces` endpoint to return protobuf instead of JSON (#18929, @copilot-swe-agent)
- [Tracing] Pin `click!=8.3.0` in MCP extra to fix MCP server failure (#18748, @copilot-swe-agent)
- [Tracing] Fix MCP server `uv` installation command for external users (#18745, @copilot-swe-agent)
- [Evaluation] Fix trace-based scorer evaluation by using agentic judge adapter (#19123, @alkispoly-db)
- [Evaluation] Fix managed scorer registration failure (#19146, @xsh310)
- [Evaluation] Fix `InstructionsJudge` using scorer description as assessment value (#19121, @alkispoly-db)
- [Evaluation] Add validation to correctness judge expectation fields (#19026, @smoorjani)
- [Evaluation] Fix model URI underscore handling (#18849, @RohanRouth)
- [Evaluation] Fix `evaluate_traces` MCP tool error: use `result_df` instead of `tables` (#18825, @alkispoly-db)
- [Evaluation] Fix Bedrock Anthropic adapter by adding required `anthropic_version` field (#17744, @harupy)
- [Evaluation] Fix migration for pre-existing auth tables (#18793, @BenWilson2)
- [Tracking] Fix tracking URI propagation (#18023, @shaperilio)
- [Tracking] Fix `SqlLoggedModelMetric` association with `experiment_id` (#18382, @mcompen)
- [Tracking] Add Flask routes to auth validators (#18486, @BenWilson2)
- [Tracking] Add missing proto handler for Experiment association handling for datasets (#18769, @BenWilson2)
- [UI] Show full dataset record content and add search bar in evaluation datasets UI (#19000, @dbczumar)
- [UI] Request TraceInfo and Trace Assessments from a relative API path (#19032, @kbolashev)
- [UI] Define `LoggedModelOutput.to_dictionary()` so `LoggedModelOutput` and runs containing them can be JSON serialized (#19017, @nicklamiller)
- [UI] Fix router issue in TracesUI page (#19044, @joelrobin18)
- [Build] Fix `mlflow gc` to remove model artifacts (#17282, @joelrobin18)
- [Build] Fix Click 8.3.0 `Sentinel.UNSET` handling in MCP server (#18858, @harupy)
- [Build] Add bucket-ownership checks for Amazon S3 (#18542, @kingroryg)
- [Docs] Fix Python indentation in custom trace quickstart example (#19185, @copilot-swe-agent)
- [Docs] Fix property blocks rendering horizontally in API documentation (#19125, @copilot-swe-agent)
- [Docs] Fix CLI link missing api_reference prefix in documentation sidebars (#18893, @copilot-swe-agent)
- [Docs] Fix notebook download URLs to use versioned paths (#18806, @harupy)
- [Docs] Fix documentation redirects for removed getting-started pages (#18789, @copilot-swe-agent)
- [Models] Fix shared cluster Py4j statefulness issue (#19139, @BenWilson2)
- [Models] Prevent symlink path traversal in local artifact store (#18964, @BenWilson2)

### Documentation Updates

- [Docs] Add LangGraph optimization guide (#19180, @TomeHirata)
- [Docs] Add documentation for milestone 1 of multi-turn evaluation support (#19033, @smoorjani)
- [Docs] Update transformers and sentence transformers docs (#18925, @BenWilson2)
- [Docs] Clean up Classic Eval docs (#19013, @BenWilson2)
- [Docs] Improve documentation for `prompt_template` (#19105, @ingo-stallknecht)
- [Docs] Fix typos in ML documentation main page (#19048, @copilot-swe-agent)
- [Docs] Convert documentation GIF animations to MP4 videos (#18946, @harupy)
- [Docs] Improve readability by adjusting sidebar layout and style (#18937, @kevin-lyn)
- [Docs] Clean up scikit-learn docs (#18794, @BenWilson2)
- [Docs] Clean up XGBoost docs (#18790, @BenWilson2)
- [Docs] Clean up TensorFlow docs (#18850, @BenWilson2)
- [Docs] Use the correct OTLP HTTP exporter in OTel collector YAML (#18930, @Miaoxiang-philips)
- [Docs] Clean up SpaCy and Keras docs (#18895, @BenWilson2)
- [Docs] Fix contents in tracing doc pages (#18750, @B-Step62)
- [Docs] Improve file store deprecation warning messages (#18900, @harupy)
- [Docs] Clean up the MLflow 3 docs content (#18871, @BenWilson2)
- [Docs] Add multi-turn judge creation with `make_judge` API and direct judge invocation (#18897, @xsh310)
- [Docs] Clean up PyTorch docs (#18816, @BenWilson2)
- [Docs] Clean up Prophet docs (#18814, @BenWilson2)
- [Docs] Clean up SparkML docs (#18811, @BenWilson2)
- [Docs] Clean up the traditional ML landing page (#18799, @BenWilson2)
- [Docs] Clean up the Deep Learning landing page (#18820, @BenWilson2)
- [Docs] Clean up evaluation datasets docs (#18766, @BenWilson2)
- [Docs] Fix OpenTelemetry documentation (#18810, @joelrobin18)
- [Docs] Clarify `mlflow gc` command behavior for pinned runs and registered models (#18704, @copilot-swe-agent)

Small bug fixes and documentation updates:

#19220, #19140, #19141, #18984, #18985, #18822, @dbczumar; #19148, @ingo-stallknecht; #19183, #19201, #19130, #19049, #19030, #18778, #18780, #18556, #18555, @serena-ruan; #19153, #19181, #18784, #18783, #18802, #18881, #18695, #18879, #18782, #18845, #18787, #18786, #18590, @B-Step62; #19208, #19021, #19023, #18723, #18622, @smoorjani; #13314, @alokshenoy; #19138, #19171, #19146, #19067, #19064, #19045, #18968, #18967, #19018, #18966, #18990, #18912, @xsh310; #19168, @mcompen; #19145, #18702, #18642, @BenWilson2; #19126, #19022, #18951, #18887, #18954, #18949, #18934, #18914, #18903, #18877, #18859, #18838, #18828, #18821, #18717, #18710, #18756, #18713, @harupy; #18890, #18862, #18836, #18792, #18818, #18579, @TomeHirata; #19084, #18886, #18911, #18904, #18885, #18837, #18795, #18646, @daniellok-db; #18992, #19025, #19020, #18950, @kevin-lyn; #19069, #19072, #19043, #19027, #19028, #19019, #18995, #18997, #18989, #18991, #18987, #18983, #18980, #18979, #18974, #18972, #18969, #18948, #18940, #18942, #18939, #18938, #18933, #18932, #18931, #18915, #18882, #18865, #18861, #18860, #18846, #18841, #18830, #18824, #18823, #18819, #18789, #18804, #18779, #18775, #18772, #18704, #18606, #18748, #18746, #18745, #18743, #18732, #18737, #18736, #18729, #18718, #18703, #18693, #18686, #18682, #18633, #18675, #18671, #18653, #18652, @copilot-swe-agent; #19001, #18945, @danielseong1; #18815, @kevin-wangg; #19039, #18898, @AveshCSingh; #18742, @Killian-fal; #18923, @HomeLH; #18922, #18920, @UnfixedMold; #18798, @WeichenXu123; #18776, @pcliupc; #18417, @shaperilio

## 3.7.0rc0 (2025-11-27)

MLflow 3.7.0rc0 includes several major features and improvements!

### Major Features

- ⚖️ **Trace Comparison**: New UI feature allowing side-by-side comparison of traces to analyze and debug LLM application behavior across different runs. (#17138, @joelrobin18, @daniellok-db)
- 💬 **Multi-turn conversation support for Evaluation**: Enhanced evaluation support for multi-turn conversations in `mlflow.genai.evaluate`, enabling comprehensive assessment of conversational AI applications. (#18971, #19039, @AveshCSingh)
- 🔎 **Full Text Trace Search from UI**: Search across all trace content directly from the UI, making it easier to find specific traces by searching through inputs, outputs, and span details. (#18683, @dbczumar)
- 🌐 **Gemini TypeScript SDK**: Auto-tracing support for Gemini in TypeScript, expanding MLflow's observability capabilities for JavaScript/TypeScript AI applications. (#18207, @joelrobin18)

### Breaking Changes

- **SQLite as Default Backend**: MLflow now uses SQLite as the default backend instead of file-based storage, unless existing mlruns data is detected. This improves performance and reliability for tracking experiments. (#18497, @harupy)
- **Removed Deprecated Flavors**: The `diviner` and `promptflow` flavors have been removed from MLflow. Please migrate to supported alternatives. (#18808, #18805, @copilot-swe-agent)

### Important Notice

- **Installation ID for Telemetry**: MLflow now generates a unique installation ID (a randomly generated UUID) for telemetry purposes to better understand usage patterns. This ID is fully anonymous and persists across sessions. Telemetry can be disabled anytime by setting `MLFLOW_DISABLE_TELEMETRY=true` or `DO_NOT_TRACK=true`. See the [usage tracking documentation](https://mlflow.org/docs/latest/community/usage-tracking/) for details. (#18881, @B-Step62)

Stay tuned for the full release, which will be packed with more features and bugfixes.

To try out this release candidate, please run:

`pip install mlflow==3.7.0rc0`

## 3.6.0 (2025-11-07)

MLflow 3.6.0 includes several major features and improvements for AI Observability, Experiment UI, Agent Evaluation and Deployment.

- 🔗 **Full OpenTelemetry Support in OSS Server**: MLflow now offers comprehensive OpenTelemetry integration, allowing you to ingest OpenTelemetry traces into MLflow and use both SDK seamlessly together. (#18540, #18532, #18357, @B-Step62, @serena-ruan)
- 💬 **Session-level View in Trace UI**: New chat sessions tab provides a dedicated view for organizing and analyzing related traces at the session level, making it easier to track conversational workflows. (#18594, @daniellok-db)
- 🧭 **New experiment tab bar**: The experiment tab navigation bar has been moved from the top of the page to the left side. As MLflow continues to grow, this layout provides more room to add new tabs while keeping everything easy to find. (#18594, @daniellok-db)
- 🚀 **New Supported Frameworks in TypeScript Tracing SDK**: Auto-tracing support for **Vercel AI SDK**, **Gemini**, **Anthropic**, **Mastra** in TypeScript, expanding MLflow's observability capabilities across popular JavaScript/TypeScript frameworks. (#18402, @B-Step62)
- 💰 **Tracking Judge Cost and Traces**: Comprehensive tracking of LLM judge evaluation costs and traces, providing visibility into evaluation expenses and performance with automatic cost calculation and rendering. (#18481, #18484, @B-Step62)
- ⚙️ **Agent Server**: New agent server infrastructure for managing and deploying scoring agents with enhanced orchestration capabilities. (#18596, @bbqiu)

Breaking changes:

- Deprecate pmdarima, promptflow, diviner flavors (#18597, #18577, @copilot-swe-agent)
- Drop numbering suffix (`_1`, `_2`, ...) from span names (#18531, @serena-ruan)

Features:

- [Tracing] Add RLIKE operator support for trace search (#18591, @serena-ruan)
- [Tracing] Attributes translation for OpenTelemetry clients (#18532, @serena-ruan)
- [Tracing] Implement auto-tracing logic for Vercel AI SDK (#18402, @B-Step62)
- [Tracing] Anthropic Typescript SDK (#18189, @joelrobin18)
- [Tracing] Support search by span details for traces in OSS MLflow server (#17918, @serena-ruan)
- [Tracing] Minor clean up for the trace summary view (#18436, @B-Step62)
- [Tracing] Log assessments to DSPy eval traces (#18136, @B-Step62)
- [Tracking] Add support for using the same DB for tracking and auth (#18384, @BenWilson2)
- [Tracking] Make Pytorch lightning autologging support logging model signature (#18510, @WeichenXu123)
- [Tracking] Make `mlflow.pytorch` pyfunc loader supporting pytorch forecasting model (#18428, @WeichenXu123)
- [Tracking] Job backend: Support creating virtual python environment for job execution (#18111, @WeichenXu123)
- [Evaluation] Add `search_traces` tool for agentic judge (#18228, @dbrx-euirim)
- [Evaluation] Record and render LLM judge cost (#18481, @B-Step62)
- [Evaluation] Frontend adjustments for handle judge traces (#18485, @B-Step62)
- [Evaluation] Record judge traces (#18484, @B-Step62)
- [Evaluation] Add support for profile usage in Databricks Agents dataset API operat… (#18431, @BenWilson2)
- [Evaluation] Add mlflow traces eval CLI command (#18069, @alkispoly-db)
- [Evaluation] Add mlflow scorers register-llm-judge CLI command (#18330, @alkispoly-db)
- [Evaluation] Add description property to Scorer interface (#18383, @alkispoly-db)
- [Evaluation] Allow passing empty scorer list for manual result comparison (#18265, @B-Step62)
- [Scoring] Introduce Agent Server (#18596, @bbqiu)
- [UI] Add chat sessions tab (#18594, @daniellok-db)
- [UI] Child Parent Link (#17248, @joelrobin18)
- [Models] Use UBJSON format as default for XGBoost models (#18420, @harupy)
- [Models] Support Langchain 1.x (#18490, @BenWilson2)
- [Model Registry] Add deprecation warnings for filesystem backends (#18524, @harupy)
- [Model Registry] Allow for skipping pip installation while packing environment for model serving (#18448, @juntai-zheng)
- [Gateway] Add configuration option for long-running deployments client requests (#18363, @BenWilson2)
- [Gateway] Make Openai, Anthropic, Gemini provider supporting streamed function calling response (#18367, #18328, #18294, @WeichenXu123)
- [Gateway] Add traffic route to multiple endpoints (#18064, @WeichenXu123)
- [Docs] Add Sticky Header to CodeBlock in MLflow/DOCS Code Examples (#18508, @PavithraNelluri)
- [Build / Evaluation] Add CLI command to list registered scorers by experiment (#18255, @alkispoly-db)

Bug fixes:

- [Evaluation] Fix plugin incompatibility with circular import (#18599, @BenWilson2)
- [Tracing] Paginate `delete_traces` calls to Databricks MLflow server (#18563, @dbrx-euirim)
- [Artifacts] Fix handling of `pathlib.Path` in `validation.py` (#16660, @benglewis)
- [Tracking] Enhance SqlAlchemyStore to include model outputs in run search results (#18568, @TomeHirata)
- [Prompts] Fix typo in gepa version (#18423, @TomeHirata)
- [Tracking] Add validation checks for search runs (#18487, @BenWilson2)
- [Tracking] Fix: Update run to use the new run name when resuming an existing run (#18511, @WeichenXu123)
- [UI] Fix search filter for metrics/params with spaces in names (#18503, @serena-ruan)
- [Evaluation] Remove the ability to register or load custom scorers (#18493, @BenWilson2)
- [UI] Fix assessment editing UI resetting field values when selecting name (#18474, @serena-ruan)
- [Evaluation] Add specificity to the system prompt for metrics (#18460, @BenWilson2)
- [Evaluation] [Eval #2] Support evaluating traces and linking to run in OSS (#18415, @B-Step62)
- [Tracking] Disable autologging for pytorch forecasting model predict method (#18444, @WeichenXu123)
- [Evaluation / Tracing] Reuse traces in genai.evaluate when endpoint uses dual-write mode (#18403, @harupy)
- [UI] Remove X-Frame-Options for notebook trace renderer (#18446, @TomeHirata)
- [Evaluation] Adjust util for remote tracking server declaration (#18411, @BenWilson2)
- [Evaluation / UI] Fix evaluation runs table link to point to traces tab instead of overview (#18332, @ritoban23)
- [Models] fix-streaming (#18337, @BenWilson2)
- [Evaluation / Tracing / Tracking] Job backend: Fix job store sql engine race condition (#18233, @WeichenXu123)
- [Evaluation] Add atomicity to job_start API (#18226, @BenWilson2)
- [Evaluation / Tracking] Job backend: Eager launch huey consumer to prevent Huey race condition (#18220, @WeichenXu123)

Documentation updates:

- [Docs] Add basic doc for Otel support (#18623, @B-Step62)
- [Docs] clarify datasets package requirement (#18610, @BenWilson2)
- [Evaluation] Deprecate v2 eval (#18470, @B-Step62)
- [Docs / UI] Add Sticky Header to CodeBlock in MLflow/DOCS Code Examples (#18508, @PavithraNelluri)
- [Docs] [Doc; 1/N] Clean up getting started for classical ML/DL (#18379, @B-Step62)
- [Docs] AI-gateway-revamp: Update doc (#18397, @WeichenXu123)
- [Docs] Fix documentation: update deprecated pandas fillna usage in classic-ml tutorial (#17927, @Kalindu-C)

Small bug fixes and documentation updates:

#18735, #18429, #18530, #18416, #18401, #18400, #18465, #18453, #18414, #18421, @B-Step62; #18641, #18631, #18629, #18605, #18426, #18603, #18526, #18587, #18583, #18564, #18536, #18544, #18567, #18565, #18533, #18535, #18501, #18498, #18368, #18357, #18471, #18476, #18356, #18214, #17975, @serena-ruan; #18600, #18604, #18602, #18566, #18549, #18538, #18517, #15849, #18492, #18468, #18475, #18469, #18467, #18452, #18449, #18450, #18447, #18442, #18327, #18395, #18418, #18350, #18278, #18242, #18234, #18203, #18175, #18210, @harupy; #18625, #18424, #18028, @daniellok-db; #18616, #18615, #18607, #18598, #18588, #18586, #18584, #18572, #18580, #18571, #18554, #18553, #18552, #18551, #18548, #18546, #18528, #18527, #18525, #18521, #18520, #18515, #18519, #18518, #18506, #18507, #18505, #18502, #18495, #18494, #18472, #18463, #18464, #18462, #18443, #18440, #18399, #18394, #18393, #18392, #18390, #18389, #18380, #18376, #18378, #18377, #18366, #18362, #18361, #18343, #18340, #18318, #18311, #18307, #18269, #18268, #18261, #18260, #18259, #18258, #18257, #18256, #18253, #18254, #18252, #18250, #18243, #18238, #18213, #18206, #18198, #18184, #18179, @copilot-swe-agent; #18578, #18569, @TomeHirata; #18575, @dbrx-euirim; #18570, #18116, #18360, #18351, @WeichenXu123; #18513, #18461, #18430, #18336, @BenWilson2; #18459, @smoorjani; #18488, @raymondzhou-db; #18334, @NJAHNAVI2907

## 3.6.0rc0 (2025-11-03)

MLflow 3.6.0rc0 includes several major features and improvements!

- Full OpenTelemetry Support in OSS Server
- Session-level View in Trace UI
- New experiment tab bar
- Vercel AI Support in TypeScript Tracing SDK
- Tracking Judge Cost and Traces
- Agent Server

Stay tuned for the full release, which will be packed with more features and bugfixes.

To try out this release candidate, please run:

`pip install mlflow==3.6.0rc0`

## 3.5.1 (2025-10-21)

MLflow 3.5.1 is a patch release that includes several bug fixes and improvements.

Features:

- [CLI] Add CLI command to list registered scorers by experiment (#18255, @alkispoly-db)
- [Deployments] Add configuration option for long-running deployments client requests (#18363, @BenWilson2)
- [Deployments] Create `set_databricks_monitoring_sql_warehouse_id` API (#18346, @dbrx-euirim)
- [Prompts] Show instructions for prompt optimization on prompt registry (#18375, @TomeHirata)

Bug fixes:

- [Evaluation] Validate if trace is None before accessing the value in mlflow.genai.evaluate (#18285, @srinathmkce)
- [Evaluation] Revert "Add atomicity to job_start API (#18226)" (@serena-ruan)
- [MCP] Move fastmcp to optional mcp extra (#18422, @harupy)
- [Model Registry] Fix serialization bug in file store (#18365, @BenWilson2)
- [Scoring] Pin uvloop<0.22 to fix mlserver compatibility (#18370, @harupy)
- [Tracing] Fix a forward-compatibility issue with Span to_dict (#18439, @serena-ruan)
- [Tracing] Whitelist notebook trace UI renderer to allow display with default security settings (#18446, @TomeHirata)
- [Tracing] Fix attribute error in StrandsAgent tracing (#18409, @B-Step62)
- [Tracing] Adjust truncation logic in trace previews (#18412, @BenWilson2)
- [Tracing] Revert "Fix response handling in log_spans (#18280)" (#18349, @serena-ruan)
- [Tracking] Adjust util for remote tracking server declaration (#18411, @BenWilson2)
- [Tracking] Handle Databricks FMAPI style in openai autolog (#18354, @TomeHirata)
- [Tracking] Fetch config after adding first record (#18338, @serena-ruan)
- [UI] Fix span ID parsing in the UI (#18419, @daniellok-db)
- [UI] Fix Chat message parsing within the trace summary view modal (#18454, @daniellok-db)
- [UI] Fix an issue with display of the assessments pane in the UI (#18333, @BenWilson2)

Documentation updates:

- [Docs] Fix Kubernetes Deployment Tutorial Code (#18381, @Maeril)
- [Docs] Update the documentation around requirements for optimize_prompts (#18398, @TomeHirata)
- [Docs] Fix example FastAPI in track user sessions (#18388, @maxscheijen)

## 3.5.0 (2025-10-16)

MLflow 3.5.0 includes several major features and improvements!

### Major Features

- ⚙️ **Job Execution Backend**: Introduced a new job execution backend infrastructure for running asynchronous tasks with individual execution pools, job search capabilities, and transient error handling. (#17676, #18012, #18070, #18071, #18112, #18049, @WeichenXu123)
- 🎯 **Flexible Prompt Optimization API**: Introduced a new flexible API for prompt optimization with support for model switching and the GEPA algorithm, enabling more efficient prompt tuning with fewer rollouts. See the [documentation](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/) to get started. (#18183, #18031, @TomeHirata)
- 🎨 **Enhanced UI Onboarding**: Improved in-product onboarding experience with trace quickstart drawer and updated homepage guidance to help users discover MLflow's latest features. (#18098, #18187, @B-Step62)
- 🔐 **Security Middleware for Tracking Server**: Added a security middleware layer to protect against DNS rebinding, CORS attacks, and other security threats. Read the [documentation](https://mlflow.org/docs/latest/self-hosting/security/network/) for configuration details. (#17910, @BenWilson2)

### Features

- [Tracing / Tracking] Add `unlink_traces_from_run` batch operation (#18316, @harupy)
- [Tracing] Add batch trace link/unlink operations to DatabricksTracingRestStore (#18295, @harupy)
- [Tracking] Claude Code SDK autologging support (#18022, @smoorjani)
- [Tracing] Add support for reading trace configuration from environment variables (#17792, @joelrobin18)
- [Tracking] Mistral tracing improvements (#16370, @joelrobin18)
- [Tracking] Gemini token count tracking (#16248, @joelrobin18)
- [Tracking] Gemini streaming support (#16249, @joelrobin18)
- [Tracking] CrewAI token count tracking with documentation updates (#16373, @joelrobin18)
- [Evaluation] Allow passing empty scorer list for manual result comparison (#18265, @B-Step62)
- [Evaluation] Log assessments to DSPy evaluation traces (#18136, @B-Step62)
- [Evaluation] Add support for trace inputs to built-in scorers (#17943, @BenWilson2)
- [Evaluation] Add synonym handling for built-in scorers (#17980, @BenWilson2)
- [Evaluation] Add span timing tool for Agent Judges (#17948, @BenWilson2)
- [Evaluation] Allow disabling evaluation sample check (#18032, @B-Step62)
- [Evaluation] Reduce verbosity of SIMBA optimizer logs when aligning judges (#17795, @BenWilson2)
- [Evaluation] Add `__repr__` method for Judges (#17794, @BenWilson2)
- [Prompts] Add prompt registry support to MLflow webhooks (#17640, @harupy)
- [Prompts] Prompt Registry Chat UI (#17334, @joelrobin18)
- [UI] Delete parent and child runs together (#18052, @joelrobin18)
- [UI] Added move to top, move to bottom for charts (#17742, @joelrobin18)
- [Tracking] Use sampling data for run comparison to improve performance (#17645, @lkuo)
- [Tracking] Add optional 'outputs' column for evaluation dataset records (#17735, @WeichenXu123)

### Bug Fixes

- [Tracing] Fix parent run resolution mechanism for LangChain (#17273, @B-Step62)
- [Tracing] Add client-side retry for `get_trace` to improve reliability (#18224, @B-Step62)
- [Tracing] Fix OpenTelemetry dual export (#18163, @B-Step62)
- [Tracing] Suppress false warnings from span logging (#18092, #18276, @B-Step62)
- [Tracing] Fix OpenTelemetry resource attributes not propagating correctly (#18019, @xiaosha007)
- [Tracing] Fix DSPy prompt display (#17988, @B-Step62)
- [Tracing] Fix usage aggregation to avoid ancestor duplication (#17921, @TomeHirata)
- [Tracing] Fix double counting in Strands tracing (#17855, @joelrobin18)
- [Tracing] Fix `to_predict_fn` to handle traces without tags field (#17784, @harupy)
- [Tracing] URL-encode trace tag keys in `delete_trace_tag` to prevent 404 errors (#18232, @copilot-swe-agent)
- [Tracking] Fix Claude Code autologging inputs not displaying (#17858, @smoorjani)
- [Tracking] Fix runs with 0-valued metrics not appearing in experiment list contour plots (#17916, @WeichenXu123)
- [Tracking] Fix DSPy run display (#18137, @B-Step62)
- [Tracking] Allow list of types in tools JSON Schema for OpenAI autolog (#17908, @fedem96)
- [Tracking] Set tracking URI environment variable for job runner (#18073, @WeichenXu123)
- [Evaluation] Add atomicity to `job_start` API (#18226, @BenWilson2)
- [Evaluation] Fix trace ingest for outputs in `merge_records()` API (#18047, @BenWilson2)
- [Evaluation] Fix judge regression (#18039, @B-Step62)
- [Evaluation] Fix judges to use non-empty user messages for Anthropic model compatibility (#17935, @dbczumar)
- [Evaluation] Fix endpoints error in judge (#18048, @joelrobin18)
- [Model Registry] Fix creating model versions from non-Databricks tracking to Databricks Unity Catalog registry (#18244, @austinwarner-8451)
- [Model Registry] Fix registry URI instantiation for artifact download (#17982, @arpitjasa-db)
- [Model Registry] Include original error details in Unity Catalog model copy failure messages (#17997, @harupy)
- [Model Registry] Fix webhook delivery to exit early for FileStore instances (#18015, @copilot-swe-agent)
- [Prompts] Fix error suppression during prompt alias resolution when `allow_missing` is set (#17541, @mr-brobot)
- [UI] General UI improvements (#18281, @joelrobin18)
- [Models] Fix dataset issue (#18081, @joelrobin18)
- [Models] Forward dataset name and digest to PolarsDataset's `to_evaluation_dataset` method (#17886, @sadelcarpio)
- [Build] Fix `mlflow server` exiting immediately when optional `huey` package is missing (#18016, @harupy)
- [Scoring] Fix chat completion arguments (#18248, @aravind-segu)

### Documentation Updates

- [Docs] Add self-hosted documentation support (#17986, @B-Step62)
- [Docs] Add GitHub feature requests section to GenAI documentation (#18342, @TomeHirata)
- [Docs] Update Claude Code SDK tracing documentation (#18026, @smoorjani)
- [Docs] Add documentation for Analyze Experiment MCP/CLI command (#17978, @nsthorat)
- [Docs] Add deprecation notice for custom prompt judge (#18287, @smoorjani)
- [Docs] Overhaul scorer documentation (#17930, @B-Step62)
- [Docs] Add default optimizer documentation (#17814, @BenWilson2)
- [Docs] Update TypeScript SDK contribution documentation (#17995, @joelrobin18)
- [Docs] Fix Postgres 18+ mount path in documentation (#18192, @soyun11)
- [Docs] Fix typo: correct variable name from `max_few_show_examples` to `max_few_shot_examples` (#18246, @srinathmkce)
- [Docs] Replace single quotes with double quotes for Windows compatibility (#18266, @PavithraNelluri)
- [Docs] Fix typo in model registry documentation (#18038, @EddieMG)

Small bug fixes and documentation updates:

#18349, #18338, #18241, #18319, #18309, #18292, #18280, #18239, #18236, #17786, #18003, #17970, #17898, #17765, #17667, @serena-ruan; #18346, #17882, @dbrx-euirim; #18306, #18208, #18165, #18110, #18109, #18108, #18107, #18105, #18104, #18100, #18099, #18155, #18079, #18082, #18078, #18077, #18083, #18030, #18001, #17999, #17712, #17785, #17756, #17729, #17731, #17733, @daniellok-db; #18339, #18291, #18222, #18210, #18124, #18101, #18054, #18053, #18007, #17922, #17823, #17822, #17805, #17789, #17750, #17752, #17760, #17758, #17688, #17689, #17693, #17675, #17673, #17656, #17674, @harupy; #18331, #18308, #18303, #18146, @smoorjani; #18315, #18279, #18310, #18187, #18225, #18277, #18193, #18223, #18209, #18200, #18178, #17574, #18021, #18006, #17944, @B-Step62; #18290, #17946, #17627, @bbqiu; #18274, @Ninja3047; #18204, #17868, #17866, #17833, #17826, #17835, @TomeHirata; #18273, #18043, #17928, #17931, #17936, #17937, @dbczumar; #18185, #18180, #18174, #18170, #18167, #18164, #18168, #18166, #18162, #18160, #18159, #18157, #18156, #18154, #18148, #18145, #18135, #18143, #18142, #18139, #18132, #18130, #18119, #18117, #18115, #18102, #18075, #18046, #18062, #18042, #18051, #18036, #18027, #18014, #18011, #18009, #18004, #17903, #18000, #18002, #17973, #17993, #17989, #17984, #17968, #17966, #17967, #17962, #17977, #17976, #17972, #17965, #17964, #17963, #17969, #17971, #17939, #17926, #17924, #17915, #17911, #17912, #17904, #17902, #17900, #17897, #17892, #17889, #17888, #17885, #17884, #17878, #17874, #17873, #17871, #17870, #17865, #17860, #17861, #17859, #17857, #17856, #17854, #17853, #17851, #17849, #17850, #17847, #17845, #17846, #17844, #17843, #17842, #17838, #17836, #17834, #17831, #17824, #17828, #17819, #17825, #17817, #17821, #17809, #17807, #17808, #17803, #17800, #17799, #17797, #17793, #17790, #17772, #17771, #17769, #17770, #17753, #17762, #17747, #17749, #17745, #17740, #17734, #17732, #17726, #17723, #17722, #17721, #17719, #17720, #17718, #17716, #17713, #17715, #17710, #17709, #17708, #17707, #17705, #17697, #17701, #17698, #17696, #17695, @copilot-swe-agent; #18151, #18153, #17983, #18040, #17981, #17841, #17818, #17776, #17781, @BenWilson2; #18068, @alkispoly-db; #18133, @kevin-lyn; #17105, #17717, @joelrobin18; #17879, @lkuo; #17996, #17945, #17913, @WeichenXu123

## 3.5.0rc0 (2025-10-08)

MLflow 3.5.0rc0 includes several major features and improvements

Major new features:

- 🤖 **Tracing support for Claude Code SDK**: MLflow now provides a tracing integration for both the Claude Code CLI and SDK! Configure the autologging integration to track your prompts, Claude's responses, tool calls, and more. Check out this [doc page](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/claude_code/) to get started. (#18022, @smoorjani)
- ✨ **Improved UI homepage**: The MLflow UI's homepage has been updated to help you get started with more of our latest features. This page will be updated regularly moving forward, allowing you to get more in-product guidance.
- 🗂️ **Evaluation datasets UI integration**: In MLflow 3.4.0, we released backend support for creating evaluation datasets for GenAI applications. In this release, we've added a new tab to the MLflow Experiment UI, allowing you to create, manage, and export traces to your datasets without having to write a line of code.
- 🧮 **GEPA support for prompt optimization**: MLflow's prompt optimization feature now supports the [GEPA algorithm](https://dspy.ai/api/optimizers/GEPA/overview/), allowing you to achieve higher performing prompts with less rollouts. For instructions on how to get started with prompt optimization, visit this [doc page](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)!
- 🔐 **Security middleware layer for tracking server**: MLflow now ships with a security middleware layer by default, allowing you to protect against DNS rebinding, CORS attacks, and more. Read the documentation [here](https://mlflow.org/docs/latest/self-hosting/security/network/) to learn how to configure these options.

Stay tuned for the full release, which will be packed with more features and bugfixes.

To try out this release candidate, please run:

`pip install mlflow==3.5.0rc0`

## 3.4.0rc0 (2025-09-11)

MLflow 3.4.0rc0 includes several major features and improvements

### Major New Features

- 📊 **OpenTelemetry Metrics Export**: MLflow now exports span-level statistics as OpenTelemetry metrics, providing enhanced observability and monitoring capabilities for traced applications. (#17325, @dbczumar)
- 🤖 **MCP Server Integration**: Introducing the Model Context Protocol (MCP) server for MLflow, enabling AI assistants and LLMs to interact with MLflow programmatically. (#17122, @harupy)
- 🧑‍⚖️ **Custom Judges API**: New `make_judge` API enables creation of custom evaluation judges for assessing LLM outputs with domain-specific criteria. (#17647, @BenWilson2, @dbczumar, @alkispoly-db, @smoorjani)
- 📈 **Correlations Backend**: Implemented backend infrastructure for storing and computing correlations between experiment metrics using NPMI (Normalized Pointwise Mutual Information). (#17309, #17368, @BenWilson2)
- 🗂️ **Evaluation Datasets**: MLflow now supports storing and versioning evaluation datasets directly within experiments for reproducible model assessment. (#17447, @BenWilson2)
- 🔗 **Databricks Backend for MLflow Server**: MLflow server can now use Databricks as a backend, enabling seamless integration with Databricks workspaces. (#17411, @nsthorat)
- 🤖 **Claude Autologging**: Automatic tracing support for Claude AI interactions, capturing conversations and model responses. (#17305, @smoorjani)
- 🌊 **Strands Agent Tracing**: Added comprehensive tracing support for Strands agents, including automatic instrumentation for agent workflows and interactions. (#17151, @joelrobin18)
- 🧪 **Experiment Types in UI**: MLflow now introduces experiment types, helping reduce clutter between classic ML/DL and GenAI features. MLflow auto-detects the type, but you can easily adjust it via a selector next to the experiment name. (#17605, @daniellok-db)

Features:

- [Evaluation] Add ability to pass tags via dataframe in mlflow.genai.evaluate (#17549, @smoorjani)
- [Evaluation] Add custom judge model support for Safety and RetrievalRelevance builtin scorers (#17526, @dbrx-euirim)
- [Tracing] Add AI commands as MCP prompts for LLM interaction (#17608, @nsthorat)
- [Tracing] Add MLFLOW_ENABLE_OTLP_EXPORTER environment variable (#17505, @dbczumar)
- [Tracing] Support OTel and MLflow dual export (#17187, @dbczumar)
- [Tracing] Make set_destination use ContextVar for thread safety (#17219, @B-Step62)
- [CLI] Add MLflow commands CLI for exposing prompt commands to LLMs (#17530, @nsthorat)
- [CLI] Add 'mlflow runs link-traces' command (#17444, @nsthorat)
- [CLI] Add 'mlflow runs create' command for programmatic run creation (#17417, @nsthorat)
- [CLI] Add MLflow traces CLI command with comprehensive search and management capabilities (#17302, @nsthorat)
- [CLI] Add --env-file flag to all MLflow CLI commands (#17509, @nsthorat)
- [Tracking] Backend for storing scorers in MLflow experiments (#17090, @WeichenXu123)
- [Model Registry] Allow cross-workspace copying of model versions between WMR and UC (#17458, @arpitjasa-db)
- [Models] Add automatic Git-based model versioning for GenAI applications (#17076, @harupy)
- [Models] Improve WheeledModel.\_download_wheels safety (#17004, @serena-ruan)
- [Projects] Support resume run for Optuna hyperparameter optimization (#17191, @lu-wang-dl)
- [Scoring] Add MLFLOW_DEPLOYMENT_CLIENT_HTTP_REQUEST_TIMEOUT environment variable (#17252, @dbczumar)
- [UI] Add ability to hide/unhide all finished runs in Chart view (#17143, @joelrobin18)
- [Telemetry] Add MLflow OSS telemetry for invoke_custom_judge_model (#17585, @dbrx-euirim)

Bug fixes:

- [Evaluation] Implement DSPy LM interface for default Databricks model serving (#17672, @smoorjani)
- [Evaluation] Fix aggregations incorrectly applied to legacy scorer interface (#17596, @BenWilson2)
- [Evaluation] Add Unity Catalog table source support for mlflow.evaluate (#17546, @BenWilson2)
- [Evaluation] Fix custom prompt judge encoding issues with custom judge models (#17584, @dbrx-euirim)
- [Tracking] Fix OpenAI autolog to properly reconstruct Response objects from streaming events (#17535, @WeichenXu123)
- [Tracking] Add basic authentication support in TypeScript SDK (#17436, @kevin-lyn)
- [Tracking] Update scorer endpoints to v3.0 API specification (#17409, @WeichenXu123)
- [Tracking] Fix scorer status handling in MLflow tracking backend (#17379, @WeichenXu123)
- [Tracking] Fix missing source-run information in UI (#16682, @WeichenXu123)
- [Scoring] Fix spark_udf to always use stdin_serve for model serving (#17580, @WeichenXu123)
- [Scoring] Fix a bug with Spark UDF usage of uv as an environment manager (#17489, @WeichenXu123)
- [Model Registry] Extract source workspace ID from run_link during model version migration (#17600, @arpitjasa-db)
- [Models] Improve security by reducing write permissions in temporary directory creation (#17544, @BenWilson2)
- [Server-infra] Fix --env-file flag compatibility with --dev mode (#17615, @nsthorat)
- [Server-infra] Fix basic authentication with Uvicorn server (#17523, @kevin-lyn)
- [UI] Fix experiment comparison functionality in UI (#17550, @Flametaa)
- [UI] Fix compareExperimentsSearch route definitions (#17459, @WeichenXu123)

Documentation updates:

- [Docs] Add clarification for trace requirements in scorers documentation (#17542, @BenWilson2)
- [Docs] Add documentation for Claude code autotracing (#17521, @smoorjani)
- [Docs] Remove experimental status message for MPU/MPD features (#17486, @BenWilson2)
- [Docs] Remove problematic pages from documentation (#17453, @BenWilson2)
- [Docs] Add documentation for updating signatures on Databricks registered models (#17450, @arpitjasa-db)
- [Docs] Update Scorers API documentation (#17298, @WeichenXu123)
- [Docs] Add comprehensive documentation for scorers (#17258, @B-Step62)

Small bug fixes and documentation updates:

#17655, #17657, #17597, #17545, #17547, @BenWilson2; #17671, @smoorjani; #17668, #17665, #17662, #17661, #17659, #17658, #17653, #17643, #17642, #17636, #17634, #17631, #17628, #17611, #17607, #17588, #17570, #17575, #17564, #17557, #17556, #17555, #17536, #17531, #17524, #17510, #17511, #17499, #17500, #17494, #17493, #17490, #17488, #17478, #17479, #17425, #17471, #17457, #17440, #17403, #17405, #17404, #17402, #17366, #17346, #17344, #17337, #17316, #17313, #17284, #17276, #17235, #17226, #17229, @copilot-swe-agent; #17664, #17654, #17613, #17637, #17633, #17612, #17630, #17616, #17626, #17617, #17610, #17614, #17602, #17538, #17522, #17512, #17508, #17492, #17462, #17475, #17468, #17455, #17338, #17257, #17231, #17214, #17223, #17218, #17216, @harupy; #17635, #17663, #17426, #16870, #17428, #17427, #17441, #17377, @serena-ruan; #17605, #17306, @daniellok-db; #17624, #17578, #17369, #17391, #17072, #17326, #17115, @dbczumar; #17598, #17408, #17353, @nsthorat; #17601, #17553, @dbrx-euirim; #17586, #17587, #17310, #17180, @TomeHirata; #17516, @bbqiu; #17477, #17474, @WeichenXu123; #17449, @raymondzhou-db; #17470, @jacob-danner; #17378, @arpitjasa-db; #17121, @ctaymor; #17351, #17322, @ispoljari; #17292, @dsuhinin; #17287, #17281, #17230, #17245, #17237, @B-Step62

## 3.3.2 (2025-08-27)

MLflow 3.3.2 is a patch release that includes several minor improvements and bugfixes

Features:

- [Evaluation] Add support for dataset name persistence (#17250, @BenWilson2)

Bug fixes:

- [Tracing] Add retry policy support to `_invoke_litellm` for improved reliability (#17394, @dbczumar)
- [UI] fix ui sorting in experiments (#17340, @Flametaa)
- [Serving] Add Databricks Lakebase Resource (#17277, @jennsun)
- [Tracing] Fix set trace tags endpoint (#17362, @daniellok-db)

Documentation updates:

- [Docs] Add docs for package lock (#17395, @BenWilson2)
- [Docs] Fix span processor docs (#17386, @mr-brobot)

Small bug fixes and documentation updates:

#17301, #17299, @B-Step62; #17420, #17421, #17398, #17397, #17349, #17361, #17377, #17359, #17358, #17356, #17261, #17263, #17262, @serena-ruan; #17422, #17310, #17357, @TomeHirata; #17406, @sotagg; #17418, @annzhang-db; #17384, #17376, @daniellok-db

## 3.3.1 (2025-08-20)

MLflow 3.3.1 includes several major features and improvements

Bug fixes:

- [Tracking] Fix `mlflow.genai.datasets` attribute (#17307, @WeichenXu123)
- [UI] Fix tag display as column in experiment overview (#17296, @joelrobin18)
- [Tracing] Fix the slowness of dspy tracing (#17290, @TomeHirata)

Small bug fixes and documentation updates:

#17295, @gunsodo; #17272, @bbqiu

## 3.3.0 (2025-08-19)

MLflow 3.3.0 includes several major features and improvements

### Major new features:

- 🪝 **Model Registry Webhooks**: MLflow now supports [webhooks](https://mlflow.org/docs/latest/ml/webhooks/) for model registry events, enabling automated notifications and integrations with external systems. (#16583, @harupy)
- 🧭 **Agno Tracing Integration**: Added [Agno tracing integration](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/agno/) for enhanced observability of AI agent workflows. (#16995, @joelrobin18)
- 🧪 **GenAI Evaluation in OSS**: MLflow open-sources [the new evaluation capability for LLM applications](https://mlflow.org/docs/latest/genai/eval-monitor/). This suite enables systematic measurement and improvement of LLM application quality, with tight integration into MLflow's observability, feedback collection, and experiment tracking capabilities. (#17161, #17159, @B-Step62)
- 🖥️ **Revamped Trace Table View**: The new trace view in MLflow UI provides a streamlined interface for exploring, filtering, and monitoring traces, with enhanced search capabilities including full-text search across requests.(#17092, @daniellok-db)
- ⚡️ **FastAPI + Uvicorn Server**: MLflow Tracking Server now defaults to FastAPI + Uvicorn for improved performance, while maintaining Flask compatibility. (#17038, @dbczumar)

New features:

- [Tracking] Add a Docker compose file to quickly start a local MLflow server with recommended minimum setup (#17065, @joelrobin18)
- [Tracing] Add `memory` span type for agentic workflows (#17034, @B-Step62)
- [Prompts] Enable custom prompt optimizers in `optimize_prompt` including DSPy support (#17052, @TomeHirata)
- [Model Registry / Prompts] Proper support for the @latest alias (#17146, @B-Step62)
- [Metrics] Allow custom tokenizer encoding in `token_count` function (#16253, @joelrobin18)

Bug fixes:

- [Tracking] Fix Databricks secret scope check to reduce audit log errors (#17166, @harupy)
- [Tracking] Fix Databricks SDK error code mapping in retry logic (#17095, @harupy)
- [Tracking] Fix Databricks secret scope check to reduce error rates (#17166, @harupy)
- [Tracing] Remove API keys from CrewAI traces to prevent credential leakage (#17082, @diy2learn)
- [Tracing] Fix LiteLLM span association issue by making callbacks synchronous (#16982, @B-Step62)
- [Tracing] Fix OpenAI Agents tracing (#17227, @B-Step62)
- [Evaluation] Fix issue with get_label_schema has no attribute (#17163, @smoorjani)
- [Docs] Fix version selector on API Reference page by adding missing CSS class and versions.json generation (#17247, @copilot-swe-agent)

Documentation updates:

- [Docs] Document custom optimizer usage with `optimize_prompt` (#17084, @TomeHirata)
- [Docs] Fix built-in scorer documentation for expectation parameter (#17075, @smoorjani)
- [Docs] Add comprehensive documentation for scorers (#17258, @B-Step62)

Small bug fixes and documentation updates:

#17230, #17264, #17289, #17287, #17265, #17238, #17215, #17224, #17185, #17148, #17193, #17157, #17067, #17033, #17087, #16973, #16875, #16956, #16959, @B-Step62; #17269, @BenWilson2; #17285, #17259, #17260, #17236, #17196, #17169, #17062, #16943, @serena-ruan; #17253, @sotagg; #17212, #17206, #17211, #17207, #17205, #17118, #17177, #17182, #17170, #17153, #17168, #17123, #17136, #17119, #17125, #17088, #17101, #17056, #17077, #17057, #17036, #17018, #17024, #17019, #16883, #16972, #16961, #16968, #16962, #16958, @harupy; #17209, #17202, #17184, #17179, #17174, #17141, #17155, #17145, #17130, #17113, #17110, #17098, #17104, #17100, #17060, #17044, #17032, #17008, #17001, #16994, #16991, #16984, #16976, @copilot-swe-agent; #17069, @hayescode; #17199, #17081, #16928, #16931, @TomeHirata; #17198, @WeichenXu123; #17195, #17192, #17131, #17128, #17124, #17120, #17102, #17093, #16941, @daniellok-db; #17070, #17074, #17073, @dbczumar

## 3.2.0 (2025-08-05)

MLflow 3.2.0 includes several major features and improvements

### Major New Features

- 🧭 **Tracing TypeScript SDK**: MLflow Tracing now supports the [TypeScript SDK](https://github.com/mlflow/mlflow/tree/master/libs/typescript), allowing developers to trace GenAI applications in TypeScript environments. (#16871, @B-Step62)
- 🔗 **Semantic Kernel Tracing**: MLflow now provides [automatic tracing support for Semantic Kernel](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/semantic_kernel/), simplifying trace capture for SK-based workflows. (#16469, @michael-berk)
- 🧪 **Feedback Tracking**: MLflow OSS now natively supports tracking [human feedbacks](https://mlflow.org/docs/latest/genai/assessments/feedback/), [ground truths](https://mlflow.org/docs/latest/genai/assessments/expectations/), LLM judges on traces, providing integrated quality monitoring and feedback management capabilities. (#16743, @BenWilson2)
- 🖥️ **MLflow UI Improvements**: The MLflow UI now features **a redesigned experiment home view** and includes enhancements like pagination on the model page for better usability. (#16464, @frontsideair, #15801, @Flametaa)
- 🔍 **Updated Trace UI**: The Trace UI now has image support when rendering chat messages for OpenAI, Langchain, and Anthropic! Additionally, we're introducing a "summary view" which is a simplified, flat representation of the important spans in a trace. The full detail view is still available in a separate tab.
- 🛡️ **PII Masking in Tracing**: Added support for [masking personally identifiable information (PII) via a custom span post-processor](https://mlflow.org/docs/latest/genai/tracing/observe-with-traces/masking). (#16344, @B-Step62)
- 🐻‍❄️ **Polars Dataset Support**: MLflow now supports [Polars datasets](https://mlflow.org/docs/latest/ml/dataset/#dataset), expanding compatibility with performant DataFrame libraries. (#13006, @AlpAribal)

### 📊 Usage Tracking (New in 3.2.0)

- Starting with version 3.2.0, MLflow will begin collecting anonymized usage data about how core features of the platform are used. This data contains **no sensitive or personally identifiable information**, and users can opt out of data collection at any time. Check [MLflow documentation](https://mlflow.org/docs/latest/community/usage-tracking/) for more details. (#16439, @serena-ruan)

Features:

- [Tracing] Include mlflow-tracing as a dependency of mlflow (#16589, @B-Step62)
- [Tracing] Convert DatabricksRM output to MLflow document format (#16866, @WeichenXu123)
- [Tracing] Add unified token usage tracking for Bedrock LLMs (#16351, @mohammadsubhani)
- [Tracing] Token usage tracking for agent frameworks including Anthropic, Autogen, LlamaIndex etc. (#16251, #16362, #16246, #16258, #16313, #16312, #16340, #16357, #16358, @joelrobin18, #16387, @sanatb187)
- [Tracing] Render multi-modal trace for LangChain (#16799, @B-Step62)
- [Tracing] Support async tracing for Gemini (#16632, @B-Step62)
- [Tracing] Support global sampling for tracing (#16700, @B-Step62)
- [Tracing] ResponsesAgent tracing aggregation (#16787, @bbqiu)
- [Tracing] Add Agent and LLM complete name (#16613, @joelrobin18)
- [Tracking] Allow setting thread-local tracing destination via mlflow.tracing.set_destination (#16859, @WeichenXu123)
- [Tracking] Introduce MLFLOW_DISABLE_SCHEMA_DETAILS environment variable to toggle detailed schema errors (#16631, @NJAHNAVI2907)
- [Tracking] Add support for chat-style prompts with structured output with prompt object (#16341, @harshilprajapati96)
- [Tracking] Add support for responses.parse calls in oai autologger (#16245, @dipakkrishnan)
- [Tracking] Add support for uv as an environment manager in mlflow run (#16274, @isuyyy)
- [Evaluation] Replace guideline_adherence to guidelines (#16856, @smoorjani)
- [Evaluation] Replace Scheduled Scorers API to a Scorer Registration System (#16977, @dbrx-euirim)
- [UI] Add tag filter to the experiments page (#16648, @frontsideair)
- [UI] Add ability to the UI to edit experiment tags (#16614, @frontsideair)
- [UI] Create runs table using selected columns in the experiment view (#16804, @wangh118)
- [Scoring] Make spark_udf support 'uv' env manager (#16292, @WeichenXu123)

Bug fixes:

- [Tracking / UI] Add missing default headers and replace absolute URLs in new browser client requests (GraphQL & logged models) (#16840, @danilopeixoto)
- [Tracking] Fix tracking_uri positional argument bug in artifact repositories (#16878, @copilot-swe-agent)
- [Models] Fix UnionType support for Python 3.10 style union syntax (#16882, @harupy)
- [Tracing / Tracking] Fix OpenAI autolog Pydantic validation for enum values (#16862, @mohammadsubhani)
- [Tracking] Fix tracing for Anthropic and Langchain combination (#15151, @maver1ck)
- [Models] Fix OpenAI multimodal message logging support (#16795, @mohammadsubhani)
- [Tracing] Avoid using nested threading for Azure Databricks trace export (#16733, @TomeHirata)
- [Evaluation] Bug fix: Databricks GenAI evaluation dataset source returns string, instead of DatasetSource instance (#16712, @dbczumar)
- [Models] Fix `get_model_info` to provide logged model info (#16713, @harupy)
- [Evaluation] Fix serialization and deserialization for python scorers (#16688, @connorchenn)
- [UI] Fix GraphQL handler erroring on NaN metric values (#16628, @daniellok-db)
- [UI] Add back video artifact preview (#16620, @daniellok-db)
- [Tracing] Proper chat message reconstruction from OAI streaming response (#16519, @B-Step62)
- [Tracing] Convert trace column in search_traces() response to JSON string (#16523, @B-Step62)
- [Evaluation] Fix mlflow.evaluate crashes in \_get_binary_classifier_metrics due to … (#16485, @mohammadsubhani)
- [Evaluation] Fix trace detection logic for `mlflow.genai.evaluate` (#16932, @B-Step62)
- [Evaluation] Enable to use make_genai_metric_from_prompt for mlflow.evaluate (#16960, @TomeHirata)
- [Models] Add explicit encoding for decoding streaming Responses (#16855, @aravind-segu)
- [Tracking] Prevent from tracing DSPy model API keys (#17021, @czyzby)
- [Tracking] Fix pytorch datetime issue (#17030, @serena-ruan)
- [Tracking] Fix predict with pre-releases (#16998, @serena-ruan)

Documentation updates:

- [Docs] Overhaul of top level version management GenAI docs (#16728, @BenWilson2)
- [Docs] Fix Additional GenAI Docs pages (#16691, @BenWilson2)
- [Docs] Update the docs selector dropdown (#16280, @BenWilson2)
- [Docs] Update docs font sizing and link coloring (#16281, @BenWilson2)
- [Docs] Fix typo in model deployment page (#16999, @premkiran-o7)

Small bug fixes and documentation updates:

#17003, #17049, #17035, #17026, #16981, #16971, #16953, #16930, #16917, #16738, #16717, #16693, #16694, #16684, #16678, #16656, #16513, #16459, #16277, #16276, #16275, #16170, #16217, @serena-ruan; #16927, #16915, #16913, #16911, #16909, #16889, #16727, #16600, #16543, #16551, #16526, #16533, #16535, #16531, #16472, #16392, #16389, #16385, #16376, #16369, #16367, #16321, #16311, #16307, #16273, #16268, #16265, #16112, #16243, #16231, #16226, #16221, #16196, @copilot-swe-agent; #17050, #17048, #16955, #16894, #16885, #16860, #16841, #16835, #16801, #16701, @daniellok-db; #16898, #16881, #16858, #16735, #16823, #16814, #16647, #16750, #16809, #16794, #16793, #16789, #16780, #16770, #16773, #16771, #16772, #16768, #16752, #16754, #16751, #16748, #16730, #16729, #16346, #16709, #16704, #16703, #16702, #16658, #16662, #16645, #16639, #16640, #16626, #16572, #16566, #16565, #16563, #16561, #16559, #16544, #16539, #16520, #16508, #16505, #16494, #16495, #16491, #16487, #16482, #16473, #16465, #16456, #16458, #16394, #16445, #16433, #16434, #16413, #16417, #16416, #16414, #16415, #16378, #16350, #16323, #15788, #16263, #16256, #16237, #16234, #16219, #16216, #16207, #16199, #16192, #16705, @harupy; #17047, #17017, #17005, #16989, #16952, #16951, #16903, #16900, #16755, #16762, #16757, #15860, #16661, #16630, #16657, #16605, #16602, #16568, #16569, #16553, #16345, #16454, #16489, #16486, #16438, #16266, #16382, #16381, #16303, @B-Step62; #17028, #17027, #17020, @he7d3r; #16969, #16957, #16852, #16829, #16816, #16808, #16775, #16807, #16806, #16624, #16524, #16410, #16403, @TomeHirata; #16987, @wangh118; #16760, #16761, #16736, #16737, #16699, #16718, #16663, #16676, #16574, #16477, #16552, #16527, #16515, #16452, #16210, #16204, #16610, @frontsideair; #16723, #16124, @AveshCSingh; #16744, @BenWilson2; #16683, @dsuhinin; #16877, #16502, @bbqiu; #16619, @AchimGaedkeLynker; #16595, @Aiden-Jeon; #16480, #16479, @shushantrishav; #16398, #16331, #16328, #16329, #16293, @WeichenXu123

## 3.1.4 (2025-07-23)

MLflow 3.1.4 includes several major features and improvements

Small bug fixes and documentation updates:

#16835, #16820, @daniellok-db

## 3.1.3 (2025-07-22)

MLflow 3.1.3 includes several major features and improvements

Features:

- [Artifacts / Tracking] Do not copy file permissions when logging artifacts to local artifact repo (#16642, @connortann)
- [Tracking] Add support for OpenAI ChatCompletions parse method (#16493, @harupy)

Bug fixes:

- [Deployments] Propagate `MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT` to databricks-sdk (#16783, @bbqiu)
- [Model Registry] Fix issue with search_registered_models with Databricks UC backend not supporting filter_string (#16766, @BenWilson2)
- [Evaluation] Bug fix: Databricks GenAI evaluation dataset source returns string, instead of DatasetSource instance (#16712, @dbczumar)
- [Tracking] Fix the position of added tracking_uri param to artifact store implementations (#16653, @BenWilson2)

Small bug fixes and documentation updates:

#16786, #16692, @daniellok-db; #16594, @ngoduykhanh; #16475, @harupy

## 3.1.2 (2025-07-08)

MLflow 3.1.2 is a patch release that includes several bug fixes.

Bug fixes:

- [Tracking] Fix `download_artifacts` ignoring `tracking_uri` parameter (#16461, @harupy)
- [Models] Fix event type for ResponsesAgent error (#16427, @bbqiu)
- [Models] Remove falsey chat conversion for LangGraph models (#16601, @B-Step62)
- [Tracing] Use empty Resource when instantiating OTel provider to fix LiteLLM tracing issue (#16590, @B-Step62)

Small fixes and documentation updates:

#16568, #16454, #16617, #16605, #16569, #16553, #16625, @B-Step62; #16571, #16552, #16452, #16395, #16446, #16420, #16447, #16554, #16515, @frontsideair; #16558, #16443, #16457, @16442, #16449, @harupy; #16509, #16512, #16524, #16514, #16607, @TomeHirata; #16541, @copilot-swe-agent; #16427, @bbqiu; #16573, @daniellok-db; #16470, #16281, @BenWilson2

## 3.1.1 (2025-06-25)

MLflow 3.1.1 includes several major features and improvements

Features:

- [Model Registry / Sqlalchemy] Increase prompt text limit from 5K to 100K (#16377, @harupy)
- [Tracking] Support pagination in get-history of FileStore and SqlAlchemyStore (#16325, @TomeHirata)

Bug fixes:

- [Artifacts] Support downloading logged model artifacts (#16356, @TomeHirata)
- [Models] Fix bedrock provider, configured inference profile compatibility (#15604, @lloydhamilton)
- [Tracking] Specify attribute.run_id when search_traces filters by run_id (#16295, @artjen)
- [Tracking] Fix graphql batching attacks (#16227, @serena-ruan)
- [Model Registry] Make the chunk size configurable in DatabricksSDKModelsArtifactRepository (#16247, @TomeHirata)

Documentation updates:

- [Docs] Move the Lighthouse main signup page to GenAI (#16404, @BenWilson2)
- [Docs] [DOC-FIX] Dspy doc fix (#16397, @joelrobin18)
- [Docs] Fix(docs): Resolve self-referencing 'Next' link on GenAI Tracing overview page (#16334, @mohammadsubhani)
- [Docs] Update the docs selector dropdown (#16280, @BenWilson2)
- [Docs] Update utm_source for source tracking to signup URL (#16316, @BenWilson2)
- [Docs] Fix footer rendering in docs for light mode display (#16214, @BenWilson2)

Small bug fixes and documentation updates:

#16261, @rohitarun-db; #16411, #16352, #16327, #16324, #16279, #16193, #16197, @harupy; #16409, #16348, #16347, #16290, #16286, #16283, #16271, #16223, @TomeHirata; #16326, @mohammadsubhani; #16364, @BenWilson2; #16308, #16218, @serena-ruan; #16262, @raymondzhou-db; #16191, @copilot-swe-agent; #16212, @B-Step62; #16208, @frontsideair; #16205, #16200, #16198, @daniellok-db

## 3.0.1 (2025-06-25)

MLflow 3.0.1 includes several major features and improvements

Features:

- [Model Registry / Sqlalchemy] Increase prompt text limit from 5K to 100K (#16377, @harupy)

Bug fixes:

- [Models] Fix bedrock provider, configured inference profile compatibility (#15604, @lloydhamilton)

Small bug fixes and documentation updates:

#16364, @BenWilson2; #16347, @TomeHirata; #16279, #15835, @harupy; #16182, @B-Step62

## 3.1 (2025-06-11)

MLflow 3 includes several major features and improvements

Features:

- [Tracking] MLflow 3.0 (#13211, @harupy)
- [Prompts] Add Custom Prompt Judges to `mlflow[databricks]` (#16097, @dbrx-euirim)
- [Artifacts / Model Registry / Tracking] Package model environment when registering model (#15783, @qyc)
- [Tracking] Add `MlflowSparkStudy` (#15418, @lu-wang-dl)
- [Scoring] Make `spark_udf` support DBConnect + DBR 15.4 / DBR dedicated cluster (#15968, @WeichenXu123)
- [Tracking] Lock model dependencies when logging a model using `uv` (#15875, @harupy)
- [Model Registry] Introduce `mlflow.genai.optimize_prompt` to optimize prompts (#15861, @TomeHirata)
- [Tracing] Support custom request/response preview (#15919, @B-Step62)
- [Tracking] Add integration for AutoGen > 0.4 (#14729, @TomeHirata)
- [Tracking] Support token tracking for OpenAI (#15870, @B-Step62)
- [Tracking] Support tracing `ResponsesAgent.predict_stream` (#15762, @bbqiu)
- [Tracking] Introduce client and fluent APIs for `LogLoggedModelParams` (#15717, @artjen)
- [Models] Support `predict_stream` in DSPy flavor (#15678, @TomeHirata)
- [Tracking] Record notebook and git metadata in trace metadata (#15650, @B-Step62)
- [Model Registry] Added `search_prompts` function to list all the prompts registered (#15445, @joelrobin18)
- [Models] Support compression for pyfunc log model (#14700, @antbbn)
- [Gateway] Add support for Gemini in AI Gateway (#15069, @joelrobin18)
- [Tracing] PydanticAI Autologging (#15553, @joelrobin18)
- [Tracking] Support setting databricks auth profile by `DATABRICKS_CONFIG_PROFILE` environment variable. (#15587, @WeichenXu123)
- [Tracking] create mlflow tracing for `smolagents` (#15574, @y-okt)
- [Artifacts / UI] Support for video artifacts (#15518, @joelrobin18)
- [Model Registry] Add `allow_missing` parameter in `load_prompt` (#15371, @joelrobin18)
- [Tracking] Emit a warning for `mlflow.get_artifact_uri()` usage outside active run (#12902, @Shashank1202)

Bug fixes:

- [GenAI] Add Databricks App resource (#15867, @aravind-segu)
- [Tracking] Support json-string for inputs/expectations column in Spark Dataframe (#16011, @B-Step62)
- [Tracking] Avoid generating traces from scorers during evaluation (#16004, @B-Step62)
- [GenAI] Allow multi inputs module in DSPy (#15859, @TomeHirata)
- [Tracking] Improve error handling if tracking URI is not set when running `mlflow gc` (#11773, @oleg-z)
- [Tracking] Trace search: Avoid spawning threads for span fetching if `include_spans=False` (#15634, @dbczumar)
- [Tracking] Fix `global_guideline_adherence` (#15572, @artjen)
- [Model Registry] Log `Resources` from `SystemAuthPolicy` in `CreateModelVersion` (#15485, @aravind-segu)
- [Models] `ResponsesAgent` interface update (#15601, #15741, @bbqiu)

Breaking changes:

- [Tracking] Move prompt registry APIs under `mlflow.genai.prompts` namespace (#16174, @B-Step62)
- [Model Registry] Default URI to databricks-uc when tracking URI is databricks & registry URI is unspecified (#16135, @dbczumar)
- [Tracking] Do not log SHAP explainer in `mlflow.evaluate` (#15827, @harupy)
- [Tracking] Update DataFrame schema returned from `mlflow.search_trace()` to be V3 format (#15643, @B-Step62)

Documentation updates:

- [Docs] Documentation revamp for MLflow 3.0 (#15954, @harupy)
- [Docs] Add Prompt Optimization Document Page (#15958, @TomeHirata)
- [Docs] Redesign API reference page (#15811, @besirovic)
- [Docs] MLflow 3 breaking changes list (#15716, @WeichenXu123)
- [Docs] Update Lighthouse signup and signin links (#15740, @BenWilson2)
- [Docs] Document models:/ URIs explicitly in OSS MLflow docs (#15727, @WeichenXu123)
- [Docs] Spark UDF Doc update (#15586, @WeichenXu123)

Small bug fixes and documentation updates:

#16193, #16192, #16171, #16119, #16036, #16130, #16081, #16101, #16047, #16086, #16077, #16045, #16065, #16067, #16063, #16061, #16058, #16050, #16043, #16034, #16033, #15966, #16025, #16015, #16002, #15970, #16001, #15999, #15942, #15960, #15955, #15951, #15939, #15885, #15883, #15890, #15887, #15874, #15869, #15846, #15845, #15826, #15834, #15822, #15830, #15796, #15821, #15818, #15817, #15805, #15804, #15798, #15793, #15797, #15782, #15775, #15772, #15790, #15773, #15776, #15756, #15767, #15766, #15765, #15746, #15747, #15748, #15751, #15743, #15731, #15720, #15722, #15670, #15614, #15715, #15677, #15708, #15673, #15680, #15686, #15671, #15657, #15669, #15664, #15675, #15667, #15666, #15668, #15651, #15649, #15647, #15640, #15638, #15630, #15627, #15624, #15622, #15558, #15610, #15577, #15575, #15545, #15576, #15559, #15563, #15555, #15557, #15548, #15551, #15547, #15542, #15536, #15524, #15531, #15525, #15520, #15521, #15502, #15499, #15442, #15426, #15315, #15392, #15397, #15399, #15394, #15358, #15352, #15349, #15328, #15336, #15335, @harupy; #16196, #16191, #16093, #16114, #16080, #16088, #16053, #15856, #16039, #15987, #16009, #16014, #16007, #15996, #15993, #15991, #15989, #15978, #15839, #15953, #15934, #15929, #15926, #15909, #15900, #15893, #15889, #15881, #15879, #15877, #15865, #15863, #15854, #15852, #15848, @copilot-swe-agent; #16178, #16153, #16155, #15823, #15754, #15794, #15800, #15799, #15615, #15777, #15726, #15752, #15745, #15753, #15738, #15681, #15684, #15682, #15702, #15679, #15623, #15645, #15612, #15533, #15607, #15522, @serena-ruan; #16177, #16167, #16168, #16166, #16152, #16144, #15920, #16134, #16128, #16098, #16059, #16024, #15974, #15917, #15676, #15750, @dbczumar; #16162, #16161, #16137, #16126, #16127, #16099, #16074, #16041, #16040, #16010, #15945, #15697, #15588, #15602, #15581, @rohitarun-db; #16150, #15984, #16125, #16102, #16062, #16060, #15986, #15985, #15983, #15982, #15980, #15763, @smoorjani; #16160, #16149, #16103, #15538, #16055, #16054, #16048, #16012, #16029, #16003, #15940, #15956, #15950, #15906, #15922, #15932, #15930, #15905, #15910, #15902, #15901, #15840, #15896, #15898, #15895, #15850, #15833, #15824, #15819, #15816, #15806, #15803, #15795, #15759, #15791, #15792, #15774, #15769, #15768, #15770, #15755, #15771, #15737, #15690, #15733, #15730, #15687, #15660, #15735, #15688, #15705, #15590, #15663, #15665, #15658, #15594, #15620, #15644, #15648, #15605, #15639, #15642, #15619, #15618, #15611, #15597, #15589, #15580, #15593, #15437, #15584, #15582, #15448, #15351, #15317, #15353, #15320, #15319, @B-Step62; #16151, #16142, #16111, #16106, #16051, #16046, #16044, #15971, #15957, #15810, #15749, #15706, #15683, #15728, #15732, #15707, #15621, #15567, #15566, #15523, #15479, #15404, #15400, #15378, @TomeHirata; #16026, #16072, @AveshCSingh; #15967, @euirim; #15884, #15924, #15395, #15393, #15390, @daniellok-db; #15786, @rahuja23; #15734, @lhrotk; #15809, #15739, #15695, #15654, #15694, #15655, #15653, #15608, #15543, #15573, @dhruyads; #15596, @mrharishkumar; #15742, #15723, #15633, #15606, @ShaylanDias; #15703, #15637, #15613, #15473, @joelrobin18; #15636, #15659, #15616, #15617, @raymondzhou-db; #15674, #15598, #15357, #15586, @WeichenXu123; #15691, @artjen; #15698, @prithvikannan; #15631, @hubertzub-db; #15569, @Anand1923; #15578, @y-okt; #14790, @singh-kristian; #14129, @jamblejoe; #15552, @BenWilson2; #14197, @clarachristiansen; #15505, @Conor0Callaghan; #15509, @tr33k; #15507, @vzamboulingame; #15459, @UnMelow; #13991, @abhishekpawar1060; #12161, @zhouyou9505; #15293, @tornikeo

## 2.x

[changelogs/v2.x.md](changelogs/v2.x.md)

## 1.x

[changelogs/v1.x.md](changelogs/v1.x.md)

## 0.x

[changelogs/v0.x.md](changelogs/v0.x.md)
