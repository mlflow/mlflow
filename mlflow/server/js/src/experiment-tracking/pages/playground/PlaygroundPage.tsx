import {
  Alert,
  Button,
  HoverCard,
  Notification,
  PlayIcon,
  Spacer,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { isEqual } from 'lodash';
import { useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  useGetTracesById,
  PLAYGROUND_TRACE_ID_QUERY_PARAM,
  PLAYGROUND_SPAN_ID_QUERY_PARAM,
} from '@databricks/web-shared/model-trace-explorer';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { Link, generatePath, useParams, useSearchParams } from '../../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import { RoutePaths } from '../../routes';
import { buildPlaygroundPrefillFromTrace } from './traceToPlayground';
import type { PrefillToolResult } from './traceToPlayground';
import { PlaygroundTopBar } from './components/PlaygroundTopBar';
import { PromptInputPanel } from './components/PromptInputPanel';
import { PromptRegistryPicker } from './components/PromptRegistryPicker';
import type { PromptLoadPayload } from './components/PromptRegistryPicker';
import { SavePromptVersionModal } from './components/SavePromptVersionModal';
import { useChatCompletionMutation } from './hooks/useChatCompletionMutation';
import { useLogPlaygroundTraceMutation } from './hooks/useLogPlaygroundTraceMutation';
import type {
  ConversationMessage,
  PlaygroundParams,
  PlaygroundTool,
  PromptType,
  ResponseFormatType,
  ToolChoice,
} from './types';
import {
  BLANK_JSON_SCHEMA,
  buildResponseFormat,
  getEmptyVariables,
  getToolParametersError,
  substituteVariables,
  toWireTool,
} from './utils';

const EMPTY_USER_MESSAGE: ConversationMessage = { role: 'user', content: '' };

// Upper bound on automatic tool rounds per submit, so a model that keeps requesting tools cannot
// loop the playground (and the user's tokens) forever.
const MAX_AUTO_TOOL_ROUNDS = 6;

const parseToolArgs = (raw?: string): unknown => {
  if (!raw) {
    return undefined;
  }
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
};

// Finds the captured result for a tool call among the source trace's executions: an execution of
// the same tool with deep-equal arguments is an *exact* replay; otherwise any execution of the
// same tool serves as a fallback stand-in (the arguments may differ after the user edits the
// prompt). The exact/fallback distinction gates the automatic loop: only exact replays are safe
// to feed back without user review. Returns undefined when the trace never ran that tool.
const resolveToolResult = (
  captured: PrefillToolResult[],
  name: string | undefined,
  args: string | undefined,
): { result: string; exact: boolean } | undefined => {
  if (!name) {
    return undefined;
  }
  const wantArgs = parseToolArgs(args);
  const exact = captured.find((entry) => entry.name === name && isEqual(parseToolArgs(entry.args), wantArgs));
  if (exact) {
    return { result: exact.result, exact: true };
  }
  const fallback = captured.find((entry) => entry.name === name);
  return fallback ? { result: fallback.result, exact: false } : undefined;
};

const PlaygroundPage = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [endpointName, setEndpointName] = useState<string>('');
  const [messages, setMessages] = useState<ConversationMessage[]>([{ ...EMPTY_USER_MESSAGE }]);
  const [params, setParams] = useState<PlaygroundParams>({});
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [tools, setTools] = useState<PlaygroundTool[]>([]);
  const [toolChoice, setToolChoice] = useState<ToolChoice>('auto');
  // Monotonic counter for stable client-side tool ids (used as React keys).
  const toolIdRef = useRef(0);
  const [responseFormatType, setResponseFormatType] = useState<ResponseFormatType>('text');
  const [responseFormatSchemaText, setResponseFormatSchemaText] = useState<string>('');
  const [showRegistryPicker, setShowRegistryPicker] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);
  // The registry prompt currently loaded into the playground, if any. Lets the
  // save modal default to appending a new version of that prompt and preserve
  // its type.
  const [loadedPrompt, setLoadedPrompt] = useState<{ name: string; version: string; promptType: PromptType } | null>(
    null,
  );
  const [loadedToast, setLoadedToast] = useState<{
    name: string;
    version: string;
    withSettings: boolean;
  } | null>(null);
  const [savedToast, setSavedToast] = useState<{ name: string; version: string } | null>(null);

  const { experimentId } = useParams<{ experimentId: string }>();
  const [searchParams] = useSearchParams();
  const prefillTraceId = searchParams.get(PLAYGROUND_TRACE_ID_QUERY_PARAM) ?? undefined;
  const prefillSpanId = searchParams.get(PLAYGROUND_SPAN_ID_QUERY_PARAM) ?? undefined;
  // The trace this playground session was opened from ("Open in Playground"), if any. Drives the
  // "loaded from trace" banner and attributes a saved-back trace to its source for comparison.
  const [sourceTrace, setSourceTrace] = useState<{ traceId: string; spanName?: string } | null>(null);
  // Prefill is applied exactly once, on the first successful fetch, so it never clobbers edits the
  // user makes afterward (or on a background refetch).
  const prefillAppliedRef = useRef(false);
  // Tool executions captured in the source trace, used to auto-answer the model's tool calls.
  const sourceToolResultsRef = useRef<PrefillToolResult[]>([]);
  const { data: prefillTraces } = useGetTracesById(prefillTraceId ? [prefillTraceId] : []);

  const { mutate, error, isLoading, reset } = useChatCompletionMutation();
  const {
    mutate: logTraceMutate,
    isLoading: isSavingTrace,
    error: saveTraceError,
    reset: resetSaveTrace,
  } = useLogPlaygroundTraceMutation();
  // The trace saved back from the current playground run ("Save as trace"), if any.
  const [savedTrace, setSavedTrace] = useState<{ traceId: string } | null>(null);

  useEffect(() => {
    if (prefillAppliedRef.current || !prefillTraceId) {
      return;
    }
    const trace = prefillTraces?.[0];
    if (!trace) {
      return;
    }
    prefillAppliedRef.current = true;
    const prefill = buildPlaygroundPrefillFromTrace(trace as ModelTrace, prefillSpanId);
    if (!prefill) {
      return;
    }
    setMessages(prefill.messages.length > 0 ? prefill.messages : [{ ...EMPTY_USER_MESSAGE }]);
    if (prefill.endpointName) {
      setEndpointName(prefill.endpointName);
    }
    setParams(prefill.params);
    // Assign client-side ids the same way handleAddTool does, so the restored tools have stable
    // React keys that never collide with tools added later.
    setTools(prefill.tools.map((tool) => ({ ...tool, id: `tool-${toolIdRef.current++}` })));
    setToolChoice(prefill.toolChoice);
    setResponseFormatType(prefill.responseFormatType);
    setResponseFormatSchemaText(prefill.responseFormatSchemaText);
    sourceToolResultsRef.current = prefill.toolResults;
    setSourceTrace({ traceId: prefillTraceId, spanName: prefill.spanName });
  }, [prefillTraces, prefillTraceId, prefillSpanId]);

  useEffect(() => {
    if (!loadedToast) return;
    const t = window.setTimeout(() => setLoadedToast(null), 4000);
    return () => window.clearTimeout(t);
  }, [loadedToast]);

  useEffect(() => {
    if (!savedToast) return;
    const t = window.setTimeout(() => setSavedToast(null), 4000);
    return () => window.clearTimeout(t);
  }, [savedToast]);

  const handleRegistryLoad = (payload: PromptLoadPayload) => {
    setMessages(payload.messages.length > 0 ? payload.messages : [{ ...EMPTY_USER_MESSAGE }]);
    if (payload.settings !== null) {
      setParams(payload.settings.params);
      setResponseFormatType(payload.settings.responseFormatType);
      setResponseFormatSchemaText(payload.settings.responseFormatSchemaText);
    }
    setShowRegistryPicker(false);
    setLoadedPrompt({ name: payload.promptName, version: payload.versionLabel, promptType: payload.promptType });
    setLoadedToast({
      name: payload.promptName,
      version: payload.versionLabel,
      withSettings: payload.settings !== null,
    });
  };

  const handleSaved = ({ name, version, promptType }: { name: string; version: string; promptType: PromptType }) => {
    setShowSaveModal(false);
    // Chain subsequent saves onto the freshly created version.
    setLoadedPrompt({ name, version, promptType });
    setSavedToast({ name, version });
  };

  const handleAddTool = () => {
    setTools((current) => [
      ...current,
      { id: `tool-${toolIdRef.current++}`, name: '', description: '', params: BLANK_JSON_SCHEMA },
    ]);
  };
  const handleRemoveTool = (id: string) => {
    setTools((current) => current.filter((tool) => tool.id !== id));
  };
  const handleUpdateTool = (id: string, patch: Partial<PlaygroundTool>) => {
    setTools((current) => current.map((tool) => (tool.id === id ? { ...tool, ...patch } : tool)));
  };

  const handleResponseFormatTypeChange = (next: ResponseFormatType) => {
    setResponseFormatType(next);
    // Pre-populate a bare-minimum schema the first time JSON schema is selected.
    if (next === 'json_schema' && !responseFormatSchemaText.trim()) {
      setResponseFormatSchemaText(BLANK_JSON_SCHEMA);
    }
  };

  // First validation error across all tools — a missing function name or an
  // invalid parameters schema — or null when every tool is complete and valid.
  const firstToolError = useMemo(() => {
    for (const tool of tools) {
      if (!tool.name.trim()) {
        return intl.formatMessage({
          defaultMessage: 'Give every tool a function name',
          description: 'Submit blocker detail shown when a playground tool is missing its function name',
        });
      }
      if (getToolParametersError(tool.params)) {
        return intl.formatMessage({
          defaultMessage: 'Fix the tool parameters schema',
          description: 'Submit blocker shown when a playground tool has an invalid parameters schema',
        });
      }
    }
    return null;
  }, [tools, intl]);

  const responseFormatSchemaError = useMemo(() => {
    if (responseFormatType !== 'json_schema') {
      return null;
    }
    if (!responseFormatSchemaText.trim()) {
      return 'Schema is required';
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(responseFormatSchemaText);
    } catch (e) {
      return e instanceof Error ? e.message : 'Invalid JSON';
    }
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return 'Schema must be a JSON object';
    }
    return null;
  }, [responseFormatType, responseFormatSchemaText]);

  const submitBlockers = useMemo(() => {
    const blockers: string[] = [];
    if (!endpointName) {
      blockers.push(
        intl.formatMessage({
          defaultMessage: 'Select a model endpoint',
          description: 'Reason shown when the playground Submit button is disabled because no endpoint is selected',
        }),
      );
    }
    if (messages.length === 0 || messages.some((m) => (m.content ?? '').trim().length === 0 && !m.tool_calls?.length)) {
      blockers.push(
        intl.formatMessage({
          defaultMessage: 'Fill in every message',
          description: 'Reason shown when the playground Submit button is disabled because a message is empty',
        }),
      );
    }
    if (tools.length > 0 && firstToolError) {
      blockers.push(firstToolError);
    }
    if (responseFormatSchemaError) {
      blockers.push(
        intl.formatMessage(
          {
            defaultMessage: 'Fix the response format schema: {error}',
            description:
              'Reason shown when the playground Submit button is disabled because the response format schema is invalid',
          },
          { error: responseFormatSchemaError },
        ),
      );
    }
    for (const name of getEmptyVariables(messages, variables)) {
      blockers.push(
        intl.formatMessage(
          {
            defaultMessage: 'Provide a value for the variable {name}',
            description: 'Reason shown when the playground Submit button is disabled because a variable is empty',
          },
          { name },
        ),
      );
    }
    return blockers;
  }, [endpointName, messages, tools, firstToolError, responseFormatSchemaError, variables, intl]);

  const canSubmit = submitBlockers.length === 0 && !isLoading;

  // Sends a conversation to the endpoint and drives the tool loop. When the model responds with
  // tool calls, each call is answered from the source trace's captured executions; if every call
  // is answered, the conversation automatically continues (bounded by MAX_AUTO_TOOL_ROUNDS) until
  // the model produces its final response. Calls the trace never executed are left as editable
  // tool-result inputs for the user to fill in and resubmit.
  const submitConversation = (conversation: ConversationMessage[], depth: number) => {
    // Validity is guaranteed by canSubmit at depth 0 (every tool has a name and valid parameters
    // JSON) and by construction on automatic rounds, so the parses are safe.
    const wireTools = tools.length > 0 ? tools.map(toWireTool) : undefined;
    const response_format = buildResponseFormat(responseFormatType, responseFormatSchemaText);
    // Forward the full tool exchange: assistant turns keep their tool_calls and tool-result turns
    // keep their tool_call_id, so the model can produce the final response. Display-only fields
    // (usage, toolName, contentIsJson) are stripped.
    const wireMessages = substituteVariables(conversation, variables).map(
      ({ role, content, tool_calls, tool_call_id }) => ({
        role,
        content,
        ...(tool_calls?.length ? { tool_calls } : {}),
        ...(tool_call_id ? { tool_call_id } : {}),
      }),
    );
    mutate(
      {
        model: endpointName,
        messages: wireMessages,
        ...(params.temperature !== undefined && { temperature: params.temperature }),
        ...(params.max_tokens !== undefined && { max_tokens: params.max_tokens }),
        ...(params.top_p !== undefined && { top_p: params.top_p }),
        ...(params.top_k !== undefined && { top_k: params.top_k }),
        ...(params.presence_penalty !== undefined && { presence_penalty: params.presence_penalty }),
        ...(params.frequency_penalty !== undefined && { frequency_penalty: params.frequency_penalty }),
        ...(params.stop && params.stop.length > 0 && { stop: params.stop }),
        ...(wireTools && { tools: wireTools }),
        ...(wireTools && { tool_choice: toolChoice }),
        ...(response_format && { response_format }),
      },
      {
        onSuccess: (response) => {
          const assistant = response.choices?.[0]?.message;
          if (!assistant) return;
          const namedToolCalls = (assistant.tool_calls ?? []).filter((toolCall) => toolCall.function?.name);
          const hasToolCalls = namedToolCalls.length > 0;
          const appended: ConversationMessage = {
            ...assistant,
            content: assistant.content ?? '',
            // Only named calls are kept: each kept call gets a paired tool-result message below,
            // so the forwarded history never carries an unanswered tool call.
            ...(hasToolCalls && { tool_calls: namedToolCalls }),
            ...(response_format && { contentIsJson: true }),
            usage: response.usage,
          };
          if (!hasToolCalls) {
            // Final response — append it plus a fresh user composer for the next turn.
            setMessages([...conversation, appended, { ...EMPTY_USER_MESSAGE }]);
            return;
          }
          // Answer each tool call from the trace's captured executions.
          let allExact = true;
          const toolMessages: ConversationMessage[] = namedToolCalls.map((toolCall) => {
            const resolved = resolveToolResult(
              sourceToolResultsRef.current,
              toolCall.function?.name,
              toolCall.function?.arguments,
            );
            if (!resolved?.exact) {
              allExact = false;
            }
            return {
              role: 'tool',
              content: resolved?.result ?? '',
              ...(toolCall.id ? { tool_call_id: toolCall.id } : {}),
              toolName: toolCall.function?.name,
            };
          });
          const next = [...conversation, appended, ...toolMessages];
          setMessages(next);
          // Continue automatically only when every call was answered by an exact-arguments replay.
          // Fallback results (same tool, different arguments — a stale stand-in) pre-fill the
          // inputs but stop the loop so the user can review or edit them before resubmitting.
          if (allExact && depth < MAX_AUTO_TOOL_ROUNDS) {
            submitConversation(next, depth + 1);
          }
        },
      },
    );
  };

  const handleSubmit = () => {
    if (!canSubmit) {
      return;
    }
    submitConversation(messages, 0);
  };

  const conversationIsEmpty = messages.length <= 1 && !messages[0]?.content?.trim();

  const handleClearConversation = () => {
    setMessages([{ ...EMPTY_USER_MESSAGE }]);
    reset();
    setSavedTrace(null);
    resetSaveTrace();
    // Clearing discards the trace's captured input, so every piece of trace-derived state must go
    // with it: auto-replaying the old trace's tool results into an unrelated new prompt (or
    // attributing a save to the discarded trace) would silently corrupt the fresh session.
    setSourceTrace(null);
    sourceToolResultsRef.current = [];
  };

  // The most recent assistant reply, present once the user has run at least one completion.
  // "Save as trace" persists the input that produced it plus the reply.
  const lastAssistantIndex = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].role === 'assistant') {
        return i;
      }
    }
    return -1;
  }, [messages]);

  // A run is saveable only once the model has produced a *final* reply. While the tool loop is
  // paused for review, the last assistant turn is a contentless tool-call request — saving then
  // would drop the trailing tool results and store that request as the trace's "response".
  const hasFinalResponse = lastAssistantIndex >= 0 && !messages[lastAssistantIndex].tool_calls?.length;

  const canSaveTrace = hasFinalResponse && Boolean(experimentId) && !isSavingTrace;

  const handleSaveAsTrace = () => {
    if (!hasFinalResponse || !experimentId) {
      return;
    }
    // A new attempt starts from a clean state: clear the previous result and error so a failed
    // re-save surfaces its error banner instead of hiding behind a stale success banner.
    setSavedTrace(null);
    resetSaveTrace();
    const inputMessages = substituteVariables(messages.slice(0, lastAssistantIndex), variables).map(
      ({ role, content, tool_calls, tool_call_id }) => ({
        role,
        content,
        ...(tool_calls?.length ? { tool_calls } : {}),
        ...(tool_call_id ? { tool_call_id } : {}),
      }),
    );
    const assistant = messages[lastAssistantIndex];
    // Capture the current tool + response-format configuration so the saved trace is a faithful,
    // reloadable record. Unlike submit, save isn't gated on a valid config, so drop invalid tools
    // and skip an invalid response-format schema rather than throwing.
    const validTools = tools.filter((tool) => tool.name.trim() && !getToolParametersError(tool.params));
    const wireTools = validTools.length > 0 ? validTools.map(toWireTool) : undefined;
    const responseFormat =
      responseFormatType === 'text' || responseFormatSchemaError
        ? undefined
        : buildResponseFormat(responseFormatType, responseFormatSchemaText);
    logTraceMutate(
      {
        experiment_id: experimentId,
        messages: inputMessages,
        response: {
          role: assistant.role,
          content: assistant.content,
          ...(assistant.tool_calls ? { tool_calls: assistant.tool_calls } : {}),
        },
        ...(assistant.usage ? { usage: assistant.usage } : {}),
        ...(endpointName ? { model: endpointName } : {}),
        params,
        ...(wireTools ? { tools: wireTools, tool_choice: toolChoice } : {}),
        ...(responseFormat ? { response_format: responseFormat } : {}),
        ...(sourceTrace?.traceId ? { source_trace_id: sourceTrace.traceId } : {}),
      },
      { onSuccess: ({ trace_id }) => setSavedTrace({ traceId: trace_id }) },
    );
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      <PlaygroundTopBar
        endpointName={endpointName}
        onEndpointSelect={setEndpointName}
        params={params}
        onParamsChange={setParams}
        tools={tools}
        onAddTool={handleAddTool}
        onRemoveTool={handleRemoveTool}
        onUpdateTool={handleUpdateTool}
        toolChoice={toolChoice}
        onToolChoiceChange={setToolChoice}
        responseFormatType={responseFormatType}
        onResponseFormatTypeChange={handleResponseFormatTypeChange}
        responseFormatSchemaText={responseFormatSchemaText}
        onResponseFormatSchemaChange={setResponseFormatSchemaText}
        responseFormatSchemaError={responseFormatSchemaError}
        messages={messages}
        variables={variables}
        onVariablesChange={setVariables}
        onOpenRegistry={() => setShowRegistryPicker(true)}
        onOpenSave={() => setShowSaveModal(true)}
        saveDisabled={conversationIsEmpty}
      />
      <Spacer size="sm" shrinks={false} />
      <div css={{ borderTop: `1px solid ${theme.colors.border}`, flexShrink: 0 }} role="separator" aria-hidden="true" />
      <Spacer size="sm" shrinks={false} />
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          gap: theme.spacing.md,
          flex: 1,
          minHeight: 0,
          overflowY: 'auto',
        }}
      >
        {sourceTrace && (
          <Alert
            type="info"
            componentId="mlflow.playground.loaded_from_trace"
            closable
            onClose={() => setSourceTrace(null)}
            message={
              <FormattedMessage
                defaultMessage="Loaded from trace"
                description="Title of the banner shown on the playground when it was opened from a trace via 'Open in Playground'"
              />
            }
            description={
              <span>
                {sourceTrace.spanName ? (
                  <FormattedMessage
                    defaultMessage="Testing the input captured in span “{spanName}”. Edit the prompt or model and re-run. "
                    description="Body of the playground banner shown when opened from a specific span of a trace"
                    values={{ spanName: sourceTrace.spanName }}
                  />
                ) : (
                  <FormattedMessage
                    defaultMessage="Testing the input captured in this trace. Edit the prompt or model and re-run. "
                    description="Body of the playground banner shown when opened from a trace"
                  />
                )}
                {experimentId && (
                  <Link
                    componentId="mlflow.playground.loaded_from_trace.view_trace"
                    to={generatePath(RoutePaths.experimentPageTabTraceDetail, {
                      experimentId,
                      traceId: sourceTrace.traceId,
                    })}
                  >
                    <FormattedMessage
                      defaultMessage="View original trace"
                      description="Link in the playground banner that navigates back to the trace the playground was opened from"
                    />
                  </Link>
                )}
              </span>
            }
          />
        )}
        {savedTrace && (
          <Alert
            type="success"
            componentId="mlflow.playground.saved_as_trace"
            closable
            onClose={() => setSavedTrace(null)}
            message={
              <FormattedMessage
                defaultMessage="Saved as a new trace"
                description="Title of the success banner shown after saving a playground run back as a new trace"
              />
            }
            description={
              experimentId ? (
                <Link
                  componentId="mlflow.playground.saved_as_trace.view_trace"
                  to={generatePath(RoutePaths.experimentPageTabTraceDetail, {
                    experimentId,
                    traceId: savedTrace.traceId,
                  })}
                >
                  <FormattedMessage
                    defaultMessage="View trace for side-by-side comparison"
                    description="Link in the playground success banner that opens the newly saved trace"
                  />
                </Link>
              ) : undefined
            }
          />
        )}
        {saveTraceError && !savedTrace && (
          <Alert
            type="error"
            componentId="mlflow.playground.save_as_trace.error"
            closable
            onClose={() => resetSaveTrace()}
            message={
              <FormattedMessage
                defaultMessage="Failed to save as trace"
                description="Title of the error banner shown when saving a playground run as a trace fails"
              />
            }
            description={saveTraceError.message}
          />
        )}
        <PromptInputPanel messages={messages} onChange={setMessages} />
        {isLoading && (
          <div css={{ display: 'flex', justifyContent: 'center', padding: theme.spacing.md }}>
            <Spinner />
          </div>
        )}
        {!isLoading &&
          error &&
          (() => {
            const status = (error as { status?: number }).status;
            const detail = status ? `HTTP ${status} — ${error.message}` : error.message;
            return (
              <Alert
                type="error"
                componentId="mlflow.playground.output.error"
                closable={false}
                message={
                  <FormattedMessage
                    defaultMessage="Chat completion failed"
                    description="Title of the error alert shown on the playground when a chat completion request fails"
                  />
                }
                description={
                  <pre
                    css={{
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      margin: 0,
                      fontFamily: 'inherit',
                      maxHeight: 240,
                      overflow: 'auto',
                    }}
                  >
                    {detail}
                  </pre>
                }
              />
            );
          })()}
        <div
          css={{
            display: 'flex',
            justifyContent: 'flex-end',
            alignItems: 'center',
            gap: theme.spacing.sm,
            paddingTop: theme.spacing.md,
            paddingBottom: theme.spacing.md,
            position: 'sticky',
            bottom: 0,
            backgroundColor: theme.colors.backgroundPrimary,
          }}
        >
          <Button
            componentId="mlflow.playground.clear"
            disabled={conversationIsEmpty}
            onClick={handleClearConversation}
          >
            <FormattedMessage
              defaultMessage="Clear conversation"
              description="Label for the button that resets the playground conversation to an empty user message"
            />
          </Button>
          {hasFinalResponse && (
            <Button
              componentId="mlflow.playground.save_as_trace"
              disabled={!canSaveTrace}
              loading={isSavingTrace}
              onClick={handleSaveAsTrace}
            >
              <FormattedMessage
                defaultMessage="Save as trace"
                description="Label for the playground button that saves the current run back as a new MLflow trace for side-by-side comparison"
              />
            </Button>
          )}
          {(() => {
            const submitButton = (
              <Button
                componentId="mlflow.playground.submit"
                type="primary"
                icon={<PlayIcon />}
                disabled={!canSubmit}
                loading={isLoading}
                onClick={handleSubmit}
              >
                <FormattedMessage
                  defaultMessage="Submit"
                  description="Label for the submit button on the playground page that runs the chat completion request"
                />
              </Button>
            );
            if (submitBlockers.length === 0) {
              return submitButton;
            }
            return (
              <HoverCard
                trigger={<span css={{ display: 'inline-block' }}>{submitButton}</span>}
                content={
                  <div
                    css={{
                      display: 'flex',
                      flexDirection: 'column',
                      gap: theme.spacing.sm,
                      padding: theme.spacing.sm,
                      maxWidth: 360,
                    }}
                  >
                    <Typography.Paragraph withoutMargins>
                      <FormattedMessage
                        defaultMessage="To submit:"
                        description="Lead-in line of the popup shown when the playground Submit button is disabled"
                      />
                    </Typography.Paragraph>
                    <ul css={{ margin: `${theme.spacing.xs}px 0 0`, paddingLeft: theme.spacing.lg }}>
                      {submitBlockers.map((reason) => (
                        <li key={reason}>{reason}</li>
                      ))}
                    </ul>
                  </div>
                }
                side="top"
                align="end"
              />
            );
          })()}
        </div>
      </div>

      <PromptRegistryPicker
        visible={showRegistryPicker}
        onCancel={() => setShowRegistryPicker(false)}
        onLoad={handleRegistryLoad}
      />
      <SavePromptVersionModal
        visible={showSaveModal}
        onCancel={() => setShowSaveModal(false)}
        messages={messages}
        params={params}
        responseFormatType={responseFormatType}
        responseFormatSchemaText={responseFormatSchemaText}
        loadedPromptName={loadedPrompt?.name}
        loadedPromptType={loadedPrompt?.promptType}
        onSaved={handleSaved}
      />
      {loadedToast && (
        <Notification.Provider>
          <Notification.Root severity="success" componentId="mlflow.playground.prompt_registry_picker.loaded">
            <Notification.Title>
              {loadedToast.withSettings ? (
                <FormattedMessage
                  defaultMessage="Loaded {name} v{version} with settings"
                  description="Success toast shown on the playground after loading a prompt version that also applied stored settings"
                  values={{ name: loadedToast.name, version: loadedToast.version }}
                />
              ) : (
                <FormattedMessage
                  defaultMessage="Loaded {name} v{version}"
                  description="Success toast shown on the playground after loading a prompt version that did not include stored settings"
                  values={{ name: loadedToast.name, version: loadedToast.version }}
                />
              )}
            </Notification.Title>
            <Notification.Close componentId="mlflow.playground.prompt_registry_picker.loaded.close" />
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}
      {savedToast && (
        <Notification.Provider>
          <Notification.Root severity="success" componentId="mlflow.playground.save_prompt_version.saved">
            <Notification.Title>
              <FormattedMessage
                defaultMessage="Saved {name} v{version} to the registry"
                description="Success toast shown on the playground after saving the current messages as a new prompt version"
                values={{ name: savedToast.name, version: savedToast.version }}
              />
            </Notification.Title>
            <Notification.Close componentId="mlflow.playground.save_prompt_version.saved.close" />
          </Notification.Root>
          <Notification.Viewport />
        </Notification.Provider>
      )}
    </div>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, PlaygroundPage);
