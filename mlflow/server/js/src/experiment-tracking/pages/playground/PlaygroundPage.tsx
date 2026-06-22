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
import { useEffect, useMemo, useRef, useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import { PlaygroundTopBar } from './components/PlaygroundTopBar';
import { PromptInputPanel } from './components/PromptInputPanel';
import { PromptRegistryPicker } from './components/PromptRegistryPicker';
import type { PromptLoadPayload } from './components/PromptRegistryPicker';
import { SavePromptVersionModal } from './components/SavePromptVersionModal';
import { useChatCompletionMutation } from './hooks/useChatCompletionMutation';
import type {
  ConversationMessage,
  PlaygroundParams,
  PlaygroundTool,
  PromptType,
  ResponseFormat,
  ResponseFormatType,
  ToolChoice,
} from './types';
import { BLANK_JSON_SCHEMA, getEmptyVariables, getToolParametersError, substituteVariables } from './utils';

const EMPTY_USER_MESSAGE: ConversationMessage = { role: 'user', content: '' };

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

  const { mutate, error, isLoading, reset } = useChatCompletionMutation();

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
      const paramsError = getToolParametersError(tool.params);
      if (paramsError) {
        return paramsError;
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
      blockers.push(
        intl.formatMessage(
          {
            defaultMessage: 'Fix the tool definitions: {error}',
            description:
              'Reason shown when the playground Submit button is disabled because a tool definition is empty or invalid',
          },
          { error: firstToolError },
        ),
      );
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

  const handleSubmit = () => {
    if (!canSubmit) {
      return;
    }
    // canSubmit guarantees every tool has a name and valid parameters JSON, so the parses are safe.
    const wireTools =
      tools.length > 0
        ? tools.map((tool) => ({
            type: 'function',
            function: {
              name: tool.name.trim(),
              ...(tool.description.trim() && { description: tool.description.trim() }),
              parameters: JSON.parse(tool.params),
            },
          }))
        : undefined;
    let response_format: ResponseFormat | undefined;
    if (responseFormatType === 'json_object') {
      response_format = { type: 'json_object' };
    } else if (responseFormatType === 'json_schema') {
      response_format = {
        type: 'json_schema',
        json_schema: {
          name: 'response_schema',
          schema: JSON.parse(responseFormatSchemaText),
          strict: true,
        },
      };
    }
    // Multi-turn tool conversations that forward tool_calls plus tool results are a known follow-up.
    const wireMessages = substituteVariables(messages, variables).map(({ role, content }) => ({ role, content }));
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
          const hasToolCalls = Boolean(assistant.tool_calls?.length);
          const appended: ConversationMessage = {
            ...assistant,
            content: assistant.content ?? '',
            ...(hasToolCalls && { tool_calls: assistant.tool_calls }),
            ...(response_format && { contentIsJson: true }),
            usage: response.usage,
          };
          setMessages((current) => [...current, appended, { ...EMPTY_USER_MESSAGE }]);
        },
      },
    );
  };

  const conversationIsEmpty = messages.length <= 1 && !messages[0]?.content?.trim();

  const handleClearConversation = () => {
    setMessages([{ ...EMPTY_USER_MESSAGE }]);
    reset();
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
