import { Button, Header, PlayIcon, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import { CompletionOutputPanel } from './components/CompletionOutputPanel';
import { PlaygroundTopBar } from './components/PlaygroundTopBar';
import { PromptInputPanel } from './components/PromptInputPanel';
import { PromptRegistryPicker } from './components/PromptRegistryPicker';
import { useChatCompletionMutation } from './hooks/useChatCompletionMutation';
import type { ChatMessage, PlaygroundParams, ResponseFormat, ResponseFormatType, ToolChoice } from './types';
import { substituteVariables } from './utils';

const EMPTY_USER_MESSAGE: ChatMessage = { role: 'user', content: '' };

const PlaygroundPage = () => {
  const { theme } = useDesignSystemTheme();
  const [endpointName, setEndpointName] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([{ ...EMPTY_USER_MESSAGE }]);
  const [params, setParams] = useState<PlaygroundParams>({});
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [toolsText, setToolsText] = useState<string>('');
  const [toolChoice, setToolChoice] = useState<ToolChoice>('none');
  const [responseFormatType, setResponseFormatType] = useState<ResponseFormatType>('text');
  const [responseFormatSchemaText, setResponseFormatSchemaText] = useState<string>('');
  const [showRegistryPicker, setShowRegistryPicker] = useState(false);

  const { mutate, data, error, isLoading } = useChatCompletionMutation();

  const toolsError = useMemo(() => {
    if (!toolsText.trim()) {
      return null;
    }
    try {
      const parsed = JSON.parse(toolsText);
      if (!Array.isArray(parsed)) {
        return 'Tools must be a JSON array';
      }
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : 'Invalid JSON';
    }
  }, [toolsText]);

  const responseFormatSchemaError = useMemo(() => {
    if (responseFormatType !== 'json_schema') {
      return null;
    }
    if (!responseFormatSchemaText.trim()) {
      return 'Schema is required';
    }
    try {
      JSON.parse(responseFormatSchemaText);
      return null;
    } catch (e) {
      return e instanceof Error ? e.message : 'Invalid JSON';
    }
  }, [responseFormatType, responseFormatSchemaText]);

  const canSubmit =
    Boolean(endpointName) &&
    messages.length > 0 &&
    messages.some((m) => m.content.trim().length > 0) &&
    (toolChoice === 'none' || !toolsError) &&
    !responseFormatSchemaError &&
    !isLoading;

  const handleSubmit = () => {
    if (!canSubmit) {
      return;
    }
    const tools = toolChoice !== 'none' && toolsText.trim() ? (JSON.parse(toolsText) as unknown[]) : undefined;
    let response_format: ResponseFormat | undefined;
    if (responseFormatType === 'json_object') {
      response_format = { type: 'json_object' };
    } else if (responseFormatType === 'json_schema') {
      response_format = { type: 'json_schema', json_schema: JSON.parse(responseFormatSchemaText) };
    }
    mutate({
      model: endpointName,
      messages: substituteVariables(messages, variables),
      ...(params.temperature !== undefined && { temperature: params.temperature }),
      ...(params.max_tokens !== undefined && { max_tokens: params.max_tokens }),
      ...(params.top_p !== undefined && { top_p: params.top_p }),
      ...(params.top_k !== undefined && { top_k: params.top_k }),
      ...(params.presence_penalty !== undefined && { presence_penalty: params.presence_penalty }),
      ...(params.frequency_penalty !== undefined && { frequency_penalty: params.frequency_penalty }),
      ...(params.stop && params.stop.length > 0 && { stop: params.stop }),
      ...(tools && { tools }),
      ...(tools && { tool_choice: toolChoice }),
      ...(response_format && { response_format }),
    });
  };

  return (
    <ScrollablePageWrapper css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span
              css={{
                display: 'flex',
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
              }}
            >
              <PlayIcon />
            </span>
            <FormattedMessage defaultMessage="Playground" description="Title of the LLM playground page" />
          </span>
        }
      />
      <Spacer shrinks={false} />
      <PlaygroundTopBar
        endpointName={endpointName}
        onEndpointSelect={setEndpointName}
        params={params}
        onParamsChange={setParams}
        toolsText={toolsText}
        onToolsChange={setToolsText}
        toolsError={toolsError}
        toolChoice={toolChoice}
        onToolChoiceChange={setToolChoice}
        responseFormatType={responseFormatType}
        onResponseFormatTypeChange={setResponseFormatType}
        responseFormatSchemaText={responseFormatSchemaText}
        onResponseFormatSchemaChange={setResponseFormatSchemaText}
        responseFormatSchemaError={responseFormatSchemaError}
        messages={messages}
        variables={variables}
        onVariablesChange={setVariables}
        onOpenRegistry={() => setShowRegistryPicker(true)}
      />
      <Spacer size="sm" shrinks={false} />
      <div css={{ borderTop: `1px solid ${theme.colors.border}`, flexShrink: 0 }} role="separator" aria-hidden="true" />
      <Spacer size="sm" shrinks={false} />
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, flex: 1, minHeight: 0 }}>
        <PromptInputPanel messages={messages} onChange={setMessages} />
        <div css={{ display: 'flex', justifyContent: 'flex-end' }}>
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
        </div>
        <Spacer size="sm" />
        <CompletionOutputPanel response={data} error={error ?? undefined} isLoading={isLoading} />
      </div>

      <PromptRegistryPicker
        visible={showRegistryPicker}
        onCancel={() => setShowRegistryPicker(false)}
        onLoad={(loadedMessages) => {
          setMessages(loadedMessages.length > 0 ? loadedMessages : [{ ...EMPTY_USER_MESSAGE }]);
          setShowRegistryPicker(false);
        }}
      />
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, PlaygroundPage);
