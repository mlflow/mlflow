import { Button, Header, PlayIcon, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import { EndpointSelector } from '../../components/EndpointSelector';
import { CompletionOutputPanel } from './components/CompletionOutputPanel';
import { ParametersPanel } from './components/ParametersPanel';
import { PromptInputPanel } from './components/PromptInputPanel';
import { PromptRegistryPicker } from './components/PromptRegistryPicker';
import { useChatCompletionMutation } from './hooks/useChatCompletionMutation';
import type { ChatMessage, PlaygroundParams } from './types';

const EMPTY_USER_MESSAGE: ChatMessage = { role: 'user', content: '' };

const PlaygroundPage = () => {
  const { theme } = useDesignSystemTheme();
  const [endpointName, setEndpointName] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([{ ...EMPTY_USER_MESSAGE }]);
  const [params, setParams] = useState<PlaygroundParams>({});
  const [showRegistryPicker, setShowRegistryPicker] = useState(false);

  const { mutate, data, error, isLoading } = useChatCompletionMutation();

  const canSubmit =
    Boolean(endpointName) && messages.length > 0 && messages.some((m) => m.content.trim().length > 0) && !isLoading;

  const handleSubmit = () => {
    if (!canSubmit) {
      return;
    }
    mutate({
      model: endpointName,
      messages,
      ...(params.temperature !== undefined && { temperature: params.temperature }),
      ...(params.max_tokens !== undefined && { max_tokens: params.max_tokens }),
      ...(params.top_p !== undefined && { top_p: params.top_p }),
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
      <div
        css={{
          display: 'grid',
          gridTemplateColumns: '320px 1fr',
          gap: theme.spacing.lg,
          flex: 1,
          minHeight: 0,
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <EndpointSelector
            componentIdPrefix="mlflow.playground.endpoint-selector"
            currentEndpointName={endpointName}
            onEndpointSelect={setEndpointName}
            showCreateButton={false}
          />
          <ParametersPanel value={params} onChange={setParams} />
          <Button componentId="mlflow.playground.load_from_registry" onClick={() => setShowRegistryPicker(true)}>
            <FormattedMessage
              defaultMessage="Load prompt from registry"
              description="Label for the button that opens the registered prompt picker on the playground page"
            />
          </Button>
        </div>

        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md, minHeight: 0 }}>
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
