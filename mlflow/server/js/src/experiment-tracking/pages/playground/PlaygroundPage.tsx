import { Button, PlayIcon, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import { useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';
import { PageHeader } from '../../../shared/building_blocks/PageHeader';
import { CompletionOutputPanel } from './components/CompletionOutputPanel';
import { EndpointPicker } from './components/EndpointPicker';
import { PromptInputPanel } from './components/PromptInputPanel';
import { PromptRegistryPicker } from './components/PromptRegistryPicker';
import { useChatCompletionMutation } from './hooks/useChatCompletionMutation';
import type { ChatMessage } from './types';

const PlaygroundPage = () => {
  const { theme } = useDesignSystemTheme();
  const [endpointName, setEndpointName] = useState<string>('');
  const [messages, setMessages] = useState<ChatMessage[]>([{ role: 'user', content: '' }]);
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
    });
  };

  return (
    <ScrollablePageWrapper css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <PageHeader
        title={<FormattedMessage defaultMessage="Playground" description="Title of the LLM playground page" />}
        preview
      />
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
          <EndpointPicker value={endpointName} onChange={setEndpointName} />
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
          setMessages(loadedMessages.length > 0 ? loadedMessages : [{ role: 'user', content: '' }]);
          setShowRegistryPicker(false);
        }}
      />
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, PlaygroundPage);
