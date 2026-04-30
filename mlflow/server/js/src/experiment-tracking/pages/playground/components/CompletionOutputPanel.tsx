import { Alert, Empty, Spinner, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { GenAIMarkdownRenderer } from '@databricks/web-shared/genai-markdown-renderer';
import { FormattedMessage } from 'react-intl';
import type { ChatCompletionResponse } from '../types';

interface Props {
  response?: ChatCompletionResponse;
  error?: Error;
  isLoading: boolean;
}

const getCompletionContent = (response?: ChatCompletionResponse): string => {
  if (!response) {
    return '';
  }
  return response.choices?.[0]?.message?.content ?? '';
};

export const CompletionOutputPanel = ({ response, error, isLoading }: Props) => {
  const { theme } = useDesignSystemTheme();
  const completion = getCompletionContent(response);
  const usage = response?.usage;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        flex: 1,
        minHeight: 0,
      }}
    >
      <Typography.Title level={4} withoutMargins>
        <FormattedMessage
          defaultMessage="Output"
          description="Section header for the completion output panel on the playground page"
        />
      </Typography.Title>

      <div
        css={{
          border: `1px solid ${theme.colors.border}`,
          borderRadius: theme.general.borderRadiusBase,
          padding: theme.spacing.md,
          backgroundColor: theme.colors.backgroundSecondary,
          minHeight: 200,
          flex: 1,
          overflow: 'auto',
        }}
      >
        {isLoading ? (
          <div css={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
            <Spinner />
          </div>
        ) : error ? (
          <Alert type="error" message={error.message} componentId="mlflow.playground.output.error" closable={false} />
        ) : completion ? (
          <GenAIMarkdownRenderer>{completion}</GenAIMarkdownRenderer>
        ) : (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              minHeight: 160,
              '& > div': {
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
              },
            }}
          >
            <Empty
              description={
                <FormattedMessage
                  defaultMessage="Submit a message to see the response here."
                  description="Empty state for the playground completion output panel"
                />
              }
            />
          </div>
        )}
      </div>

      {usage && (
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Tokens — prompt: {prompt}, completion: {completion}, total: {total}"
            description="Token usage footer on the playground completion output panel"
            values={{
              prompt: usage.prompt_tokens ?? '—',
              completion: usage.completion_tokens ?? '—',
              total: usage.total_tokens ?? '—',
            }}
          />
        </Typography.Hint>
      )}
    </div>
  );
};
