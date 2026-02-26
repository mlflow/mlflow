import { Button, Tooltip, Typography, useDesignSystemTheme, Input } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { useTryIt } from '../../hooks/useTryIt';

export interface TryItPanelProps {
  description: ReactNode;
  requestTooltipContent: ReactNode;
  requestTooltipComponentId: string;
  tryItRequestUrl: string;
  tryItDefaultBody: string;
}

export const TryItPanel = ({
  description,
  requestTooltipContent,
  requestTooltipComponentId,
  tryItRequestUrl,
  tryItDefaultBody,
}: TryItPanelProps) => {
  const { theme } = useDesignSystemTheme();
  const [requestBody, setRequestBody] = useState(tryItDefaultBody);
  const [responseBody, setResponseBody] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [sendError, setSendError] = useState<string | null>(null);

  const onSuccess = useCallback((formattedResponse: string) => {
    setResponseBody(formattedResponse);
    setSendError(null);
  }, []);

  const onError = useCallback((error: Error, responseText?: string) => {
    setSendError(error.message);
    setResponseBody(responseText ?? '');
  }, []);

  const onSendingChange = useCallback((sending: boolean) => {
    setIsSending(sending);
    if (sending) {
      setSendError(null);
      setResponseBody('');
    }
  }, []);

  const onReset = useCallback(() => {
    setRequestBody(tryItDefaultBody);
    setResponseBody('');
    setSendError(null);
  }, [tryItDefaultBody]);

  const { handleSendRequest, handleResetExample } = useTryIt({
    requestBody,
    tryItRequestUrl,
    onSuccess,
    onError,
    onSendingChange,
    onReset,
  });

  // When default body changes (e.g. variant or provider changed), reset to the new default
  useEffect(() => {
    setRequestBody(tryItDefaultBody);
    setResponseBody('');
    setSendError(null);
  }, [tryItDefaultBody]);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <Typography.Text color="secondary">{description}</Typography.Text>
      <div
        css={{
          display: 'flex',
          gap: theme.spacing.md,
          minHeight: 0,
          flex: 1,
        }}
      >
        <div css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              marginBottom: theme.spacing.xs,
            }}
          >
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Request" description="Request body label" />
            </Typography.Text>
            <Tooltip componentId={requestTooltipComponentId} content={requestTooltipContent}>
              <span css={{ cursor: 'help', color: theme.colors.textSecondary }} aria-label="Request help">
                ?
              </span>
            </Tooltip>
          </div>
          <Input.TextArea
            componentId="mlflow.gateway.usage-modal.try-it.request"
            value={requestBody}
            onChange={(e) => setRequestBody(e.target.value)}
            disabled={isSending}
            rows={14}
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              minHeight: 220,
            }}
          />
        </div>
        <div css={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column' }}>
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              marginBottom: theme.spacing.xs,
            }}
          >
            <Typography.Text bold>
              <FormattedMessage defaultMessage="Response" description="Response body label" />
            </Typography.Text>
            <Tooltip
              componentId="mlflow.gateway.usage-modal.try-it.response-tooltip"
              content={
                <FormattedMessage
                  defaultMessage="Response from the endpoint after clicking Send request."
                  description="Response body tooltip"
                />
              }
            >
              <span css={{ cursor: 'help', color: theme.colors.textSecondary }} aria-label="Response help">
                ?
              </span>
            </Tooltip>
          </div>
          <Input.TextArea
            componentId="mlflow.gateway.usage-modal.try-it.response"
            value={responseBody}
            readOnly
            rows={14}
            placeholder={
              sendError ? undefined : isSending ? undefined : 'Click "Send request" to see the response here.'
            }
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              backgroundColor: theme.colors.backgroundSecondary,
              minHeight: 220,
            }}
          />
          {sendError && (
            <Typography.Text color="error" css={{ marginTop: theme.spacing.xs }}>
              {sendError}
            </Typography.Text>
          )}
        </div>
      </div>
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <Button
          componentId="mlflow.gateway.usage-modal.try-it.send"
          type="primary"
          onClick={handleSendRequest}
          disabled={isSending}
          loading={isSending}
        >
          {isSending ? (
            <FormattedMessage defaultMessage="Sending..." description="Send request button loading state" />
          ) : (
            <FormattedMessage defaultMessage="Send request" description="Send request button" />
          )}
        </Button>
        <Button componentId="mlflow.gateway.usage-modal.try-it.reset" onClick={handleResetExample} disabled={isSending}>
          <FormattedMessage defaultMessage="Reset example" description="Reset example button" />
        </Button>
      </div>
    </div>
  );
};
