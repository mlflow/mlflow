import { Button, Tooltip, Typography, useDesignSystemTheme, Input } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { useTryIt, type TryItFetchOptions } from '../../hooks/useTryIt';

const TEXTAREA_ROWS = 14;
const TEXTAREA_MIN_HEIGHT_PX = 220;
const ERROR_LINE_MIN_HEIGHT = '1.5em';

export interface TryItPanelProps {
  description: ReactNode;
  requestTooltipContent: ReactNode;
  requestTooltipComponentId: string;
  tryItRequestUrl: string;
  tryItDefaultBody: string;
  /** Optional fetch options (headers, signal) passed through to Try-it requests. */
  tryItOptions?: TryItFetchOptions;
}

export const TryItPanel = ({
  description,
  requestTooltipContent,
  requestTooltipComponentId,
  tryItRequestUrl,
  tryItDefaultBody,
  tryItOptions,
}: TryItPanelProps) => {
  const { theme } = useDesignSystemTheme();
  const [requestBody, setRequestBody] = useState(tryItDefaultBody);

  const {
    data,
    isLoading,
    error,
    sendRequest,
    reset: resetTryIt,
  } = useTryIt({
    tryItRequestUrl,
    options: tryItOptions,
  });

  const responseBody = error ? (error.responseBody ?? '') : (data ?? '');
  const sendError = error?.message ?? null;

  const handleSendRequest = useCallback(() => {
    sendRequest(requestBody);
  }, [sendRequest, requestBody]);

  const handleResetExample = useCallback(() => {
    resetTryIt();
    setRequestBody(tryItDefaultBody);
  }, [resetTryIt, tryItDefaultBody]);

  // When default body changes (e.g. variant or provider changed), reset to the new default
  useEffect(() => {
    handleResetExample();
  }, [handleResetExample]);

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
            disabled={isLoading}
            rows={TEXTAREA_ROWS}
            css={{
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
              minHeight: TEXTAREA_MIN_HEIGHT_PX,
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
          <div aria-live="polite" aria-atomic="true" role="status">
            <Input.TextArea
              componentId="mlflow.gateway.usage-modal.try-it.response"
              value={responseBody}
              readOnly
              rows={TEXTAREA_ROWS}
              placeholder={
                sendError ? undefined : isLoading ? undefined : 'Click "Send request" to see the response here.'
              }
              css={{
                fontFamily: 'monospace',
                fontSize: theme.typography.fontSizeSm,
                backgroundColor: theme.colors.backgroundSecondary,
                minHeight: TEXTAREA_MIN_HEIGHT_PX,
              }}
            />
          </div>
          <div
            css={{ minHeight: ERROR_LINE_MIN_HEIGHT, marginTop: theme.spacing.xs }}
            aria-live="polite"
            aria-atomic="true"
            role="status"
          >
            {sendError && <Typography.Text color="error">{sendError}</Typography.Text>}
          </div>
        </div>
      </div>
      <div css={{ display: 'flex', gap: theme.spacing.sm }}>
        <Button
          componentId="mlflow.gateway.usage-modal.try-it.send"
          type="primary"
          onClick={handleSendRequest}
          disabled={isLoading}
          loading={isLoading}
        >
          {isLoading ? (
            <FormattedMessage defaultMessage="Sending..." description="Send request button loading state" />
          ) : (
            <FormattedMessage defaultMessage="Send request" description="Send request button" />
          )}
        </Button>
        <Button componentId="mlflow.gateway.usage-modal.try-it.reset" onClick={handleResetExample} disabled={isLoading}>
          <FormattedMessage defaultMessage="Reset example" description="Reset example button" />
        </Button>
      </div>
    </div>
  );
};
