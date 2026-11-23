import { Button, FormUI, useDesignSystemTheme } from '@databricks/design-system';
import type { ReactNode } from 'react';
import { FormattedMessage } from 'react-intl';

export interface SettingSaveErrorMessageProps {
  id: string;
  message: ReactNode;
  /** Shows a Retry button, for a save the server rejected. */
  onRetry?: () => void;
  /** For the Retry button. */
  componentId: string;
  retryDisabled?: boolean;
}

/** The inline error under an autosaved setting, with Retry when the save itself failed. */
export const SettingSaveErrorMessage = ({
  id,
  message,
  onRetry,
  componentId,
  retryDisabled = false,
}: SettingSaveErrorMessageProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      id={id}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        marginBlockStart: theme.spacing.xs,
      }}
    >
      <FormUI.Message type="error" message={message} />
      {onRetry && (
        <Button
          componentId={componentId}
          type="link"
          onClick={retryDisabled ? undefined : onRetry}
          disabled={retryDisabled}
        >
          <FormattedMessage defaultMessage="Retry" description="Link-styled button to retry saving a setting" />
        </Button>
      )}
    </div>
  );
};
