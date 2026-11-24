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
}

/** The inline error under an autosaved setting, with Retry when the save itself failed. */
export const SettingSaveErrorMessage = ({ id, message, onRetry, componentId }: SettingSaveErrorMessageProps) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      id={id}
      css={{
        display: 'flex',
        // Retry lines up with the message's first line when the message wraps.
        alignItems: 'baseline',
        gap: theme.spacing.xs,
        marginBlockStart: theme.spacing.xs,
        // The value column is right-aligned; the message reads from its icon.
        textAlign: 'start',
      }}
    >
      <FormUI.Message type="error" message={message} />
      {onRetry && (
        <Button componentId={componentId} type="link" onClick={onRetry}>
          <FormattedMessage defaultMessage="Retry" description="Link-styled button to retry saving a setting" />
        </Button>
      )}
    </div>
  );
};
