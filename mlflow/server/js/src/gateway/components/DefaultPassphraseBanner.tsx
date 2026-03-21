import { useCallback } from 'react';
import { Alert, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useLocalStorage } from '../../shared/web-shared/hooks/useLocalStorage';

export const DefaultPassphraseBanner = () => {
  const { theme } = useDesignSystemTheme();
  const [isDismissed, setIsDismissed] = useLocalStorage({
    key: 'mlflow.gateway.default-passphrase-warning.dismissed',
    version: 1,
    initialValue: false,
  });

  const handleClose = useCallback(() => {
    setIsDismissed(true);
  }, [setIsDismissed]);

  if (isDismissed) {
    return null;
  }

  return (
    <div css={{ padding: `0 ${theme.spacing.md}px`, paddingTop: theme.spacing.md, paddingBottom: theme.spacing.md }}>
      <Alert
        componentId="mlflow.gateway.default-passphrase-warning"
        type="warning"
        closable
        onClose={handleClose}
        message={
          <FormattedMessage
            defaultMessage="Security Notice: Default Passphrase in Use"
            description="Gateway > Default passphrase warning banner title"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="The AI Gateway is using the default encryption passphrase. This is acceptable for development or single-user deployments, but for multi-user production environments, you should rotate the passphrase using the CLI command: mlflow crypto rotate-kek"
            description="Gateway > Default passphrase warning banner description"
          />
        }
      />
    </div>
  );
};
