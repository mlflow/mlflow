import { useMemo } from 'react';
import { FormattedMessage } from 'react-intl';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';

import { useAssistantConfigQuery } from '../hooks/useAssistantConfigQuery';
import { AssistantSettingsForm } from './AssistantSettingsForm';

interface AssistantSettingsPageProps {
  experimentId?: string;
  onBack: () => void;
}

export const AssistantSettingsPage = ({ experimentId, onBack }: AssistantSettingsPageProps) => {
  const { theme } = useDesignSystemTheme();
  const { config } = useAssistantConfigQuery();
  const selectedProvider = useMemo(
    () => Object.entries(config?.providers ?? {}).find(([, provider]) => provider.selected)?.[0] ?? 'claude_code',
    [config],
  );

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        padding: theme.spacing.lg,
        overflow: 'auto',
      }}
    >
      <Typography.Title level={4} css={{ marginBottom: theme.spacing.sm }}>
        <FormattedMessage defaultMessage="Settings" description="Title for the MLflow Assistant settings page" />
      </Typography.Title>

      <div css={{ flex: 1, overflow: 'auto' }}>
        <AssistantSettingsForm
          experimentId={experimentId}
          provider={selectedProvider}
          onBack={onBack}
          onComplete={onBack}
          nextLabel="Save"
          backLabel="Cancel"
        />
      </div>
    </div>
  );
};
