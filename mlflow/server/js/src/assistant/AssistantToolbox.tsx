/**
 * Session toolbox: a single full-access switch. When on, the assistant runs
 * tool calls without asking; when off, each tool call surfaces a Yes/No prompt.
 * The setting is session-scoped and never written to the global config.
 */

import { Switch, Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';

import { useAssistant } from './AssistantContext';

export const AssistantToolbox = () => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const { sessionFullAccess, setSessionFullAccess } = useAssistant();

  return (
    <Tooltip
      componentId="mlflow.assistant.toolbox.full_access.tooltip"
      content={
        <FormattedMessage
          defaultMessage="Full access runs tool calls without asking for permission each time. Applies to this session only."
          description="Tooltip explaining the assistant session full-access switch"
        />
      }
    >
      <span css={{ display: 'inline-flex', alignItems: 'center', marginRight: theme.spacing.xs }}>
        <Switch
          componentId="mlflow.assistant.toolbox.full_access"
          checked={sessionFullAccess}
          onChange={setSessionFullAccess}
          label={intl.formatMessage({
            defaultMessage: 'Full access',
            description: 'Label for the assistant session full-access switch',
          })}
        />
      </span>
    </Tooltip>
  );
};
