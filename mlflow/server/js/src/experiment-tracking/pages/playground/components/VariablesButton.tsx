import { Button, CodeIcon, Drawer, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChatMessage } from '../types';
import { extractTemplateVariables } from '../utils';
import { VariablesForm } from './VariablesForm';

interface Props {
  messages: ChatMessage[];
  value: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}

export const VariablesButton = ({ messages, value, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const variableNames = useMemo(() => extractTemplateVariables(messages), [messages]);

  const triggerLabel = (
    <FormattedMessage
      defaultMessage="Variables{count, plural, =0 {} other { ({count})}}"
      description="Label for the playground top-bar button that opens variable values; suffix shows the count when > 0"
      values={{ count: variableNames.length }}
    />
  );

  return (
    <Drawer.Root>
      <Drawer.Trigger>
        <Button
          componentId="mlflow.playground.variables.drawer.trigger"
          icon={<CodeIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Open variable values',
            description: 'Aria label for the top-bar button that opens the playground variables drawer',
          })}
        >
          {triggerLabel}
        </Button>
      </Drawer.Trigger>
      <Drawer.Content
        componentId="mlflow.playground.variables.drawer"
        title={intl.formatMessage({
          defaultMessage: 'Variables',
          description: 'Title of the playground variables drawer',
        })}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Wrap a name in {syntax} inside any message to define a variable. Each variable becomes an input here. Values set below will be substituted in at submit time."
              description="Help text inside the playground variables drawer"
              values={{ syntax: <Typography.Text code>{'{{ }}'}</Typography.Text> }}
            />
          </Typography.Hint>
          <VariablesForm messages={messages} value={value} onChange={onChange} />
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};
