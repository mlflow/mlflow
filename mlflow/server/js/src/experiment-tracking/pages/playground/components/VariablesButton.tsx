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
        title={
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              fontSize: theme.typography.fontSizeXl,
              fontWeight: theme.typography.typographyBoldFontWeight,
              lineHeight: theme.typography.lineHeightXl,
            }}
          >
            <CodeIcon />
            <FormattedMessage defaultMessage="Variables" description="Title of the playground variables drawer" />
          </span>
        }
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <Typography.Paragraph withoutMargins>
            <FormattedMessage
              defaultMessage="Use {example} to create reusable variables. Values entered below replace the placeholders on submit, while your templates stay unchanged for reuse with different inputs."
              description="Intro paragraph at the top of the playground variables drawer"
              values={{ example: <Typography.Text code>{'{{ name }}'}</Typography.Text> }}
            />
          </Typography.Paragraph>
          <VariablesForm messages={messages} value={value} onChange={onChange} />
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};
