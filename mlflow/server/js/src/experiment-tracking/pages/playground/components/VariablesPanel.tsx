import { FormUI, InfoSmallIcon, Input, Popover, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { Fragment, useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChatMessage } from '../types';
import { extractTemplateVariables } from '../utils';

const { TextArea } = Input;

interface Props {
  messages: ChatMessage[];
  value: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}

export const VariablesPanel = ({ messages, value, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const variableNames = useMemo(() => extractTemplateVariables(messages), [messages]);

  if (variableNames.length === 0) {
    return null;
  }

  const handleField = (name: string) => (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange({ ...value, [name]: event.target.value });
  };

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        padding: theme.spacing.md,
      }}
    >
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
        <Typography.Title level={4} withoutMargins>
          <FormattedMessage
            defaultMessage="Variables"
            description="Section header for the variable-substitution panel on the playground page"
          />
        </Typography.Title>
        <Popover.Root componentId="mlflow.playground.variables.help">
          <Popover.Trigger
            aria-label={intl.formatMessage({
              defaultMessage: 'About prompt variables',
              description: 'Aria label for the info popover next to the playground variables header',
            })}
            css={{ border: 0, background: 'none', padding: 0, display: 'inline-flex', cursor: 'pointer' }}
          >
            <InfoSmallIcon />
          </Popover.Trigger>
          <Popover.Content align="start" css={{ maxWidth: 360 }}>
            <Typography.Paragraph withoutMargins>
              <FormattedMessage
                defaultMessage="Each placeholder in your messages becomes an input here. Values are substituted into a copy of the messages at submit time; the templates above stay intact so you can re-run with different inputs."
                description="Help text in the playground variables panel popover"
              />
            </Typography.Paragraph>
            <Popover.Arrow />
          </Popover.Content>
        </Popover.Root>
      </div>

      {variableNames.map((name) => {
        const inputId = `mlflow.playground.variables.input.${name}`;
        return (
          <Fragment key={name}>
            <FormUI.Label htmlFor={inputId}>{name}</FormUI.Label>
            <TextArea
              componentId="mlflow.playground.variables.input"
              id={inputId}
              value={value[name] ?? ''}
              onChange={handleField(name)}
              autoSize={{ minRows: 1, maxRows: 8 }}
              placeholder={intl.formatMessage(
                {
                  defaultMessage: 'Value for {name}',
                  description: 'Placeholder for a variable-value textarea on the playground page',
                },
                { name },
              )}
            />
          </Fragment>
        );
      })}
    </div>
  );
};
