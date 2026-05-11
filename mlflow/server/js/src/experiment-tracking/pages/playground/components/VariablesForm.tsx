import { FormUI, Input, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { Fragment } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ChatMessage } from '../types';
import { extractTemplateVariables } from '../utils';

const { TextArea } = Input;

interface Props {
  messages: ChatMessage[];
  value: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
}

export const VariablesForm = ({ messages, value, onChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const variableNames = extractTemplateVariables(messages);

  if (variableNames.length === 0) {
    return (
      <Typography.Hint>
        <FormattedMessage
          defaultMessage="No variables detected. Add a placeholder like {example} to a message to define one."
          description="Empty-state hint shown inside the playground variables popover when no variables are detected"
          values={{ example: <Typography.Text code>{'{{ name }}'}</Typography.Text> }}
        />
      </Typography.Hint>
    );
  }

  const handleField = (name: string) => (event: ChangeEvent<HTMLTextAreaElement>) => {
    onChange({ ...value, [name]: event.target.value });
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
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
