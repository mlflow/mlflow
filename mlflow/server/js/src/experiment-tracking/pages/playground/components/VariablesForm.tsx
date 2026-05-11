import { FormUI, Input, useDesignSystemTheme } from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { Fragment } from 'react';
import { useIntl } from 'react-intl';
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
