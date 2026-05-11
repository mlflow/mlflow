import { FormUI, Input, Typography, useDesignSystemTheme } from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';

const { TextArea } = Input;

interface Props {
  value: string;
  onChange: (next: string) => void;
  error?: string | null;
}

const TOOLS_PLACEHOLDER = `[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": { "type": "string" }
        },
        "required": ["location"]
      }
    }
  }
]`;

export const ToolsForm = ({ value, onChange, error }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <FormUI.Label htmlFor="mlflow.playground.tools.input">
        <FormattedMessage
          defaultMessage="Tools (JSON array)"
          description="Label for the playground tools JSON textarea inside the settings drawer"
        />
      </FormUI.Label>
      <TextArea
        componentId="mlflow.playground.tools.input"
        id="mlflow.playground.tools.input"
        value={value}
        onChange={(event: ChangeEvent<HTMLTextAreaElement>) => onChange(event.target.value)}
        autoSize={{ minRows: 4, maxRows: 16 }}
        placeholder={intl.formatMessage(
          {
            defaultMessage: 'e.g. {example}',
            description: 'Placeholder shown above the playground tools JSON textarea',
          },
          { example: TOOLS_PLACEHOLDER },
        )}
        css={{
          fontFamily: 'monospace',
          fontSize: theme.typography.fontSizeSm,
        }}
      />
      {error ? (
        <FormUI.Message type="error" message={error} />
      ) : (
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Provide an array of tool definitions the model is allowed to call. Leave blank to disable tool use."
            description="Help text under the playground tools textarea"
          />
        </Typography.Hint>
      )}
    </div>
  );
};
