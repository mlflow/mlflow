import {
  FormUI,
  Input,
  SegmentedControlButton,
  SegmentedControlGroup,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ToolChoice } from '../types';

const { TextArea } = Input;

interface Props {
  value: string;
  onChange: (next: string) => void;
  error?: string | null;
  toolChoice: ToolChoice;
  onToolChoiceChange: (next: ToolChoice) => void;
}

const TOOL_CHOICE_OPTIONS: ToolChoice[] = ['auto', 'none', 'required'];

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

export const ToolsForm = ({ value, onChange, error, toolChoice, onToolChoiceChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const hasTools = value.trim().length > 0;

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

      {hasTools && (
        <>
          <FormUI.Label htmlFor="mlflow.playground.tools.tool_choice">
            <FormattedMessage
              defaultMessage="Tool choice"
              description="Label for the tool_choice picker on the playground tools section"
            />
          </FormUI.Label>
          <SegmentedControlGroup
            componentId="mlflow.playground.tools.tool_choice"
            name="mlflow.playground.tools.tool_choice"
            size="small"
            value={toolChoice}
            onChange={(event) => onToolChoiceChange(event.target.value as ToolChoice)}
          >
            {TOOL_CHOICE_OPTIONS.map((option) => (
              <SegmentedControlButton key={option} value={option}>
                {option}
              </SegmentedControlButton>
            ))}
          </SegmentedControlGroup>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="auto — model decides. none — never call a tool. required — must call a tool."
              description="Help text under the tool_choice picker on the playground tools section"
            />
          </Typography.Hint>
        </>
      )}
    </div>
  );
};
