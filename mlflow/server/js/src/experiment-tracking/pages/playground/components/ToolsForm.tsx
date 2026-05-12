import {
  FormUI,
  Input,
  SegmentedControlButton,
  SegmentedControlGroup,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ToolChoice } from '../types';
import { isToolsValueEmpty } from '../utils';

const { TextArea } = Input;

interface Props {
  value: string;
  onChange: (next: string) => void;
  error?: string | null;
  toolChoice: ToolChoice;
  onToolChoiceChange: (next: ToolChoice) => void;
}

const TOOL_CHOICE_OPTIONS: { value: ToolChoice; label: string }[] = [
  { value: 'none', label: 'None' },
  { value: 'auto', label: 'Auto' },
  { value: 'required', label: 'Required' },
];

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

  const fieldGroupCss = {
    display: 'flex' as const,
    flexDirection: 'column' as const,
    gap: theme.spacing.xs,
  };

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <div css={fieldGroupCss}>
        <FormUI.Label htmlFor="mlflow.playground.tools.tool_choice">
          <FormattedMessage
            defaultMessage="Tool choice"
            description="Label above the tool choice segmented picker inside the Tools card"
          />
        </FormUI.Label>
        <SegmentedControlGroup
          componentId="mlflow.playground.tools.tool_choice"
          id="mlflow.playground.tools.tool_choice"
          name="mlflow.playground.tools.tool_choice"
          size="small"
          value={toolChoice}
          onChange={(event) => onToolChoiceChange(event.target.value as ToolChoice)}
        >
          {TOOL_CHOICE_OPTIONS.map(({ value: optionValue, label }) => (
            <SegmentedControlButton key={optionValue} value={optionValue}>
              {label}
            </SegmentedControlButton>
          ))}
        </SegmentedControlGroup>
      </div>

      {toolChoice !== 'none' && (
        <div css={fieldGroupCss}>
          <FormUI.Label htmlFor="mlflow.playground.tools.input">
            <FormattedMessage
              defaultMessage="JSON Tool Definition"
              description="Label for the JSON tool definitions textarea inside the Tools card"
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
          {isToolsValueEmpty(value) ? (
            <FormUI.Message
              type="error"
              message={intl.formatMessage({
                defaultMessage: 'Add at least one tool definition',
                description:
                  'Inline error shown in the Tools card when tool choice is not none but no tool definitions are provided',
              })}
            />
          ) : (
            error && <FormUI.Message type="error" message={error} />
          )}
        </div>
      )}
    </div>
  );
};
