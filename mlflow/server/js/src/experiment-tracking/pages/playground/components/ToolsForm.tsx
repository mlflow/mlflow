import {
  Button,
  FormUI,
  Input,
  PlusIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  TrashIcon,
  Typography,
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
  toolsAdded: boolean;
  onToolsAddedChange: (next: boolean) => void;
  toolChoice: ToolChoice;
  onToolChoiceChange: (next: ToolChoice) => void;
}

const TOOL_CHOICE_OPTIONS: { value: ToolChoice; label: string }[] = [
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

export const ToolsForm = ({
  value,
  onChange,
  error,
  toolsAdded,
  onToolsAddedChange,
  toolChoice,
  onToolChoiceChange,
}: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  if (!toolsAdded) {
    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.general.borderRadiusBase,
            padding: theme.spacing.lg,
            minHeight: 104,
          }}
        >
          <Button
            componentId="mlflow.playground.tools.add"
            icon={<PlusIcon />}
            onClick={() => onToolsAddedChange(true)}
          >
            <FormattedMessage
              defaultMessage="Add tools"
              description="Label for the button that reveals the playground tool definitions input and tool choice selector"
            />
          </Button>
        </div>
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Provide one or more function definitions in a JSON array for the model to call."
            description="Help text shown under the Add tools button in the playground settings drawer before any tools are added"
          />
        </Typography.Hint>
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        padding: theme.spacing.md,
      }}
    >
      <div>
        <div
          css={{
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
            marginBottom: theme.spacing.sm,
            '& label': { marginBottom: 0 },
          }}
        >
          <FormUI.Label htmlFor="mlflow.playground.tools.tool_choice">
            <FormattedMessage
              defaultMessage="Tool choice"
              description="Label above the tool choice segmented picker inside the Tools card"
            />
          </FormUI.Label>
          <Button
            componentId="mlflow.playground.tools.remove"
            type="tertiary"
            size="small"
            icon={<TrashIcon />}
            onClick={() => onToolsAddedChange(false)}
            aria-label={intl.formatMessage({
              defaultMessage: 'Remove tools',
              description: 'Aria label for the trash button that removes the playground tools from the request',
            })}
            css={{
              '& svg': { color: theme.colors.textSecondary },
              '&:hover svg': { color: theme.colors.textPrimary },
            }}
          />
        </div>
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
        <Typography.Hint css={{ display: 'block', marginTop: theme.spacing.xs }}>
          {toolChoice === 'required' ? (
            <FormattedMessage
              defaultMessage="The model must call at least one tool."
              description="Hint shown under the tool choice selector when Required is selected"
            />
          ) : (
            <FormattedMessage
              defaultMessage="The model decides whether to call one or more tools."
              description="Hint shown under the tool choice selector when Auto is selected"
            />
          )}
        </Typography.Hint>
      </div>

      <div>
        <FormUI.Label htmlFor="mlflow.playground.tools.input">
          <FormattedMessage
            defaultMessage="JSON tool definitions"
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
                'Inline error shown in the Tools card when tools are added but no tool definitions are provided',
            })}
          />
        ) : (
          error && <FormUI.Message type="error" message={error} />
        )}
      </div>
    </div>
  );
};
