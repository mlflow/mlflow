import {
  Button,
  FormUI,
  IndentIncreaseIcon,
  Input,
  PlusIcon,
  SegmentedControlButton,
  SegmentedControlGroup,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import type { ChangeEvent } from 'react';
import { useState } from 'react';
import { FormattedMessage, useIntl, type IntlShape } from 'react-intl';
import type { PlaygroundTool, ToolChoice } from '../types';
import type { ToolParametersError } from '../utils';
import { formatJson, getToolParametersError } from '../utils';
import { LazyJsonRecordEditor } from '../../experiment-evaluation-datasets-v2/components/LazyJsonRecordEditor';

// Maps a structured parameters-validation result to a localized, user-visible string.
const formatToolParametersError = (error: ToolParametersError, intl: IntlShape): string => {
  switch (error.code) {
    case 'empty':
      return intl.formatMessage({
        defaultMessage: 'Add a parameters schema',
        description: 'Inline error shown when a tool parameters editor is empty',
      });
    case 'parseError':
      return intl.formatMessage(
        {
          defaultMessage: 'Invalid JSON: {detail}',
          description: 'Inline error shown when a tool parameters editor contains unparseable JSON',
        },
        { detail: error.detail },
      );
    case 'notObject':
      return intl.formatMessage({
        defaultMessage: 'Parameters must be a JSON object',
        description: 'Inline error shown when tool parameters are valid JSON but not an object',
      });
    case 'missingProperties':
      return intl.formatMessage({
        defaultMessage: 'Parameters schema must include a "properties" object',
        description: 'Inline error shown when a tool parameters schema is missing its properties map',
      });
  }
};

interface Props {
  tools: PlaygroundTool[];
  onAddTool: () => void;
  onRemoveTool: (id: string) => void;
  onUpdateTool: (id: string, patch: Partial<PlaygroundTool>) => void;
  toolChoice: ToolChoice;
  onToolChoiceChange: (next: ToolChoice) => void;
}

const TOOL_CHOICE_OPTIONS: { value: ToolChoice; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: 'required', label: 'Required' },
];

export const ToolsForm = ({ tools, onAddTool, onRemoveTool, onUpdateTool, toolChoice, onToolChoiceChange }: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  // Track which tool name fields have been blurred so a freshly added tool does
  // not flash a "required" error before the user has touched it.
  const [touchedNames, setTouchedNames] = useState<Record<string, boolean>>({});

  if (tools.length === 0) {
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
          <Button componentId="mlflow.playground.tools.add" icon={<PlusIcon />} onClick={onAddTool}>
            <FormattedMessage
              defaultMessage="Add tools"
              description="Label for the button that reveals the playground tool definitions input and tool choice selector"
            />
          </Button>
        </div>
        <Typography.Hint>
          <FormattedMessage
            defaultMessage="Provide one or more function definitions for the model to call."
            description="Help text shown under the Add tools button in the playground settings drawer before any tools are added"
          />
        </Typography.Hint>
      </div>
    );
  }

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FormUI.Label htmlFor="mlflow.playground.tools.tool_choice" css={{ marginBottom: theme.spacing.sm }}>
          <FormattedMessage
            defaultMessage="Tool choice"
            description="Label above the tool choice segmented picker inside the Tools card"
          />
        </FormUI.Label>
        <SegmentedControlGroup
          componentId="mlflow.playground.tools.tool_choice"
          id="mlflow.playground.tools.tool_choice"
          name="mlflow.playground.tools.tool_choice"
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

      {tools.map((tool, index) => {
        const toolNumber = index + 1;
        const nameMissing = tool.name.trim().length === 0;
        const showNameError = nameMissing && (touchedNames[tool.id] ?? false);
        const paramsError = getToolParametersError(tool.params);
        const nameId = `mlflow.playground.tools.name.${tool.id}`;
        const descriptionId = `mlflow.playground.tools.description.${tool.id}`;
        const paramsId = `mlflow.playground.tools.params.${tool.id}`;
        return (
          <div
            key={tool.id}
            css={{
              display: 'flex',
              flexDirection: 'column',
              gap: theme.spacing.sm,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.general.borderRadiusBase,
              padding: theme.spacing.md,
            }}
          >
            <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <Typography.Text bold>
                <FormattedMessage
                  defaultMessage="Tool {number}"
                  description="Header label for an individual tool card in the playground Tools section"
                  values={{ number: toolNumber }}
                />
              </Typography.Text>
              <Button
                componentId="mlflow.playground.tools.remove"
                type="tertiary"
                size="small"
                icon={<TrashIcon />}
                onClick={() => onRemoveTool(tool.id)}
                aria-label={intl.formatMessage(
                  {
                    defaultMessage: 'Remove tool {number}',
                    description:
                      'Aria label for the trash button that removes a single tool from the playground request',
                  },
                  { number: toolNumber },
                )}
                css={{
                  '& svg': { color: theme.colors.textSecondary },
                  '&:hover svg': { color: theme.colors.textPrimary },
                }}
              />
            </div>

            <div>
              <FormUI.Label htmlFor={nameId}>
                <FormattedMessage
                  defaultMessage="Function name"
                  description="Label for the tool function name input in the playground Tools card"
                />
              </FormUI.Label>
              <Input
                componentId="mlflow.playground.tools.name"
                id={nameId}
                value={tool.name}
                onChange={(event: ChangeEvent<HTMLInputElement>) => onUpdateTool(tool.id, { name: event.target.value })}
                onBlur={() => setTouchedNames((prev) => ({ ...prev, [tool.id]: true }))}
                validationState={showNameError ? 'error' : undefined}
                placeholder={intl.formatMessage({
                  defaultMessage: 'e.g. get_weather',
                  description: 'Placeholder for the tool function name input in the playground Tools card',
                })}
              />
              {showNameError && (
                <FormUI.Message
                  type="error"
                  message={intl.formatMessage({
                    defaultMessage: 'Function name is required',
                    description: 'Inline error shown when a tool is missing its function name',
                  })}
                />
              )}
            </div>

            <div>
              <FormUI.Label htmlFor={descriptionId}>
                <FormattedMessage
                  defaultMessage="Function description"
                  description="Label for the tool description input in the playground Tools card"
                />
              </FormUI.Label>
              <Input
                componentId="mlflow.playground.tools.description"
                id={descriptionId}
                value={tool.description}
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  onUpdateTool(tool.id, { description: event.target.value })
                }
                placeholder={intl.formatMessage({
                  defaultMessage: 'e.g. Get the weather in a given location',
                  description: 'Placeholder for the tool description input in the playground Tools card',
                })}
              />
            </div>

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
                <FormUI.Label id={paramsId}>
                  <FormattedMessage
                    defaultMessage="Function parameters schema"
                    description="Label for the tool parameters JSON schema editor in the playground Tools card"
                  />
                </FormUI.Label>
                <Button
                  componentId="mlflow.playground.tools.format"
                  size="small"
                  icon={<IndentIncreaseIcon />}
                  disabled={paramsError !== null}
                  onClick={() => {
                    const formatted = formatJson(tool.params);
                    if (formatted !== null) {
                      onUpdateTool(tool.id, { params: formatted });
                    }
                  }}
                >
                  <FormattedMessage
                    defaultMessage="Format"
                    description="Button that pretty-prints the tool parameters JSON in the playground"
                  />
                </Button>
              </div>
              <LazyJsonRecordEditor
                ariaLabel={intl.formatMessage(
                  {
                    defaultMessage: 'Tool {number} parameters',
                    description: 'Accessible label for an individual tool parameters JSON editor in the playground',
                  },
                  { number: toolNumber },
                )}
                value={tool.params}
                onChange={(next) => onUpdateTool(tool.id, { params: next })}
                labelledById={paramsId}
                height="160px"
                maxHeight="360px"
                transparentBackground
                errorMessage={paramsError ? formatToolParametersError(paramsError, intl) : undefined}
              />
            </div>
          </div>
        );
      })}

      <Button
        componentId="mlflow.playground.tools.add_tool"
        icon={<PlusIcon />}
        onClick={onAddTool}
        css={{ width: '100%', borderStyle: 'dotted' }}
      >
        <FormattedMessage
          defaultMessage="Add tool"
          description="Label for the button that appends another tool definition card in the playground Tools section"
        />
      </Button>
    </div>
  );
};
