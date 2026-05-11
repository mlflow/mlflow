import { Button, Drawer, Spacer, Typography, WrenchIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { PlaygroundParams, ResponseFormatType, ToolChoice } from '../types';
import { ParametersForm } from './ParametersForm';
import { ResponseFormatForm } from './ResponseFormatForm';
import { ToolsForm } from './ToolsForm';

interface Props {
  value: PlaygroundParams;
  onChange: (next: PlaygroundParams) => void;
  toolsText: string;
  onToolsChange: (next: string) => void;
  toolsError?: string | null;
  toolChoice: ToolChoice;
  onToolChoiceChange: (next: ToolChoice) => void;
  responseFormatType: ResponseFormatType;
  onResponseFormatTypeChange: (next: ResponseFormatType) => void;
  responseFormatSchemaText: string;
  onResponseFormatSchemaChange: (next: string) => void;
  responseFormatSchemaError?: string | null;
}

export const ParametersButton = ({
  value,
  onChange,
  toolsText,
  onToolsChange,
  toolsError,
  toolChoice,
  onToolChoiceChange,
  responseFormatType,
  onResponseFormatTypeChange,
  responseFormatSchemaText,
  onResponseFormatSchemaChange,
  responseFormatSchemaError,
}: Props) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  return (
    <Drawer.Root>
      <Drawer.Trigger>
        <Button
          componentId="mlflow.playground.params.drawer.trigger"
          icon={<WrenchIcon />}
          aria-label={intl.formatMessage({
            defaultMessage: 'Open model parameters',
            description: 'Aria label for the wrench button that opens the playground parameters drawer',
          })}
        >
          <FormattedMessage
            defaultMessage="Settings"
            description="Label for the playground top-bar button that opens model parameters"
          />
        </Button>
      </Drawer.Trigger>
      <Drawer.Content
        componentId="mlflow.playground.params.drawer"
        width={420}
        title={intl.formatMessage({
          defaultMessage: 'Settings',
          description: 'Title of the playground settings drawer',
        })}
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Title level={4} withoutMargins>
            <FormattedMessage
              defaultMessage="Parameters"
              description="Section header for sampling parameters inside the playground settings drawer"
            />
          </Typography.Title>
          <Typography.Hint>
            <FormattedMessage
              defaultMessage="Leave a field blank to use the provider's default."
              description="Help text inside the playground parameters section of the settings drawer"
            />
          </Typography.Hint>
          <ParametersForm value={value} onChange={onChange} />

          <Spacer size="md" />

          <Typography.Title level={4} withoutMargins>
            <FormattedMessage
              defaultMessage="Tools"
              description="Section header for tool definitions inside the playground settings drawer"
            />
          </Typography.Title>
          <ToolsForm
            value={toolsText}
            onChange={onToolsChange}
            error={toolsError}
            toolChoice={toolChoice}
            onToolChoiceChange={onToolChoiceChange}
          />

          <Spacer size="md" />

          <Typography.Title level={4} withoutMargins>
            <FormattedMessage
              defaultMessage="Response format"
              description="Section header for the structured-output picker inside the playground settings drawer"
            />
          </Typography.Title>
          <ResponseFormatForm
            type={responseFormatType}
            onTypeChange={onResponseFormatTypeChange}
            schemaText={responseFormatSchemaText}
            onSchemaChange={onResponseFormatSchemaChange}
            schemaError={responseFormatSchemaError}
          />
        </div>
      </Drawer.Content>
    </Drawer.Root>
  );
};
