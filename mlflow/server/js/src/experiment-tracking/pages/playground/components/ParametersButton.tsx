import {
  Button,
  Drawer,
  InfoIcon,
  Popover,
  Spacer,
  Typography,
  WrenchIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
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
  toolAdded: boolean;
  onAddTool: () => void;
  onRemoveTool: () => void;
  toolChoice: ToolChoice;
  onToolChoiceChange: (next: ToolChoice) => void;
  responseFormatType: ResponseFormatType;
  onResponseFormatTypeChange: (next: ResponseFormatType) => void;
  responseFormatSchemaText: string;
  onResponseFormatSchemaChange: (next: string) => void;
  responseFormatSchemaError?: string | null;
}

const infoTriggerCss = (theme: ReturnType<typeof useDesignSystemTheme>['theme']) => ({
  border: 0,
  background: 'none',
  padding: 0,
  display: 'inline-flex' as const,
  cursor: 'pointer' as const,
  color: theme.colors.textSecondary,
  '&:hover': { color: theme.colors.textPrimary },
});

const sectionHeaderCss = (theme: ReturnType<typeof useDesignSystemTheme>['theme']) => ({
  display: 'flex' as const,
  alignItems: 'center' as const,
  gap: theme.spacing.xs,
});

export const ParametersButton = ({
  value,
  onChange,
  toolsText,
  onToolsChange,
  toolsError,
  toolAdded,
  onAddTool,
  onRemoveTool,
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
        title={
          <span
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              fontSize: theme.typography.fontSizeXl,
              fontWeight: theme.typography.typographyBoldFontWeight,
              // Keep the line box generous enough for descenders (the `g` in
              // "Settings") to render fully inside the design system's
              // `overflow: hidden` title container.
              lineHeight: theme.typography.lineHeightXl,
            }}
          >
            <WrenchIcon />
            <FormattedMessage defaultMessage="Settings" description="Title of the playground settings drawer" />
          </span>
        }
      >
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
          <Typography.Paragraph withoutMargins>
            <FormattedMessage
              defaultMessage="Configure how the model handles your messages, tools, and structured responses."
              description="Intro paragraph at the top of the playground settings drawer"
            />
          </Typography.Paragraph>

          <Spacer size="sm" />

          <div css={sectionHeaderCss(theme)}>
            <Typography.Title level={4} withoutMargins>
              <FormattedMessage
                defaultMessage="Parameters"
                description="Section header for sampling parameters inside the playground settings drawer"
              />
            </Typography.Title>
            <Popover.Root componentId="mlflow.playground.params.section.parameters_help">
              <Popover.Trigger
                aria-label={intl.formatMessage({
                  defaultMessage: 'About sampling parameters',
                  description: 'Aria label for the info popover next to the Parameters section header',
                })}
                css={infoTriggerCss(theme)}
              >
                <InfoIcon />
              </Popover.Trigger>
              <Popover.Content align="start" css={{ maxWidth: 360 }}>
                <Typography.Paragraph withoutMargins>
                  <FormattedMessage
                    defaultMessage="Sampling controls. Adjust how creative or deterministic the model's responses are. Leave a field blank to use the provider's default. Some parameters may not be supported by the provider you are using."
                    description="Body of the info popover next to the Parameters section header."
                  />
                </Typography.Paragraph>
                <Popover.Arrow />
              </Popover.Content>
            </Popover.Root>
          </div>
          <ParametersForm value={value} onChange={onChange} />

          <Spacer size="md" />

          <div css={sectionHeaderCss(theme)}>
            <Typography.Title level={4} withoutMargins>
              <FormattedMessage
                defaultMessage="Tools"
                description="Section header for tool definitions inside the playground settings drawer"
              />
            </Typography.Title>
            <Popover.Root componentId="mlflow.playground.params.section.tools_help">
              <Popover.Trigger
                aria-label={intl.formatMessage({
                  defaultMessage: 'About tool definitions',
                  description: 'Aria label for the info popover next to the Tools section header',
                })}
                css={infoTriggerCss(theme)}
              >
                <InfoIcon />
              </Popover.Trigger>
              <Popover.Content align="start" css={{ maxWidth: 360 }}>
                <Typography.Paragraph withoutMargins>
                  <FormattedMessage
                    defaultMessage="Have the model call functions you define. By default no tool is configured. Click ‘Add tool’ to define a tool, then pick how the model uses it:"
                    description="Intro line of the info popover next to the Tools section header describing the add-tool flow"
                  />
                </Typography.Paragraph>
                <ul css={{ margin: `${theme.spacing.xs}px 0 0`, paddingLeft: theme.spacing.lg }}>
                  <li>
                    <Typography.Text bold>Auto</Typography.Text>{' '}
                    <FormattedMessage
                      defaultMessage="— model decides whether to call a tool."
                      description="Description of the Auto tool choice in the Tools info popover"
                    />
                  </li>
                  <li>
                    <Typography.Text bold>Required</Typography.Text>{' '}
                    <FormattedMessage
                      defaultMessage="— model must call a tool."
                      description="Description of the Required tool choice in the Tools info popover"
                    />
                  </li>
                </ul>
                <Popover.Arrow />
              </Popover.Content>
            </Popover.Root>
          </div>
          <div
            css={{
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.general.borderRadiusBase,
              padding: theme.spacing.md,
            }}
          >
            <ToolsForm
              value={toolsText}
              onChange={onToolsChange}
              error={toolsError}
              toolAdded={toolAdded}
              onAddTool={onAddTool}
              onRemoveTool={onRemoveTool}
              toolChoice={toolChoice}
              onToolChoiceChange={onToolChoiceChange}
            />
          </div>

          <Spacer size="md" />

          <div css={sectionHeaderCss(theme)}>
            <Typography.Title level={4} withoutMargins>
              <FormattedMessage
                defaultMessage="Response format"
                description="Section header for the structured-output picker inside the playground settings drawer"
              />
            </Typography.Title>
            <Popover.Root componentId="mlflow.playground.params.section.response_format_help">
              <Popover.Trigger
                aria-label={intl.formatMessage({
                  defaultMessage: 'About response formats',
                  description: 'Aria label for the info popover next to the Response format section header',
                })}
                css={infoTriggerCss(theme)}
              >
                <InfoIcon />
              </Popover.Trigger>
              <Popover.Content align="start" css={{ maxWidth: 360 }}>
                <Typography.Paragraph withoutMargins>
                  <FormattedMessage
                    defaultMessage="Constrain how the model formats its output:"
                    description="Intro line in the info popover next to the Response format section header"
                  />
                </Typography.Paragraph>
                <ul css={{ margin: `${theme.spacing.xs}px 0 0`, paddingLeft: theme.spacing.lg }}>
                  <li>
                    <Typography.Text bold>Text</Typography.Text>{' '}
                    <FormattedMessage
                      defaultMessage="— model's natural-language response."
                      description="Description of the Text mode in the response_format info popover"
                    />
                  </li>
                  <li>
                    <Typography.Text bold>JSON</Typography.Text>{' '}
                    <FormattedMessage
                      defaultMessage="— model returns valid JSON."
                      description="Description of the JSON mode in the response_format info popover"
                    />
                  </li>
                  <li>
                    <Typography.Text bold>JSON schema</Typography.Text>{' '}
                    <FormattedMessage
                      defaultMessage="— model output conforms to the schema you provide."
                      description="Description of the JSON schema mode in the response_format info popover"
                    />
                  </li>
                </ul>
                <Popover.Arrow />
              </Popover.Content>
            </Popover.Root>
          </div>
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
