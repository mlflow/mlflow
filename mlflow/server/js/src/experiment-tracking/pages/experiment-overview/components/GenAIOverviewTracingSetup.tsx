import {
  Button,
  CopyIcon,
  Drawer,
  LoadingIcon,
  Tabs,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { CopyButton } from '../../../../shared/building_blocks/CopyButton';
import { TracesViewTableNoTracesQuickstart } from '../../../components/traces/quickstart/TracesViewTableNoTracesQuickstart';

const AGENT_SETUP_SCRIPT_URL = 'https://mlflow.org/wizard/setup.sh';

const shellQuote = (value: string) => `'${value.replaceAll("'", `'"'"'`)}'`;

const getAgentSetupCommand = ({ serverUrl, experimentId }: { serverUrl: string; experimentId: string }) => {
  const args = ['--tracking-uri', shellQuote(serverUrl), '--experiment-id', shellQuote(experimentId)];

  return `curl -LsSf ${AGENT_SETUP_SCRIPT_URL} | sh -s -- ${args.join(' ')}`;
};

export const WaitingForFirstTrace = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      role="status"
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: theme.spacing.sm,
        boxSizing: 'border-box',
        flexShrink: 0,
        width: theme.spacing.xl * 5,
        padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
        borderRadius: theme.borders.borderRadiusFull,
        backgroundColor: theme.colors.actionDefaultBackgroundHover,
        color: theme.colors.actionTertiaryTextDefault,
        whiteSpace: 'nowrap',
      }}
    >
      <LoadingIcon spin aria-hidden="true" />
      <span>
        <FormattedMessage
          defaultMessage="Waiting for traces…"
          description="Status shown while MLflow waits for the first trace"
        />
      </span>
    </div>
  );
};

const SetupCopyBox = ({
  ariaLabel,
  componentId,
  text,
  useMonospace = false,
}: {
  ariaLabel: string;
  componentId: string;
  text: string;
  useMonospace?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: theme.spacing.sm,
        padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundSecondary,
      }}
    >
      <Typography.Text
        css={{
          minWidth: 0,
          flex: 1,
          overflow: useMonospace ? 'hidden' : undefined,
          textOverflow: useMonospace ? 'ellipsis' : undefined,
          fontFamily: useMonospace ? 'monospace' : undefined,
          whiteSpace: useMonospace ? 'nowrap' : 'normal',
        }}
      >
        {text}
      </Typography.Text>
      <CopyButton
        componentId={componentId}
        copyText={text}
        showLabel={false}
        size="small"
        type="tertiary"
        icon={<CopyIcon />}
        aria-label={ariaLabel}
      />
    </div>
  );
};

export const TracingSetup = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const agentSetupCommand = getAgentSetupCommand({
    serverUrl: window.location.origin,
    experimentId: experimentIds[0],
  });
  const agentSetupPrompt = intl.formatMessage(
    {
      defaultMessage:
        'Install the MLflow AI skill from https://github.com/mlflow/skills and use it to add tracing to this application following MLflow best practices. Configure tracing to use MLflow tracking server {serverUrl} and experiment ID {experimentId}. Run one traced operation and confirm the trace reaches the experiment. Do not create a new experiment or write credentials into the repository.',
      description: 'Coding agent prompt for setting up tracing',
    },
    {
      serverUrl: window.location.origin,
      experimentId: experimentIds[0],
    },
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
      <Tabs.Root componentId="mlflow.genai-overview.tracing-quickstart.tabs" defaultValue="cli">
        <Tabs.List tabListCss={{ marginBottom: theme.spacing.mid }}>
          <Tabs.Trigger value="cli">
            <FormattedMessage defaultMessage="CLI" description="CLI tracing setup tab" />
          </Tabs.Trigger>
          <Tabs.Trigger value="ai-assistant">
            <FormattedMessage defaultMessage="AI Assistant" description="AI assistant tracing setup tab" />
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="cli">
          <SetupCopyBox
            componentId="mlflow.genai-overview.tracing-quickstart.quick-setup.copy"
            text={agentSetupCommand}
            useMonospace
            ariaLabel={intl.formatMessage({
              defaultMessage: 'Copy setup command',
              description: 'Accessible label for the button that copies the tracing setup command',
            })}
          />
        </Tabs.Content>
        <Tabs.Content value="ai-assistant">
          <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
            <Typography.Text color="secondary" size="sm">
              <FormattedMessage
                defaultMessage="Copy this prompt into Claude, Cursor, Copilot, or another coding agent."
                description="Instructions for using the tracing setup prompt with a coding agent"
              />
            </Typography.Text>
            <SetupCopyBox
              componentId="mlflow.genai-overview.tracing-quickstart.ai-assistant.copy"
              text={agentSetupPrompt}
              ariaLabel={intl.formatMessage({
                defaultMessage: 'Copy AI assistant prompt',
                description: 'Accessible label for the button that copies the tracing setup prompt',
              })}
            />
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};

export const ManualTracingSetupLink = ({ experimentIds }: { experimentIds: string[] }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <Drawer.Root>
      <Drawer.Trigger>
        <Button componentId="mlflow.genai-overview.tracing-quickstart.manual-setup.open" type="tertiary" size="small">
          <span css={{ color: theme.colors.textSecondary, textDecoration: 'underline' }}>
            <FormattedMessage
              defaultMessage="Set up manually instead"
              description="Link to open the manual tracing setup drawer"
            />
          </span>
        </Button>
      </Drawer.Trigger>
      <Drawer.Content
        componentId="mlflow.genai-overview.tracing-quickstart.manual-setup"
        width="70vw"
        title={
          <FormattedMessage defaultMessage="Set up tracing" description="Title for the manual tracing setup drawer" />
        }
      >
        <TracesViewTableNoTracesQuickstart
          baseComponentId="mlflow.genai-overview.tracing-quickstart.manual-setup"
          experimentId={experimentIds[0]}
        />
      </Drawer.Content>
    </Drawer.Root>
  );
};
