import { useMemo } from 'react';
import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../../hooks/useMonitoringConfig';
import { TracesV3PageWrapper } from '../../../components/experiment-page/components/traces-v3/TracesV3PageWrapper';
import { useMonitoringViewState } from '../../../hooks/useMonitoringViewState';
import { getAbsoluteStartEndTime, useMonitoringFilters } from '../../../hooks/useMonitoringFilters';
import { TracesV3Toolbar } from '../../../components/experiment-page/components/traces-v3/TracesV3Toolbar';
import { TracesV3Logs } from '../../../components/experiment-page/components/traces-v3/TracesV3Logs';
import { FilterOperator } from '@databricks/web-shared/genai-traces-table';
import type { RegisteredPromptVersion } from '../types';

/**
 * Displays traces filtered by a specific prompt version.
 * Shows the traces table with an automatic filter applied for the selected prompt.
 */
const PromptFilteredTracesViewImpl = ({
  experimentId,
  promptVersion,
}: {
  experimentId: string;
  promptVersion: RegisteredPromptVersion;
}) => {
  const { theme } = useDesignSystemTheme();
  const [monitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();
  const [viewState] = useMonitoringViewState();

  const promptName = promptVersion.name;
  const versionNumber = promptVersion.version;

  const timeRange = useMemo(() => {
    const { startTime, endTime } = getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters);
    return {
      startTime: startTime ? new Date(startTime).getTime().toString() : undefined,
      endTime: endTime ? new Date(endTime).getTime().toString() : undefined,
    };
  }, [monitoringConfig.dateNow, monitoringFilters]);

  // Create initial filters for the prompt
  const initialFilters = useMemo(() => {
    return [
      {
        column: 'prompt',
        operator: FilterOperator.EQUALS,
        value: `${promptName}/${versionNumber}`,
      },
    ];
  }, [promptName, versionNumber]);

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.sm,
        height: '100%',
        overflowY: 'hidden',
      }}
    >
      <TracesV3Toolbar viewState={viewState} />
      {viewState === 'logs' && (
        <TracesV3Logs
          experimentId={experimentId || ''}
          endpointName={''}
          timeRange={timeRange}
          initialFilters={initialFilters}
        />
      )}
    </div>
  );
};

export const PromptFilteredTracesView = ({
  promptVersion,
  experimentId,
}: {
  promptVersion?: RegisteredPromptVersion;
  experimentId?: string;
}) => {
  const { theme } = useDesignSystemTheme();

  if (!experimentId) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Paragraph>
          <FormattedMessage
            defaultMessage="Traces are only available for experiment-scoped prompts."
            description="Message when prompt is not experiment-scoped"
          />
        </Typography.Paragraph>
      </div>
    );
  }

  if (!promptVersion?.name || !promptVersion?.version) {
    return (
      <div css={{ padding: theme.spacing.lg }}>
        <Typography.Paragraph>
          <FormattedMessage
            defaultMessage="No prompt version selected. Select a prompt version to view associated traces."
            description="Empty state message when no prompt version is selected"
          />
        </Typography.Paragraph>
      </div>
    );
  }

  return (
    <TracesV3PageWrapper>
      <MonitoringConfigProvider>
        <PromptFilteredTracesViewImpl experimentId={experimentId} promptVersion={promptVersion} />
      </MonitoringConfigProvider>
    </TracesV3PageWrapper>
  );
};
