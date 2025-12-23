import { useMemo, useState } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { GenAiTracesTableSearchInput } from '@databricks/web-shared/genai-traces-table';
import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { TracesV3DateSelector } from '../../components/experiment-page/components/traces-v3/TracesV3DateSelector';
import { useMonitoringFilters, getAbsoluteStartEndTime } from '../../hooks/useMonitoringFilters';
import { MonitoringConfigProvider, useMonitoringConfig } from '../../hooks/useMonitoringConfig';
import { TraceRequestsChart } from './components/TraceRequestsChart';

enum OverviewTab {
  Usage = 'usage',
}

const ExperimentGenAIOverviewPageImpl = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [activeTab, setActiveTab] = useState<OverviewTab>(OverviewTab.Usage);
  const [searchQuery, setSearchQuery] = useState('');

  invariant(experimentId, 'Experiment ID must be defined');

  // Get the current time range from monitoring filters
  const [monitoringFilters] = useMonitoringFilters();
  const monitoringConfig = useMonitoringConfig();

  // Use getAbsoluteStartEndTime to properly compute time range from labels
  const { startTime, endTime } = useMemo(
    () => getAbsoluteStartEndTime(monitoringConfig.dateNow, monitoringFilters),
    [monitoringConfig.dateNow, monitoringFilters],
  );

  // Convert ISO strings to milliseconds for the API
  const startTimeMs = startTime ? new Date(startTime).getTime() : undefined;
  const endTimeMs = endTime ? new Date(endTime).getTime() : undefined;

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        overflow: 'hidden',
      }}
    >
      <Tabs.Root
        componentId="mlflow.experiment.overview.tabs"
        value={activeTab}
        onValueChange={(value) => setActiveTab(value as OverviewTab)}
        css={{ display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}
      >
        <Tabs.List>
          <Tabs.Trigger value={OverviewTab.Usage}>
            <FormattedMessage
              defaultMessage="Usage"
              description="Label for the usage tab in the experiment overview page"
            />
          </Tabs.Trigger>
        </Tabs.List>

        {/* Control bar with search and time range */}
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            padding: `${theme.spacing.sm}px 0`,
          }}
        >
          {/* Search input */}
          <GenAiTracesTableSearchInput
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            placeholder={intl.formatMessage({
              defaultMessage: 'Search charts',
              description: 'Placeholder for search charts input',
            })}
          />

          {/* Time range selector */}
          <TracesV3DateSelector />
        </div>

        <Tabs.Content value={OverviewTab.Usage} css={{ flex: 1, overflowY: 'auto' }}>
          <div css={{ padding: `${theme.spacing.sm}px 0` }}>
            <TraceRequestsChart experimentId={experimentId} startTimeMs={startTimeMs} endTimeMs={endTimeMs} />
          </div>
        </Tabs.Content>
      </Tabs.Root>
    </div>
  );
};

// Wrap in MonitoringConfigProvider so refresh button updates are received
const ExperimentGenAIOverviewPage = () => (
  <MonitoringConfigProvider>
    <ExperimentGenAIOverviewPageImpl />
  </MonitoringConfigProvider>
);

export default ExperimentGenAIOverviewPage;
