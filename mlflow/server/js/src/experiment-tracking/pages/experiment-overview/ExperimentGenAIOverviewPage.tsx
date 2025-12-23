import { useState } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { TracesV3DateSelector } from '../../components/experiment-page/components/traces-v3/TracesV3DateSelector';
import { GenAiTracesTableSearchInput } from '@databricks/web-shared/genai-traces-table';

enum OverviewTab {
  Usage = 'usage',
}

const ExperimentGenAIOverviewPage = () => {
  const { experimentId } = useParams();
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [activeTab, setActiveTab] = useState<OverviewTab>(OverviewTab.Usage);
  const [searchQuery, setSearchQuery] = useState('');

  invariant(experimentId, 'Experiment ID must be defined');

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
            padding: `${theme.spacing.md}px 0`,
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

        <Tabs.Content value={OverviewTab.Usage} />
      </Tabs.Root>
    </div>
  );
};

export default ExperimentGenAIOverviewPage;
