import { useState } from 'react';
import invariant from 'invariant';
import { useParams } from '../../../common/utils/RoutingUtils';
import { TableFilterInput, TableFilterLayout, Tabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { TracesV3DateSelector } from '../../components/experiment-page/components/traces-v3/TracesV3DateSelector';

enum OverviewTab {
  Usage = 'usage',
}

const ExperimentOverviewPage = () => {
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
          <TableFilterLayout css={{ marginBottom: 0 }}>
            {/* Search input */}
            <TableFilterInput
              componentId="mlflow.experiment.overview.search-charts"
              placeholder={intl.formatMessage({
                defaultMessage: 'Search charts',
                description: 'Placeholder for search charts input',
              })}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </TableFilterLayout>

          {/* Time range selector - same as traces tab */}
          <TracesV3DateSelector />
        </div>

        <Tabs.Content value={OverviewTab.Usage}>{/* Usage tab content */}</Tabs.Content>
      </Tabs.Root>
    </div>
  );
};

export default ExperimentOverviewPage;
