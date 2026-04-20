import { NavigationMenu } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { RunPageTabName } from '../../constants';
import { useRunViewActiveTab } from './useRunViewActiveTab';
import type { KeyValueEntity } from '../../../common/types';

// Set of tabs that when active, the margin of the tab selector should be removed for better displaying
const TABS_WITHOUT_MARGIN = [RunPageTabName.ARTIFACTS, RunPageTabName.EVALUATIONS];

/**
 * Mode switcher for the run details page.
 */
export const RunViewModeSwitch = ({ runTags }: { runTags: Record<string, KeyValueEntity> }) => {
  const { experimentId, runUuid } = useParams<{ runUuid: string; experimentId: string }>();
  const currentTab = useRunViewActiveTab();

  if (!experimentId || !runUuid) {
    return null;
  }

  const tabs: Array<{ key: RunPageTabName; label: JSX.Element }> = [
    {
      key: RunPageTabName.OVERVIEW,
      label: (
        <FormattedMessage defaultMessage="Overview" description="Run details page > tab selector > overview tab" />
      ),
    },
    {
      key: RunPageTabName.MODEL_METRIC_CHARTS,
      label: (
        <FormattedMessage
          defaultMessage="Model metrics"
          description="Run details page > tab selector > Model metrics tab"
        />
      ),
    },
    {
      key: RunPageTabName.SYSTEM_METRIC_CHARTS,
      label: (
        <FormattedMessage
          defaultMessage="System metrics"
          description="Run details page > tab selector > Model metrics tab"
        />
      ),
    },
    {
      key: RunPageTabName.EVALUATIONS,
      label: <FormattedMessage defaultMessage="Traces" description="Run details page > tab selector > Traces tab" />,
    },
    {
      key: RunPageTabName.ARTIFACTS,
      label: (
        <FormattedMessage defaultMessage="Artifacts" description="Run details page > tab selector > artifacts tab" />
      ),
    },
  ];

  const getTabUrl = (tabKey: RunPageTabName) =>
    tabKey === RunPageTabName.OVERVIEW
      ? Routes.getRunPageRoute(experimentId, runUuid)
      : Routes.getRunPageTabRoute(experimentId, runUuid, tabKey);

  return (
    <NavigationMenu.Root aria-label="Run details navigation">
      <NavigationMenu.List css={{ marginBottom: TABS_WITHOUT_MARGIN.includes(currentTab) ? 0 : undefined }}>
        {tabs.map((tab) => (
          <NavigationMenu.Item key={tab.key} active={currentTab === tab.key}>
            <Link componentId="mlflow.run.details.tabs" to={getTabUrl(tab.key)}>
              {tab.label}
            </Link>
          </NavigationMenu.Item>
        ))}
      </NavigationMenu.List>
    </NavigationMenu.Root>
  );
};
