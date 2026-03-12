import { LegacyTabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { RunPageTabName } from '../../constants';
import { useRunViewActiveTab } from './useRunViewActiveTab';
import { useState, type ReactNode } from 'react';
import type { KeyValueEntity } from '../../../common/types';

// Set of tabs that when active, the margin of the tab selector should be removed for better displaying
const TABS_WITHOUT_MARGIN = [RunPageTabName.ARTIFACTS, RunPageTabName.EVALUATIONS];

// Default tabs to show in the run view
const DEFAULT_VISIBLE_TABS = [
  RunPageTabName.OVERVIEW,
  RunPageTabName.MODEL_METRIC_CHARTS,
  RunPageTabName.SYSTEM_METRIC_CHARTS,
  RunPageTabName.EVALUATIONS,
  RunPageTabName.ARTIFACTS,
];

// Tab label configuration
const TAB_LABELS: Record<RunPageTabName, ReactNode> = {
  [RunPageTabName.OVERVIEW]: (
    <FormattedMessage defaultMessage="Overview" description="Run details page > tab selector > overview tab" />
  ),
  [RunPageTabName.MODEL_METRIC_CHARTS]: (
    <FormattedMessage
      defaultMessage="Model metrics"
      description="Run details page > tab selector > Model metrics tab"
    />
  ),
  [RunPageTabName.SYSTEM_METRIC_CHARTS]: (
    <FormattedMessage
      defaultMessage="System metrics"
      description="Run details page > tab selector > System metrics tab"
    />
  ),
  [RunPageTabName.EVALUATIONS]: (
    <FormattedMessage defaultMessage="Traces" description="Run details page > tab selector > Traces tab" />
  ),
  [RunPageTabName.TRACES]: (
    <FormattedMessage defaultMessage="Traces" description="Run details page > tab selector > Traces tab" />
  ),
  [RunPageTabName.ISSUES]: (
    <FormattedMessage defaultMessage="Issues" description="Run details page > tab selector > issues tab" />
  ),
  [RunPageTabName.ARTIFACTS]: (
    <FormattedMessage defaultMessage="Artifacts" description="Run details page > tab selector > artifacts tab" />
  ),
};

export interface RunViewModeSwitchProps {
  runTags?: Record<string, KeyValueEntity>;
  /** Custom function to get the base route (overview tab) */
  getBaseRoute?: (experimentId: string, runUuid: string) => string;
  /** Custom function to get the route for a specific tab */
  getTabRoute?: (experimentId: string, runUuid: string, tabName: string) => string;
  /** List of tabs to show. Defaults to all tabs. */
  visibleTabs?: RunPageTabName[];
}

/**
 * Mode switcher for the run details page.
 */
export const RunViewModeSwitch = ({
  runTags,
  getBaseRoute,
  getTabRoute,
  visibleTabs = DEFAULT_VISIBLE_TABS,
}: RunViewModeSwitchProps) => {
  const { experimentId, runUuid } = useParams<{ runUuid: string; experimentId: string }>();
  const navigate = useNavigate();
  const { theme } = useDesignSystemTheme();
  const currentTab = useRunViewActiveTab();
  const [removeTabMargin, setRemoveTabMargin] = useState(TABS_WITHOUT_MARGIN.includes(currentTab));

  const onTabChanged = (newTabKey: string) => {
    if (!experimentId || !runUuid || currentTab === newTabKey) {
      return;
    }

    setRemoveTabMargin(TABS_WITHOUT_MARGIN.includes(newTabKey as RunPageTabName));

    if (newTabKey === RunPageTabName.OVERVIEW) {
      const route = getBaseRoute ? getBaseRoute(experimentId, runUuid) : Routes.getRunPageRoute(experimentId, runUuid);
      navigate(route);
      return;
    }
    const route = getTabRoute
      ? getTabRoute(experimentId, runUuid, newTabKey)
      : Routes.getRunPageTabRoute(experimentId, runUuid, newTabKey);
    navigate(route);
  };

  return (
    // @ts-expect-error TS(2322)
    <LegacyTabs activeKey={currentTab} onChange={onTabChanged} tabBarStyle={{ margin: removeTabMargin && '0px' }}>
      {visibleTabs.map((tabName) => (
        <LegacyTabs.TabPane tab={TAB_LABELS[tabName]} key={tabName} />
      ))}
    </LegacyTabs>
  );
};
