import { LegacyTabs, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { RunPageTabName } from '../../constants';
import { useRunViewActiveTab } from './useRunViewActiveTab';
import { useState } from 'react';

// Set of tabs that when active, the margin of the tab selector should be removed for better displaying
const TABS_WITHOUT_MARGIN = [RunPageTabName.ARTIFACTS, RunPageTabName.EVALUATIONS];

/**
 * Mode switcher for the run details page.
 */
export const RunViewModeSwitch = () => {
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
      navigate(Routes.getRunPageRoute(experimentId, runUuid));
      return;
    }
    navigate(Routes.getRunPageTabRoute(experimentId, runUuid, newTabKey));
  };

  return (
    // @ts-expect-error TS(2322)
    <LegacyTabs activeKey={currentTab} onChange={onTabChanged} tabBarStyle={{ margin: removeTabMargin && '0px' }}>
      <LegacyTabs.TabPane
        tab={
          <FormattedMessage defaultMessage="Overview" description="Run details page > tab selector > overview tab" />
        }
        key={RunPageTabName.OVERVIEW}
      />

      <LegacyTabs.TabPane
        tab={
          <FormattedMessage
            defaultMessage="Model metrics"
            description="Run details page > tab selector > Model metrics tab"
          />
        }
        key={RunPageTabName.MODEL_METRIC_CHARTS}
      />
      <LegacyTabs.TabPane
        tab={
          <FormattedMessage
            defaultMessage="System metrics"
            description="Run details page > tab selector > Model metrics tab"
          />
        }
        key={RunPageTabName.SYSTEM_METRIC_CHARTS}
      />
      <LegacyTabs.TabPane
        tab={<FormattedMessage defaultMessage="Traces" description="Run details page > tab selector > Traces tab" />}
        key={RunPageTabName.EVALUATIONS}
      />
      <LegacyTabs.TabPane
        tab={
          <FormattedMessage defaultMessage="Artifacts" description="Run details page > tab selector > artifacts tab" />
        }
        key={RunPageTabName.ARTIFACTS}
      />
    </LegacyTabs>
  );
};
