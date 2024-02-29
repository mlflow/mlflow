import { Tabs } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { RunPageTabName } from '../../constants';
import { useRunViewActiveTab } from './useRunViewActiveTab';
import { useState } from 'react';
import { shouldEnableLoggedArtifactTableView } from 'common/utils/FeatureUtils';

/**
 * Mode switcher for the run details page.
 */
export const RunViewModeSwitch = () => {
  const { experimentId, runUuid } = useParams<{ runUuid: string; experimentId: string }>();
  const navigate = useNavigate();
  const currentTab = useRunViewActiveTab();
  const [removeTabMargin, setRemoveTabMargin] = useState(
    shouldEnableLoggedArtifactTableView() && currentTab === RunPageTabName.ARTIFACTS,
  );

  const onTabChanged = (newTabKey: string) => {
    if (!experimentId || !runUuid || currentTab === newTabKey) {
      return;
    }
    if (shouldEnableLoggedArtifactTableView() && newTabKey === RunPageTabName.ARTIFACTS) {
      setRemoveTabMargin(true);
    } else {
      setRemoveTabMargin(false);
    }
    if (newTabKey === RunPageTabName.OVERVIEW) {
      navigate(Routes.getRunPageRoute(experimentId, runUuid));
      return;
    }
    navigate(Routes.getRunPageTabRoute(experimentId, runUuid, newTabKey));
  };

  return (
    // @ts-expect-error TS(2322)
    <Tabs activeKey={currentTab} onChange={onTabChanged} tabBarStyle={{ margin: removeTabMargin && '0px' }}>
      <Tabs.TabPane
        tab={
          <FormattedMessage defaultMessage="Overview" description="Run details page > tab selector > overview tab" />
        }
        key={RunPageTabName.OVERVIEW}
      />
      <Tabs.TabPane
        tab={
          <FormattedMessage
            defaultMessage="Model metrics"
            description="Run details page > tab selector > Model metrics tab"
          />
        }
        key={RunPageTabName.MODEL_METRIC_CHARTS}
      />
      <Tabs.TabPane
        tab={
          <FormattedMessage
            defaultMessage="System metrics"
            description="Run details page > tab selector > Model metrics tab"
          />
        }
        key={RunPageTabName.SYSTEM_METRIC_CHARTS}
      />
      <Tabs.TabPane
        tab={
          <FormattedMessage defaultMessage="Artifacts" description="Run details page > tab selector > artifacts tab" />
        }
        key={RunPageTabName.ARTIFACTS}
      />
    </Tabs>
  );
};
