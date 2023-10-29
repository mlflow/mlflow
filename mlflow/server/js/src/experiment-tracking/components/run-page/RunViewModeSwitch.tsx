import { Tabs } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { Link, useNavigate, useParams } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { RunPageTabName } from '../../constants';
import { useRunViewActiveTab } from './useRunViewActiveTab';

/**
 * Mode switcher for the run details page.
 */
export const RunViewModeSwitch = () => {
  const { experimentId, runUuid } = useParams<{ runUuid: string; experimentId: string }>();
  const navigate = useNavigate();
  const currentTab = useRunViewActiveTab();

  const onTabChanged = (newTabKey: string) => {
    if (!experimentId || !runUuid || currentTab === newTabKey) {
      return;
    }
    if (newTabKey === RunPageTabName.OVERVIEW) {
      navigate(Routes.getRunPageRoute(experimentId, runUuid));
      return;
    }
    navigate(Routes.getRunPageTabRoute(experimentId, runUuid, newTabKey));
  };

  return (
    <Tabs activeKey={currentTab} onChange={onTabChanged}>
      <Tabs.TabPane
        tab={
          <FormattedMessage
            defaultMessage='Overview'
            description='Run details page > tab selector > overview tab'
          />
        }
        key={RunPageTabName.OVERVIEW}
      />
      <Tabs.TabPane
        tab={
          <FormattedMessage
            defaultMessage='Metric charts'
            description='Run details page > tab selector > metric charts tab'
          />
        }
        key={RunPageTabName.CHARTS}
      />
      <Tabs.TabPane
        tab={
          <FormattedMessage
            defaultMessage='Artifacts'
            description='Run details page > tab selector > artifacts tab'
          />
        }
        key={RunPageTabName.ARTIFACTS}
      />
    </Tabs>
  );
};
