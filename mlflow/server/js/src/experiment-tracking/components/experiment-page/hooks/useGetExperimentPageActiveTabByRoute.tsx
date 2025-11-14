import { useMemo } from 'react';
import { useLocation } from '../../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../../routes';
import { ExperimentPageTabName } from '../../../constants';
import { map } from 'lodash';
import { matchPathWithWorkspace } from '../../../../common/utils/WorkspaceRouteUtils';

// Maps experiment page route paths to enumerated tab names
const ExperimentPageRoutePathToTabNameMap = map(
  {
    [RoutePaths.experimentPageTabRuns]: ExperimentPageTabName.Runs,
    [RoutePaths.experimentPageTabTraces]: ExperimentPageTabName.Traces,
    [RoutePaths.experimentPageTabModels]: ExperimentPageTabName.Models,
    [RoutePaths.experimentPageTabEvaluationRuns]: ExperimentPageTabName.EvaluationRuns,
    [RoutePaths.experimentPageTabDatasets]: ExperimentPageTabName.Datasets,
    [RoutePaths.experimentPageTabChatSessions]: ExperimentPageTabName.ChatSessions,
    [RoutePaths.experimentPageTabSingleChatSession]: ExperimentPageTabName.SingleChatSession,
  },
  (tabName, routePath) => ({ routePath, tabName }),
);

// Gets exact tab name based on given pathname
const getTabNameFromRoutePath = (pathname: string) =>
  ExperimentPageRoutePathToTabNameMap
    // Find the first route path that matches the given pathname (workspace-aware)
    .find(({ routePath }) => Boolean(matchPathWithWorkspace(routePath, pathname)))?.tabName;

// Maps exact tab names to top-level tab names
const getTopLevelTab = (tabName?: ExperimentPageTabName) => {
  return tabName;
};

export const useGetExperimentPageActiveTabByRoute = () => {
  const { pathname } = useLocation();

  const tabNameFromRoute = useMemo(() => {
    const tabName = getTabNameFromRoutePath(pathname);
    return tabName;
  }, [pathname]);

  return {
    tabName: tabNameFromRoute,
    topLevelTabName: getTopLevelTab(tabNameFromRoute),
  };
};
