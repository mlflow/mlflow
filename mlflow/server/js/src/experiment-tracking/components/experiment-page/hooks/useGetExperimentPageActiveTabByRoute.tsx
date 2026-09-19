import { useMemo } from 'react';
import { matchPath, useLocation } from '../../../../common/utils/RoutingUtils';
import { RoutePaths } from '../../../routes';
import { ExperimentPageTabName } from '../../../constants';
import { map } from 'lodash';
import { shouldEnableSessionGrouping } from '@databricks/web-shared/genai-traces-table';

// Maps experiment page route paths to enumerated tab names
const ExperimentPageRoutePathToTabNameMap = map(
  {
    [RoutePaths.experimentPageTabOverview]: ExperimentPageTabName.Overview,
    [RoutePaths.experimentPageTabRuns]: ExperimentPageTabName.Runs,
    [RoutePaths.experimentPageTabTraces]: ExperimentPageTabName.Traces,
    [RoutePaths.experimentPageTabModels]: ExperimentPageTabName.Models,
    [RoutePaths.experimentPageTabEvaluationRuns]: ExperimentPageTabName.EvaluationRuns,
    [RoutePaths.experimentPageTabDatasets]: ExperimentPageTabName.Datasets,
    [RoutePaths.experimentPageTabChatSessions]: ExperimentPageTabName.ChatSessions,
    [RoutePaths.experimentPageTabSingleChatSession]: ExperimentPageTabName.SingleChatSession,
    [RoutePaths.experimentPageTabScorers]: ExperimentPageTabName.Judges,
    [RoutePaths.experimentPageTabPlayground]: ExperimentPageTabName.Playground,
    // OSS experiment prompt page routes
    [RoutePaths.experimentPageTabPrompts]: ExperimentPageTabName.Prompts,
    [RoutePaths.experimentPageTabPromptDetails]: ExperimentPageTabName.Prompts,
    [RoutePaths.experimentPageTabReviewQueue]: ExperimentPageTabName.ReviewQueue,
  },
  (tabName, routePath) => ({ routePath, tabName }),
);

// Gets exact tab name based on given pathname
const getTabNameFromRoutePath = (pathname: string) => {
  const tabName = ExperimentPageRoutePathToTabNameMap
    // Find the first route path that matches the given pathname
    .find(({ routePath }) => Boolean(matchPath(routePath, pathname)))?.tabName;
  // The Sessions view is deprecated in favor of the Traces tab's session grouping, so single
  // chat session routes resolve to the Traces tab when session grouping is enabled.
  if (tabName === ExperimentPageTabName.SingleChatSession && shouldEnableSessionGrouping()) {
    return ExperimentPageTabName.Traces;
  }
  return tabName;
};

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
