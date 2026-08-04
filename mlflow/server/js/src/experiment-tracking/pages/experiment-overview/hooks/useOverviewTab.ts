import { useCallback } from 'react';
import { useNavigate, useParams, useLocation, generatePath } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { RoutePaths } from '../../../routes';

export enum OverviewTab {
  Usage = 'usage',
  Quality = 'quality',
  ToolCalls = 'tool-calls',
}

const DEFAULT_TAB = OverviewTab.Usage;

/**
 * Path param-powered hook that returns the active overview tab from the URL.
 * Uses URL path segments (e.g., /overview/usage) for shareable links.
 */
export const useOverviewTab = () => {
  const { experimentId, overviewTab } = useParams<{ experimentId: string; overviewTab?: string }>();
  const navigate = useNavigate();
  const location = useLocation();

  // Validate the tab value, fallback to default if invalid
  const activeTab = Object.values(OverviewTab).includes(overviewTab as OverviewTab)
    ? (overviewTab as OverviewTab)
    : DEFAULT_TAB;

  const setActiveTab = useCallback(
    (tab: OverviewTab) => {
      const path = generatePath(RoutePaths.experimentPageTabOverview, {
        experimentId: experimentId || '',
        overviewTab: tab,
      });
      navigate(`${path}${location.search}`, { replace: true });
    },
    [experimentId, navigate, location.search],
  );

  return [activeTab, setActiveTab] as const;
};
