import React from 'react';
import {
  SegmentedControlGroup,
  SegmentedControlButton,
  useDesignSystemTheme,
  Tooltip,
} from '@databricks/design-system';

import { ExperimentKind, ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { useExperimentPageViewMode } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/hooks/useExperimentPageViewMode';
import { useExperimentEvaluationRunsData } from '@mlflow/mlflow/src/experiment-tracking/components/experiment-page/hooks/useExperimentEvaluationRunsData';
import { Link, useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { coerceToEnum } from '@databricks/web-shared/utils';
import { shouldEnablePromptsTabOnDBPlatform } from '../../../../../../common/utils/FeatureUtils';
import type { TabConfigMap } from './TabSelectorBarConstants';
import {
  getGenAIExperimentTabConfigMap,
  getGenAIExperimentWithPromptsTabConfigMap,
  CustomExperimentTabConfigMap,
  DefaultTabConfigMap,
} from './TabSelectorBarConstants';
import { FormattedMessage } from 'react-intl';
import { useGetExperimentPageActiveTabByRoute } from '../../../hooks/useGetExperimentPageActiveTabByRoute';

const isRunsViewTab = (tabName: string) => ['TABLE', 'CHART', 'ARTIFACT'].includes(tabName);
const iTracesViewTab = (tabName: string) => ['TRACES'].includes(tabName);

const getExperimentTabsConfig = (experimentKind?: ExperimentKind, hasTrainingRuns = false): TabConfigMap => {
  switch (experimentKind) {
    case ExperimentKind.GENAI_DEVELOPMENT:
    case ExperimentKind.GENAI_DEVELOPMENT_INFERRED:
      return shouldEnablePromptsTabOnDBPlatform()
        ? getGenAIExperimentWithPromptsTabConfigMap({ includeRunsTab: hasTrainingRuns })
        : getGenAIExperimentTabConfigMap({ includeRunsTab: hasTrainingRuns });
    case ExperimentKind.CUSTOM_MODEL_DEVELOPMENT:
    case ExperimentKind.CUSTOM_MODEL_DEVELOPMENT_INFERRED:
    case ExperimentKind.FORECASTING:
    case ExperimentKind.REGRESSION:
    case ExperimentKind.AUTOML:
    case ExperimentKind.CLASSIFICATION:
      return CustomExperimentTabConfigMap;
    default:
      return DefaultTabConfigMap;
  }
};

export const TabSelectorBar = ({ experimentKind }: { experimentKind?: ExperimentKind }) => {
  const { experimentId, tabName } = useParams();
  const { theme } = useDesignSystemTheme();
  const [viewMode] = useExperimentPageViewMode();

  const isGenAIExperiment =
    experimentKind === ExperimentKind.GENAI_DEVELOPMENT || experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED;

  const { trainingRuns } = useExperimentEvaluationRunsData({
    experimentId: experimentId || '',
    enabled: isGenAIExperiment,
    filter: '', // not important in this case, we show the runs tab if there are any training runs
  });

  // In the tab selector bar, we're interested in top-level tab names based on the current route
  const { topLevelTabName: tabNameFromRoute } = useGetExperimentPageActiveTabByRoute();

  let tabNameFromParams = coerceToEnum(ExperimentPageTabName, tabName, undefined);
  if (tabNameFromParams === ExperimentPageTabName.Datasets) {
    // datasets is a sub-tab of evaluation runs, so we
    // should show the evaluation runs tab as active
    tabNameFromParams = ExperimentPageTabName.EvaluationRuns;
  }
  if (tabNameFromParams === ExperimentPageTabName.LabelingSchemas) {
    // labeling schemas is a sub-tab of labeling sessions, so we
    // should show the labeling sessions tab as active
    tabNameFromParams = ExperimentPageTabName.LabelingSessions;
  }

  const tabNameFromViewMode = (() => {
    if (isRunsViewTab(viewMode)) {
      return ExperimentPageTabName.Runs;
    } else if (iTracesViewTab(viewMode)) {
      return ExperimentPageTabName.Traces;
    } else {
      return viewMode;
    }
  })();

  const activeTab = tabNameFromRoute ?? tabNameFromParams ?? tabNameFromViewMode;

  const hasTrainingRuns = trainingRuns?.length > 0;
  const tabsConfig = getExperimentTabsConfig(
    experimentKind ?? ExperimentKind.NO_INFERRED_TYPE,
    isGenAIExperiment && hasTrainingRuns,
  );

  return (
    <SegmentedControlGroup
      value={activeTab}
      name="tab-toggle-bar"
      componentId="mlflow.experiment-tracking.tab-toggle-bar"
      newStyleFlagOverride
      css={{
        justifySelf: 'center',
        [theme.responsive.mediaQueries.xl]: {
          '& .tab-icon-text': {
            display: 'inline-flex',
          },
          '& .tab-icon-with-tooltip': {
            display: 'none',
          },
        },
      }}
    >
      {Object.entries(tabsConfig).map(([tabName, tabConfig]) => {
        const isActive = tabName === activeTab;

        return (
          <React.Fragment key={tabName}>
            <Link
              css={{ display: 'none' }}
              className="tab-icon-text"
              key={`${tabName}-text`}
              to={tabConfig.getRoute(experimentId ?? '')}
            >
              <SegmentedControlButton
                data-testid={`tab-selector-button-text-${tabName}-${isActive ? 'active' : 'inactive'}`}
                className="tab-icon-text"
                value={tabName}
                icon={tabConfig.icon}
              >
                <span>{tabConfig.label}</span>
              </SegmentedControlButton>
            </Link>
            <Link
              className="tab-icon-with-tooltip"
              key={`${tabName}-tooltip`}
              to={tabConfig.getRoute(experimentId ?? '')}
            >
              <SegmentedControlButton
                data-testid={`tab-selector-button-icon-${tabName}-${isActive ? 'active' : 'inactive'}`}
                className="tab-icon-with-tooltip"
                value={tabName}
                icon={
                  <Tooltip
                    delayDuration={0}
                    content={
                      <span>
                        {/* comment for formatting */}
                        {tabConfig.label}
                      </span>
                    }
                    componentId={`mlflow.experiment-tracking.tab-selector-bar.${tabName}`}
                  >
                    <span>{tabConfig.icon}</span>
                  </Tooltip>
                }
              />
            </Link>
          </React.Fragment>
        );
      })}
    </SegmentedControlGroup>
  );
};
