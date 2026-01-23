import { TitleSkeleton, useDesignSystemTheme } from '@databricks/design-system';
import { ExperimentPageTabName } from '../../../constants';
import { ExperimentKind } from '../../../constants';
import { useExperimentEvaluationRunsData } from '../../../components/experiment-page/hooks/useExperimentEvaluationRunsData';
import type { ExperimentPageSideNavSectionKey } from './constants';
import { COLLAPSED_CLASS_NAME, FULL_WIDTH_CLASS_NAME, useExperimentPageSideNavConfig } from './constants';
import { ExperimentPageSideNavSection } from './ExperimentPageSideNavSection';
import { ExperimentPageSideNavAssistantButton } from './ExperimentPageSideNavAssistantButton';
import { useParams } from '@mlflow/mlflow/src/common/utils/RoutingUtils';

const SIDE_NAV_WIDTH = 160;
const SIDE_NAV_COLLAPSED_WIDTH = 32;

export const ExperimentPageSideNav = ({
  experimentKind,
  activeTab,
}: {
  experimentKind: ExperimentKind;
  activeTab: ExperimentPageTabName;
}) => {
  const { theme } = useDesignSystemTheme();
  const { experimentId } = useParams();
  // the single chat session tab also has a sidebar. to conserve
  // horizontal space, we force the side nav to be collapsed in this tab
  const forceCollapsed = activeTab === ExperimentPageTabName.SingleChatSession;

  const isGenAIExperiment =
    experimentKind === ExperimentKind.GENAI_DEVELOPMENT || experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED;

  const { trainingRuns } = useExperimentEvaluationRunsData({
    experimentId: experimentId || '',
    enabled: isGenAIExperiment,
    filter: '', // not important in this case, we show the runs tab if there are any training runs
  });

  const hasTrainingRuns = trainingRuns?.length > 0;

  const sideNavConfig = useExperimentPageSideNavConfig({
    experimentKind,
    hasTrainingRuns,
  });

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        paddingTop: theme.spacing.sm,
        paddingRight: theme.spacing.sm,
        borderRight: `1px solid ${theme.colors.border}`,
        boxSizing: 'content-box',
        width: SIDE_NAV_COLLAPSED_WIDTH,
        [`& .${COLLAPSED_CLASS_NAME}`]: {
          display: 'flex',
        },
        [`& .${FULL_WIDTH_CLASS_NAME}`]: {
          display: 'none',
        },
        ...(!forceCollapsed
          ? {
              [theme.responsive.mediaQueries.xl]: {
                width: SIDE_NAV_WIDTH,
                [`& .${COLLAPSED_CLASS_NAME}`]: {
                  display: 'none',
                },
                [`& .${FULL_WIDTH_CLASS_NAME}`]: {
                  display: 'flex',
                },
              },
            }
          : {}),
      }}
    >
      <div>
        <ExperimentPageSideNavAssistantButton />
        {Object.entries(sideNavConfig).map(([sectionKey, items]) => (
          <ExperimentPageSideNavSection
            key={sectionKey}
            activeTab={activeTab}
            sectionKey={sectionKey as ExperimentPageSideNavSectionKey}
            items={items}
          />
        ))}
      </div>
    </div>
  );
};

export const ExperimentPageSideNavSkeleton = () => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        paddingTop: theme.spacing.sm,
        paddingRight: theme.spacing.sm,
        borderRight: `1px solid ${theme.colors.border}`,
        width: SIDE_NAV_COLLAPSED_WIDTH,
        [theme.responsive.mediaQueries.xl]: {
          width: SIDE_NAV_WIDTH,
        },
      }}
    >
      <TitleSkeleton css={{ width: '60%' }} />
      <TitleSkeleton css={{ width: '80%' }} />
      <TitleSkeleton css={{ width: '70%' }} />
    </div>
  );
};
