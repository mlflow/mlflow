import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { OverflowMenu, PageHeader } from '../../../shared/building_blocks/PageHeader';
import Routes, { PageId as ExperimentTrackingPageId } from '../../routes';
import type { ExperimentEntity } from '../../types';
import { KeyValueEntity } from '../../../common/types';
import { RunViewModeSwitch } from './RunViewModeSwitch';
import Utils from '../../../common/utils/Utils';
import { RunViewHeaderRegisterModelButton } from './RunViewHeaderRegisterModelButton';
import type { UseGetRunQueryResponseExperiment } from './hooks/useGetRunQuery';
import type { RunPageModelVersionSummary } from './hooks/useUnifiedRegisteredModelVersionsSummariesForRun';
import { ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { shouldEnableExperimentPageHeaderV2 } from '../../../common/utils/FeatureUtils';

/**
 * Run details page header component, common for all page view modes
 */
export const RunViewHeader = ({
  hasComparedExperimentsBefore,
  comparedExperimentIds = [],
  experiment,
  runDisplayName,
  runTags,
  runParams,
  runUuid,
  handleRenameRunClick,
  handleDeleteRunClick,
  artifactRootUri,
  registeredModelVersionSummaries,
  isLoading,
}: {
  hasComparedExperimentsBefore?: boolean;
  comparedExperimentIds?: string[];
  runDisplayName: string;
  runUuid: string;
  runTags: Record<string, KeyValueEntity>;
  runParams: Record<string, KeyValueEntity>;
  experiment: ExperimentEntity | UseGetRunQueryResponseExperiment;
  handleRenameRunClick: () => void;
  handleDeleteRunClick?: () => void;
  artifactRootUri?: string;
  registeredModelVersionSummaries: RunPageModelVersionSummary[];
  isLoading?: boolean;
}) => {
  function getExperimentPageLink() {
    return hasComparedExperimentsBefore && comparedExperimentIds ? (
      <Link to={Routes.getCompareExperimentsPageRoute(comparedExperimentIds)}>
        <FormattedMessage
          defaultMessage="Displaying Runs from {numExperiments} Experiments"
          // eslint-disable-next-line max-len
          description="Breadcrumb nav item to link to the compare-experiments page on compare runs page"
          values={{
            numExperiments: comparedExperimentIds.length,
          }}
        />
      </Link>
    ) : (
      <Link to={Routes.getExperimentPageRoute(experiment?.experimentId ?? '')} data-testid="experiment-runs-link">
        {experiment.name}
      </Link>
    );
  }

  const breadcrumbs = [getExperimentPageLink()];
  if (shouldEnableExperimentPageHeaderV2() && experiment.experimentId) {
    breadcrumbs.push(
      <Link
        to={Routes.getExperimentPageTabRoute(experiment.experimentId, ExperimentPageTabName.Runs)}
        data-testid="experiment-observatory-link-runs"
      >
        <FormattedMessage
          defaultMessage="Runs"
          description="Breadcrumb nav item to link to the list of runs on the parent experiment"
        />
      </Link>,
    );
  }

  const renderRegisterModelButton = () => {
    return (
      <RunViewHeaderRegisterModelButton
        runUuid={runUuid}
        experimentId={experiment?.experimentId ?? ''}
        runTags={runTags}
        artifactRootUri={artifactRootUri}
        registeredModelVersionSummaries={registeredModelVersionSummaries}
      />
    );
  };

  return (
    <div css={{ flexShrink: 0 }}>
      <PageHeader
        title={<span data-testid="runs-header">{runDisplayName}</span>}
        breadcrumbs={breadcrumbs}
        /* prettier-ignore */
      >
        <OverflowMenu
          menu={[
            {
              id: 'overflow-rename-button',
              onClick: handleRenameRunClick,
              itemName: (
                <FormattedMessage defaultMessage="Rename" description="Menu item to rename an experiment run" />
              ),
            },
            ...(handleDeleteRunClick
              ? [
                  {
                    id: 'overflow-delete-button',
                    onClick: handleDeleteRunClick,
                    itemName: (
                      <FormattedMessage defaultMessage="Delete" description="Menu item to delete an experiment run" />
                    ),
                  },
                ]
              : []),
          ]}
        />

        {renderRegisterModelButton()}
      </PageHeader>
      <RunViewModeSwitch />
    </div>
  );
};
