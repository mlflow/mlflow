import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { OverflowMenu, PageHeader } from '../../../shared/building_blocks/PageHeader';
import Routes from '../../routes';
import { ExperimentEntity, KeyValueEntity } from '../../types';
import { RunViewModeSwitch } from './RunViewModeSwitch';
import Utils from '../../../common/utils/Utils';
import { RunViewHeaderRegisterModelButton } from './RunViewHeaderRegisterModelButton';
import type { UseGetRunQueryResponseExperiment } from './hooks/useGetRunQuery';

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
      <Link to={Routes.getExperimentPageRoute(experiment?.experimentId ?? '')} data-test-id="experiment-runs-link">
        {experiment.name}
      </Link>
    );
  }

  const breadcrumbs = [getExperimentPageLink()];

  return (
    <div css={{ flexShrink: 0 }}>
      <PageHeader
        title={<span data-test-id="runs-header">{runDisplayName}</span>}
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

        <RunViewHeaderRegisterModelButton
          runUuid={runUuid}
          experimentId={experiment?.experimentId ?? ''}
          runTags={runTags}
          artifactRootUri={artifactRootUri}
        />
      </PageHeader>
      <RunViewModeSwitch />
    </div>
  );
};
