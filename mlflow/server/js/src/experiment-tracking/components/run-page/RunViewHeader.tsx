import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { OverflowMenu, PageHeader } from '../../../shared/building_blocks/PageHeader';
import Routes from '../../routes';
import { ExperimentEntity } from '../../types';
import { RenameRunModal } from '../modals/RenameRunModal';
import { RunViewModeSwitch } from './RunViewModeSwitch';
import { shouldEnableDeepLearningUI } from '../../../common/utils/FeatureUtils';

/**
 * Run details page header component, common for all page view modes
 */
export const RunViewHeader = ({
  hasComparedExperimentsBefore,
  comparedExperimentIds = [],
  experiment,
  runDisplayName,
  runUuid,
  handleRenameRunClick,
}: {
  hasComparedExperimentsBefore?: boolean;
  comparedExperimentIds?: string[];
  runUuid: string;
  runDisplayName: string;
  experiment: ExperimentEntity;
  handleRenameRunClick: () => void;
}) => {
  function getExperimentPageLink() {
    return hasComparedExperimentsBefore && comparedExperimentIds ? (
      <Link to={Routes.getCompareExperimentsPageRoute(comparedExperimentIds)}>
        <FormattedMessage
          defaultMessage='Displaying Runs from {numExperiments} Experiments'
          // eslint-disable-next-line max-len
          description='Breadcrumb nav item to link to the compare-experiments page on compare runs page'
          values={{
            numExperiments: comparedExperimentIds.length,
          }}
        />
      </Link>
    ) : (
      <Link
        to={Routes.getExperimentPageRoute(experiment.experiment_id)}
        data-test-id='experiment-runs-link'
      >
        {experiment.getName()}
      </Link>
    );
  }

  const breadcrumbs = [getExperimentPageLink()];

  return (
    <div css={{ flexShrink: 0 }}>
      <PageHeader
        title={<span data-test-id='runs-header'>{runDisplayName}</span>}
        breadcrumbs={breadcrumbs}
      >
        <OverflowMenu
          menu={[
            {
              id: 'overflow-rename-button',
              onClick: handleRenameRunClick,
              itemName: (
                <FormattedMessage
                  defaultMessage='Rename'
                  description='Menu item to rename an experiment run'
                />
              ),
            },
          ]}
        />
      </PageHeader>
      {shouldEnableDeepLearningUI() && <RunViewModeSwitch />}
    </div>
  );
};
