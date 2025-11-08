import { useCallback, useMemo, useState } from 'react';
import { Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { OverflowMenu } from '../../../../../shared/building_blocks/PageHeader';
import type { ExperimentEntity } from '../../../../types';
import { getExperimentType } from '../../utils/experimentPage.common-utils';
import { getShareFeedbackOverflowMenuItem } from './ExperimentViewHeader.utils';
import { getExperimentKindFromTags } from '../../../../utils/ExperimentKindUtils';
import { ExperimentKind } from '../../../../constants';
import { useNavigate } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import Routes from '@mlflow/mlflow/src/experiment-tracking/routes';
import { DeleteExperimentModal } from '../../../modals/DeleteExperimentModal';
import { RenameExperimentModal } from '../../../modals/RenameExperimentModal';
import { useInvalidateExperimentList } from '../../hooks/useExperimentListQuery';

/**
 * Experiment page header part responsible for displaying menu
 * with rename and delete buttons
 */
export const ExperimentViewManagementMenu = ({
  experiment,
  setEditing,
  baseComponentId = 'mlflow.experiment_page.managementMenu',
  refetchExperiment,
}: {
  experiment: ExperimentEntity;
  setEditing?: (editing: boolean) => void;
  baseComponentId?: string;
  refetchExperiment?: () => Promise<unknown>;
}) => {
  const [showRenameExperimentModal, setShowRenameExperimentModal] = useState(false);
  const [showDeleteExperimentModal, setShowDeleteExperimentModal] = useState(false);
  const invalidateExperimentList = useInvalidateExperimentList();
  const navigate = useNavigate();

  return (
    <>
      <OverflowMenu
        menu={[
          ...(setEditing
            ? [
                {
                  id: 'edit-description',
                  itemName: (
                    <Typography.Text>
                      <FormattedMessage
                        defaultMessage="Edit description"
                        description="Text for edit description button on experiment view page header"
                      />
                    </Typography.Text>
                  ),
                  onClick: () => setEditing(true),
                },
              ]
            : []),
          {
            id: 'rename',
            itemName: (
              <FormattedMessage
                defaultMessage="Rename"
                description="Text for rename button on the experiment view page header"
              />
            ),
            onClick: () => setShowRenameExperimentModal(true),
          },
          {
            id: 'delete',
            itemName: (
              <FormattedMessage
                defaultMessage="Delete"
                description="Text for delete button on the experiment view page header"
              />
            ),
            onClick: () => setShowDeleteExperimentModal(true),
          },
        ]}
      />
      <RenameExperimentModal
        experimentId={experiment.experimentId}
        experimentName={experiment.name}
        isOpen={showRenameExperimentModal}
        onClose={() => setShowRenameExperimentModal(false)}
        onExperimentRenamed={invalidateExperimentList}
      />
      <DeleteExperimentModal
        experimentId={experiment.experimentId}
        experimentName={experiment.name}
        isOpen={showDeleteExperimentModal}
        onClose={() => setShowDeleteExperimentModal(false)}
        onExperimentDeleted={() => {
          invalidateExperimentList();
          navigate(Routes.experimentsObservatoryRoute);
        }}
      />
    </>
  );
};
