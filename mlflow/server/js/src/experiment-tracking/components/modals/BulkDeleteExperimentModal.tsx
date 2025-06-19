import { Typography } from '@databricks/design-system';
import { ExperimentEntity } from '../../types';
import { ConfirmModal } from './ConfirmModal';
import { deleteExperimentApi } from '../../actions';
import { useDispatch } from 'react-redux';
import { ThunkDispatch } from '@mlflow/mlflow/src/redux-types';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';

type Props = {
  isOpen: boolean;
  onClose: () => void;
  experiments: Pick<ExperimentEntity, 'experimentId' | 'name'>[];
  onExperimentsDeleted: () => void;
};

export const BulkDeleteExperimentModal = ({ isOpen, onClose, experiments, onExperimentsDeleted }: Props) => {
  const dispatch = useDispatch<ThunkDispatch>();

  const handleSubmit = () => {
    return Promise.all(experiments.map((experiment) => dispatch(deleteExperimentApi(experiment.experimentId))))
      .then(onExperimentsDeleted)
      .catch((e: any) => Utils.logErrorAndNotifyUser(e));
  };

  return (
    <ConfirmModal
      isOpen={isOpen}
      onClose={onClose}
      handleSubmit={handleSubmit}
      title={`Delete ${experiments.length} Experiment(s)`}
      helpText={
        <div>
          <Typography.Paragraph>The following experiments will be deleted:</Typography.Paragraph>
          <Typography.Paragraph>
            <ul>
              {experiments.map((experiment) => (
                <li key={experiment.experimentId}>
                  <Typography.Text>
                    {experiment.name} (ID: {experiment.experimentId})
                  </Typography.Text>
                </li>
              ))}
            </ul>
          </Typography.Paragraph>
        </div>
      }
      confirmButtonText="Delete"
    />
  );
};
