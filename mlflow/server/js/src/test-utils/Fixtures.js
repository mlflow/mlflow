import { Experiment } from '../sdk/MlflowMessages';

const createExperiment = ({
  experiment_id = '0',
  name = 'Default',
  lifecycle_stage = 'active' } = {}
) => (
  Experiment.fromJs({ experiment_id, name, lifecycle_stage })
);
export default {
  createExperiment,
  experiments: [
    createExperiment(),
    createExperiment({ experiment_id: '1', name: 'Test'}),
  ],
};
