import Fixtures from '../utils/test-utils/Fixtures';
import { getFirstActiveExperiment } from './HomeView';

const experiments = [
  Fixtures.createExperiment({ experiment_id: '1', name: '1', lifecycle_stage: 'deleted' }),
  Fixtures.createExperiment({ experiment_id: '3', name: '3', lifecycle_stage: 'active' }),
  Fixtures.createExperiment({ experiment_id: '2', name: '2', lifecycle_stage: 'active' }),
];

test('getFirstActiveExperiment works', () => {
  expect(getFirstActiveExperiment(experiments).experiment_id).toEqual('2');
});
