import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentPagePersistedState } from '../sdk/MlflowLocalStorageMessages';
import Fixtures from '../test-utils/Fixtures';

test('Local storage messages ignore unknown fields', () => {
  const persistedState = ExperimentPagePersistedState({heyYallImAnUnknownField: "value"})
});

test('If activeExperimentId is undefined then choose first experiment', () => {
  const wrapper = shallow(<ExperimentListView
    onClickListExperiments={() => {}}
    experiments={Fixtures.experiments}
  />);
  expect(wrapper.find('.active-experiment-list-item').first().prop('title')).toEqual('Default');
});
