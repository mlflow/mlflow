import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentListView } from './ExperimentListView';
import Fixtures from '../test-utils/Fixtures';

test('If activeExperimentId is defined then choose that one', () => {
  const wrapper = shallow(<ExperimentListView
    onClickListExperiments={() => {}}
    experiments={Fixtures.experiments}
    activeExperimentId={1}
  />);
  expect(wrapper.find('.active-experiment-list-item').first().prop('title')).toEqual('Test');
});

test('If activeExperimentId is undefined then choose first experiment', () => {
  const wrapper = shallow(<ExperimentListView
    onClickListExperiments={() => {}}
    experiments={Fixtures.experiments}
  />);
  expect(wrapper.find('.active-experiment-list-item').first().prop('title')).toEqual('Default');
});

test('If searchInput is set to "Test" then first shown element in experiment list has the title "Test"', () => {
  const wrapper = shallow(<ExperimentListView
    onClickListExperiments={() => {}}
    experiments={Fixtures.experiments}
  />);

  wrapper.setState({ searchInput: 'Test' });
  expect(wrapper.find('.experiment-list-item').first().prop('title')).toEqual('Test');
});

test('If searchInput is set to "Test" and default experiment is active then no active element is shown in the experiment list', () => {
  const wrapper = shallow(<ExperimentListView
    onClickListExperiments={() => {}}
    experiments={Fixtures.experiments}
    activeExperimentId={0}
  />);

  wrapper.setState({ searchInput: 'Test' });
  expect(wrapper.find('.active-experiment-list-item')).toHaveLength(0);
});
