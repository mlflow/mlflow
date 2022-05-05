import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentListView } from './ExperimentListView';
import Fixtures from '../utils/test-utils/Fixtures';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { CreateExperimentModal } from './modals/CreateExperimentModal';

test('If searchInput is set to "Test" then first shown element in experiment list has the title "Test"', () => {
  const wrapper = shallow(
    <ExperimentListView onClickListExperiments={() => {}} experiments={Fixtures.experiments} />,
  );

  wrapper.setState({ searchInput: 'Test' });
  expect(
    wrapper
      .find('.experiment-list-item')
      .first()
      .prop('title'),
  ).toEqual('Test');
});

test('If searchInput is set to "Test" and default experiment is active then no active element is shown in the experiment list', () => {
  const wrapper = shallow(
    <ExperimentListView
      onClickListExperiments={() => {}}
      experiments={Fixtures.experiments}
      activeExperimentIds={['0']}
    />,
  );

  wrapper.setState({ searchInput: 'Test' });
  expect(wrapper.find('.active-experiment-list-item')).toHaveLength(0);
});

test('If button to create experiment is pressed then open CreateExperimentModal', () => {
  const wrapper = shallow(
    <ExperimentListView onClickListExperiments={() => {}} experiments={Fixtures.experiments} />,
  );
  // find create experiment link
  const createExpLink = wrapper.find('.experiment-list-create-btn');
  // mock event that is passed when clicking the link
  createExpLink.simulate('click');

  expect(wrapper.find(CreateExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to delete experiment is pressed then open DeleteExperimentModal', () => {
  const wrapper = shallow(
    <ExperimentListView onClickListExperiments={() => {}} experiments={Fixtures.experiments} />,
  );
  // find delete experiment link
  const deleteLink = wrapper
    .find('.active-experiment-list-item')
    .children()
    .at(3);
  // mock event that is passed when clicking the link
  const mockedExperiment = Fixtures.experiments[0];
  const mockedEvent = {
    currentTarget: {
      dataset: {
        experimentid: mockedExperiment.experiment_id,
        experimentname: mockedExperiment.name,
      },
    },
  };
  deleteLink.simulate('click', mockedEvent);

  expect(wrapper.find(DeleteExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to edit experiment is pressed then open RenameExperimentModal', () => {
  const wrapper = shallow(
    <ExperimentListView onClickListExperiments={() => {}} experiments={Fixtures.experiments} />,
  );

  // find edit experiment link
  const editLink = wrapper
    .find('.active-experiment-list-item')
    .children()
    .at(2);
  // mock event that is passed when clicking the link
  const mockedExperiment = Fixtures.experiments[0];
  const mockedEvent = {
    currentTarget: {
      dataset: {
        experimentid: mockedExperiment.experiment_id,
        experimentname: mockedExperiment.name,
      },
    },
  };
  editLink.simulate('click', mockedEvent);

  expect(wrapper.find(RenameExperimentModal).prop('isOpen')).toEqual(true);
});

test('If activeExperimentIds is defined then choose all the corresponding experiments', () => {
  const experiments = [
    Fixtures.createExperiment(),
    Fixtures.createExperiment({ experiment_id: '1', name: 'Test' }),
    Fixtures.createExperiment({ experiment_id: '2', name: 'Second' }),
    Fixtures.createExperiment({ experiment_id: '3', name: 'Third' })
  ];
  const wrapper = shallow(
    <ExperimentListView
      onClickListExperiments={() => {}}
      experiments={experiments}
      activeExperimentIds={['1', '3']}
    />,
  );
  expect(wrapper.find('.active-experiment-list-item').length).toEqual(2);
  expect(wrapper.find('.active-experiment-list-item').first().prop('title')).toEqual('Test');
  expect(wrapper.find('.active-experiment-list-item').at(1).prop('title')).toEqual('Third');
});

test('If activeExperimentIds is undefined then choose the first experiment', () => {
  const experiments = [
    Fixtures.createExperiment(),
    Fixtures.createExperiment({ experiment_id: '1', name: 'Test' }),
    Fixtures.createExperiment({ experiment_id: '2', name: 'Second' }),
    Fixtures.createExperiment({ experiment_id: '3', name: 'Third' })
  ];
  const wrapper = shallow(
    <ExperimentListView onClickListExperiments={() => {}} experiments={experiments} />,
  );
  expect(
    wrapper
      .find('.active-experiment-list-item')
      .first()
      .prop('title'),
  ).toEqual('Default');
});