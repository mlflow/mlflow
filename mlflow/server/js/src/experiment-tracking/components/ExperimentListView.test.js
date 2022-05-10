import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import { ExperimentListView } from './ExperimentListView';
import Fixtures from '../utils/test-utils/Fixtures';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import { mountWithIntl } from '../../common/utils/TestUtils';

const mountComponent = (props) => {
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  return mountWithIntl(
    <Provider
      store={mockStore({
        entities: {
          experimentsById: {},
        },
      })}
    >
      <BrowserRouter>
        <ExperimentListView {...props} />,
      </BrowserRouter>
      ,
    </Provider>,
  );
};

test('If activeExperimentId is defined then choose that one', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments, activeExperimentId: '1' });
  expect(
    wrapper
      .find('[data-test-id="active-experiment-list-item"]')
      .first()
      .text(),
  ).toContain('Test');
});

test('If activeExperimentId is undefined then choose first experiment', () => {
  const wrapper = mountComponent({
    experiments: Fixtures.experiments,
  });
  expect(
    wrapper
      .find('[data-test-id="active-experiment-list-item"]')
      .first()
      .text(),
  ).toContain('Default');
});

test('If searchInput is set to "Test" then first shown element in experiment list has the title "Test"', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments });
  wrapper
    .find('input[data-test-id="search-experiment-input"]')
    .first()
    .simulate('change', { target: { value: 'Test' } });
  expect(
    wrapper
      .find('[data-test-id="experiment-list-item"]')
      .first()
      .text(),
  ).toContain('Test');
});

test('If searchInput is set to "Test" and default experiment is active then no active element is shown in the experiment list', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments, activeExperimentId: '0' });
  wrapper
    .find('input[data-test-id="search-experiment-input"]')
    .first()
    .simulate('change', { target: { value: 'Test' } });
  expect(wrapper.find('[data-test-id="active-experiment-list-item"]')).toHaveLength(0);
});

test('If button to create experiment is pressed then open CreateExperimentModal', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments });
  wrapper
    .find('[data-test-id="create-experiment-button"]')
    .first()
    .simulate('click');
  expect(wrapper.find(CreateExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to delete experiment is pressed then open DeleteExperimentModal', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments });
  wrapper
    .find('[data-test-id="delete-experiment-button"]')
    .first()
    .simulate('click');
  expect(wrapper.find(DeleteExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to edit experiment is pressed then open RenameExperimentModal', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments });
  wrapper
    .find('[data-test-id="rename-experiment-button"]')
    .first()
    .simulate('click');
  expect(wrapper.find(RenameExperimentModal).prop('isOpen')).toEqual(true);
});
