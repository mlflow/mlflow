import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { Provider } from 'react-redux';
import ExperimentListView from './ExperimentListView';
import Fixtures from '../utils/test-utils/Fixtures';
import { DeleteExperimentModal } from './modals/DeleteExperimentModal';
import { RenameExperimentModal } from './modals/RenameExperimentModal';
import { CreateExperimentModal } from './modals/CreateExperimentModal';
import {
  emptyState,
  storeWithExperiments,
  makeStateWithExperiments,
} from '../utils/test-utils/ReduxStoreFixtures';
import { mountWithIntl } from '../../common/utils/TestUtils';
import { makeStore } from '../../store';

// Mount with a real store to test reducer interaction with experiments.
const mountComponent = (props, store = storeWithExperiments) => {
  return mountWithIntl(
    <Provider store={store}>
      <BrowserRouter>
        <ExperimentListView {...props} history={{}} />,
      </BrowserRouter>
      ,
    </Provider>,
  );
};

test('Should be two list items if there are two experiments', () => {
  const wrapper = mountComponent({
    activeExperimentIds: Fixtures.experiments.map((e) => e.experiment_id),
  });
  expect(wrapper.find('[data-test-id="active-experiment-list-item"]')).toHaveLength(2);
});

test('If searchInput is set to "Test" then first shown element in experiment list has the title "Test"', () => {
  const wrapper = mountComponent({ activeExperimentIds: ['0'] });
  wrapper
    .find('input[data-test-id="search-experiment-input"]')
    .first()
    .simulate('change', { target: { value: 'Test' } });
  expect(wrapper.find('[data-test-id="experiment-list-item"]').first().text()).toContain('Test');
});

test('If searchInput is set to "Test" and default experiment is active then no active element is shown in the experiment list', () => {
  const wrapper = mountComponent({ activeExperimentIds: ['0'] });
  wrapper
    .find('input[data-test-id="search-experiment-input"]')
    .first()
    .simulate('change', { target: { value: 'Test' } });
  expect(wrapper.find('[data-test-id="active-experiment-list-item"]')).toHaveLength(0);
});

test('If button to create experiment is pressed then open CreateExperimentModal', () => {
  const wrapper = mountComponent({ activeExperimentIds: ['0'] });
  wrapper.find('[data-test-id="create-experiment-button"]').first().simulate('click');
  expect(wrapper.find(CreateExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to delete experiment is pressed then open DeleteExperimentModal', () => {
  const wrapper = mountComponent({ activeExperimentIds: ['0'] });
  wrapper.find('button[data-test-id="delete-experiment-button"]').first().simulate('click');
  expect(wrapper.find(DeleteExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to edit experiment is pressed then open RenameExperimentModal', () => {
  const wrapper = mountComponent({ activeExperimentIds: ['0'] });
  wrapper.find('button[data-test-id="rename-experiment-button"]').first().simulate('click');
  expect(wrapper.find(RenameExperimentModal).prop('isOpen')).toEqual(true);
});

test('If activeExperimentIds is defined then choose all the corresponding experiments', () => {
  const localExperiments = [
    Fixtures.createExperiment(),
    Fixtures.createExperiment({ experiment_id: '1', name: 'Test' }),
    Fixtures.createExperiment({ experiment_id: '2', name: 'Second' }),
    Fixtures.createExperiment({ experiment_id: '3', name: 'Third' }),
  ];

  const state = makeStateWithExperiments(localExperiments);
  const wrapper = mountComponent({ activeExperimentIds: ['1', '3'] }, makeStore(state));
  const selected = wrapper.find('[data-test-id="active-experiment-list-item"]');
  expect(selected.length).toEqual(2);
  expect(selected.first().text()).toEqual('Test');
  expect(selected.at(1).text()).toEqual('Third');
});

test('should render when both experiments and activeExperimentIds are empty', () => {
  const wrapper = mountComponent({ activeExperimentIds: [] }, makeStore(emptyState));
  expect(wrapper.length).toBe(1);
});

test('UI should render in a reasonable time with a large number of experiments.', () => {
  // not intended as a benchmark, the point is the ui will render without
  // crashing nor taking an exceedingly long time.
  const ids = [...Array(1000).keys()].map((k) => k.toString());
  const localExperiments = ids.map((k) => Fixtures.createExperiment({ experiment_id: k, name: k }));

  const state = makeStateWithExperiments(localExperiments);
  const store = makeStore(state);
  const start = process.hrtime.bigint();

  const wrapper = mountComponent({ activeExperimentIds: ids }, store);

  const diff = Number(process.hrtime.bigint() - start) / 1000000000;
  expect(diff).toBeLessThanOrEqual(30);

  // It should not attempt to render all the items, but should render some.
  const items = wrapper.find('[data-test-id="active-experiment-list-item"]');
  expect(items.length).toBeLessThanOrEqual(1000);
  expect(items.length).toBeGreaterThan(1);
});
