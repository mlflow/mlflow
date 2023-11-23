/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
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

// Make the autosizer render items.
// https://github.com/bvaughn/react-virtualized/blob/v9.22.3/source/AutoSizer/AutoSizer.jest.js#L68
function mockOffsetSize(width: any, height: any) {
  Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
    configurable: true,
    value: height,
  });
  Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
    configurable: true,
    value: width,
  });
}

beforeAll(() => {
  mockOffsetSize(200, 1000);
});

afterAll(() => {
  jest.restoreAllMocks();
});

// Need to mock this since the hoc doesn't pick up theme
const designSystemThemeApi = {
  theme: {
    colors: { primary: 'solid', actionDefaultBackgroundPress: `solid` },
  },
};

const mountComponent = (props: any) => {
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
        <ExperimentListView {...props} history={[]} designSystemThemeApi={designSystemThemeApi} />,
      </BrowserRouter>
      ,
    </Provider>,
  );
};

test('If searchInput is set to "Test" then first shown element in experiment list has the title "Test"', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  wrapper
    .find('input[data-test-id="search-experiment-input"]')
    .first()
    .simulate('change', { target: { value: 'Test' } });
  expect(wrapper.find('[data-test-id="experiment-list-item"]').first().text()).toContain('Test');
});

test('If searchInput is set to "Test" and default experiment is active then no active element is shown in the experiment list', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  wrapper
    .find('input[data-test-id="search-experiment-input"]')
    .first()
    .simulate('change', { target: { value: 'Test' } });
  expect(wrapper.find('[data-test-id="active-experiment-list-item"]')).toHaveLength(0);
});

test('If button to create experiment is pressed then open CreateExperimentModal', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  wrapper.find('[data-test-id="create-experiment-button"]').first().simulate('click');
  expect(wrapper.find(CreateExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to delete experiment is pressed then open DeleteExperimentModal', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  wrapper.find('button[data-test-id="delete-experiment-button"]').first().simulate('click');
  expect(wrapper.find(DeleteExperimentModal).prop('isOpen')).toEqual(true);
});

test('If button to edit experiment is pressed then open RenameExperimentModal', () => {
  const wrapper = mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
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
  const wrapper = mountComponent({
    experiments: localExperiments,
    activeExperimentIds: ['1', '3'],
  });
  const selected = wrapper.find('[data-test-id="active-experiment-list-item"]');
  expect(selected.length).toEqual(2);
  expect(selected.first().text()).toEqual('Test');
  expect(selected.at(1).text()).toEqual('Third');
});

test('should render when both experiments and activeExperimentIds are empty', () => {
  const wrapper = mountComponent({
    experiments: [],
    activeExperimentIds: [],
  });
  expect(wrapper.length).toBe(1);
});

test('virtual list should not render everything when there are many experiments', () => {
  const keys = Array.from(Array(1000).keys()).map((k) => k.toString());
  const localExperiments = keys.map((k) =>
    Fixtures.createExperiment({ experiment_id: k, name: k }),
  );

  const wrapper = mountComponent({
    experiments: localExperiments,
    activeExperimentIds: keys,
  });
  const selected = wrapper.find('[data-test-id="active-experiment-list-item"]');
  expect(selected.length).toBeLessThan(keys.length);
});
