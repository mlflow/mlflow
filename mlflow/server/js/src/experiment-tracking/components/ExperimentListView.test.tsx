/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import userEvent from '@testing-library/user-event-14';
import { screen, fireEvent, renderWithIntl } from '../../common/utils/TestUtils.react18';
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
import { DesignSystemProvider } from '@databricks/design-system';

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
    general: { iconSize: 24 },
    spacing: { xs: 4 },
  },
};

const mountComponent = (props: any) => {
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  return renderWithIntl(
    <DesignSystemProvider>
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
      </Provider>
      ,
    </DesignSystemProvider>,
  );
};

test('If searchInput is set to "Test" then first shown element in experiment list has the title "Test"', () => {
  mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  const input = screen.getByTestId('search-experiment-input');
  fireEvent.change(input, {
    target: { value: 'Test' },
  });
  expect(screen.getAllByTestId('experiment-list-item')[0].textContent).toContain('Test');
});

test('If searchInput is set to "Test" and default experiment is active then no active element is shown in the experiment list', () => {
  mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  const input = screen.getByTestId('search-experiment-input');
  fireEvent.change(input, {
    target: { value: 'Test ' },
  });
  expect(screen.queryAllByTestId('active-experiment-list-item')).toHaveLength(0);
});

test('If button to create experiment is pressed then open CreateExperimentModal', async () => {
  mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  await userEvent.click(screen.getByTestId('create-experiment-button'));
  expect(screen.getByText('Create Experiment')).toBeInTheDocument();
});

test('If button to delete experiment is pressed then open DeleteExperimentModal', async () => {
  mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  await userEvent.click(screen.getAllByTestId('delete-experiment-button')[0]);
  expect(screen.getByText(`Delete Experiment "${Fixtures.experiments[0].name}"`)).toBeInTheDocument();
});

test('If button to edit experiment is pressed then open RenameExperimentModal', async () => {
  mountComponent({ experiments: Fixtures.experiments, activeExperimentIds: ['0'] });
  await userEvent.click(screen.getAllByTestId('rename-experiment-button')[0]);
  expect(screen.getByText('Rename Experiment')).toBeInTheDocument();
});

test('If activeExperimentIds is defined then choose all the corresponding experiments', () => {
  const localExperiments = [
    Fixtures.createExperiment(),
    Fixtures.createExperiment({ experimentId: '1', name: 'Test' }),
    Fixtures.createExperiment({ experimentId: '2', name: 'Second' }),
    Fixtures.createExperiment({ experimentId: '3', name: 'Third' }),
  ];
  mountComponent({
    experiments: localExperiments,
    activeExperimentIds: ['1', '3'],
  });
  const selected = screen.getAllByTestId('active-experiment-list-item');
  expect(selected.length).toEqual(2);
  expect(selected[0].textContent).toEqual('Test');
  expect(selected[1].textContent).toEqual('Third');
});

test('should render when both experiments and activeExperimentIds are empty', () => {
  mountComponent({
    experiments: [],
    activeExperimentIds: [],
  });

  // Get the sidebar header as proof that the component rendered
  expect(screen.getByText('Experiments')).toBeInTheDocument();
});

test('virtual list should not render everything when there are many experiments', () => {
  const keys = Array.from(Array(1000).keys()).map((k) => k.toString());
  const localExperiments = keys.map((k) => Fixtures.createExperiment({ experimentId: k, name: k }));

  mountComponent({
    experiments: localExperiments,
    activeExperimentIds: keys,
  });
  const selected = screen.getAllByTestId('active-experiment-list-item');
  expect(selected.length).toBeLessThan(keys.length);
});
