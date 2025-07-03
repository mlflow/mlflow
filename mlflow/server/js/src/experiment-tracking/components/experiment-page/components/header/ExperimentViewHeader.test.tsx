import { ExperimentViewHeader } from './ExperimentViewHeader';
import { renderWithIntl, act, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ExperimentEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { BrowserRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { Provider } from 'react-redux';
import thunk from 'redux-thunk';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';

// mock breadcrumbs
jest.mock('@databricks/design-system', () => ({
  ...jest.requireActual<typeof import('@databricks/design-system')>('@databricks/design-system'),
  Breadcrumb: () => <div />,
}));

describe('ExperimentViewHeader', () => {
  const experiment: ExperimentEntity = {
    experimentId: '123',
    name: 'test',
    artifactLocation: 'file:/tmp/mlruns',
    lifecycleStage: 'active',
    allowedActions: [],
    creationTime: 0,
    lastUpdateTime: 0,
    tags: [],
  };

  const setEditing = (editing: boolean) => {
    return;
  };

  const createComponentMock = (showAddDescriptionButton: boolean) => {
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    return renderWithIntl(
      <BrowserRouter>
        <DesignSystemProvider>
          <Provider
            store={mockStore({
              entities: {
                experimentsById: {},
              },
            })}
          >
            <ExperimentViewHeader
              experiment={experiment}
              showAddDescriptionButton={showAddDescriptionButton}
              setEditing={setEditing}
            />
          </Provider>
        </DesignSystemProvider>
      </BrowserRouter>,
    );
  };

  test('should render add description button', async () => {
    await act(async () => {
      createComponentMock(true);
    });

    expect(screen.queryByText('Add Description')).toBeInTheDocument();
  });

  test('should not render add description button', async () => {
    await act(async () => {
      createComponentMock(false);
    });

    expect(screen.queryByText('Add Description')).not.toBeInTheDocument();
  });

  test('If button to delete experiment is pressed then open DeleteExperimentModal', async () => {
    await act(async () => {
      createComponentMock(true);
    });

    await userEvent.click(screen.getByLabelText('Open header dropdown menu'));
    await userEvent.click(screen.getByText('Delete'));
    expect(screen.getByText(/Delete Experiment/)).toBeInTheDocument();
  });

  test('If button to rename experiment is pressed then open RenameExperimentModal', async () => {
    await act(async () => {
      createComponentMock(true);
    });

    await userEvent.click(screen.getByLabelText('Open header dropdown menu'));
    await userEvent.click(screen.getByText('Rename'));
    expect(screen.getByText(/Rename Experiment/)).toBeInTheDocument();
  });
});
