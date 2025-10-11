import { render, screen } from '@testing-library/react';
import { ExperimentViewManagementMenu } from './ExperimentViewManagementMenu';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { BrowserRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import type { ExperimentEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';

describe('ExperimentViewManagementMenu', () => {
  const defaultExperiment: ExperimentEntity = {
    experimentId: '123',
    name: 'test/experiment/name',
    artifactLocation: 'file:/tmp/mlruns',
    lifecycleStage: 'active',
    allowedActions: [],
    creationTime: 0,
    lastUpdateTime: 0,
    tags: [],
  };

  const renderTestComponent = (props: Partial<React.ComponentProps<typeof ExperimentViewManagementMenu>> = {}) => {
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    const queryClient = new QueryClient();

    return render(<ExperimentViewManagementMenu experiment={defaultExperiment} {...props} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          <BrowserRouter>
            <Provider
              store={mockStore({
                entities: {
                  experimentsById: {},
                },
              })}
            >
              <IntlProvider locale="en">
                <DesignSystemProvider>{children}</DesignSystemProvider>
              </IntlProvider>
            </Provider>
          </BrowserRouter>
        </QueryClientProvider>
      ),
    });
  };

  test('it should render the management menu with rename and delete buttons', async () => {
    renderTestComponent();

    // Check that the overflow menu trigger is present
    const menuTrigger = screen.getByTestId('overflow-menu-trigger');
    expect(menuTrigger).toBeInTheDocument();

    // Click the menu trigger to open the menu
    await userEvent.click(menuTrigger);

    // Check that rename and delete buttons are present
    expect(await screen.findByText('Rename')).toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
  });

  test('it should render the edit description button when setEditing is provided', async () => {
    const setEditing = jest.fn();
    renderTestComponent({ setEditing });

    // Click the menu trigger to open the menu
    const menuTrigger = screen.getByTestId('overflow-menu-trigger');
    await userEvent.click(menuTrigger);

    // Check that edit description button is present
    expect(await screen.findByText('Edit description')).toBeInTheDocument();
    expect(screen.getByText('Rename')).toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
  });
});
