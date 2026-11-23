import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { ExperimentViewManagementMenu } from './ExperimentViewManagementMenu';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { BrowserRouter, MemoryRouter, useLocation } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import type { ExperimentEntity } from '@mlflow/mlflow/src/experiment-tracking/types';
import userEvent from '@testing-library/user-event';
import { Provider } from 'react-redux';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import { shouldEnableExperimentSettingsPage } from '../../../../../common/utils/FeatureUtils';

jest.mock('../../../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../../../common/utils/FeatureUtils')>(
    '../../../../../common/utils/FeatureUtils',
  ),
  shouldEnableExperimentSettingsPage: jest.fn(() => false),
}));

describe('ExperimentViewManagementMenu', () => {
  afterEach(() => {
    jest.mocked(shouldEnableExperimentSettingsPage).mockReturnValue(false);
  });

  const defaultExperiment: ExperimentEntity = {
    experimentId: '123',
    name: 'test/experiment/name',
    artifactLocation: 'file:/tmp/mlruns',
    lifecycleStage: 'active',
    allowedActions: ['RENAME', 'DELETE'],
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

  test('it should render the management menu with only delete when editing is unavailable', async () => {
    renderTestComponent();

    const menuTrigger = screen.getByTestId('overflow-menu-trigger');
    expect(menuTrigger).toBeInTheDocument();

    await userEvent.click(menuTrigger);

    expect(screen.queryByText('Edit experiment')).not.toBeInTheDocument();
    expect(screen.queryByText('Rename')).not.toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
  });

  test('it should render the edit experiment button when setEditing is provided', async () => {
    const setEditing = jest.fn();
    renderTestComponent({ setEditing });

    const menuTrigger = screen.getByTestId('overflow-menu-trigger');
    await userEvent.click(menuTrigger);

    const editButton = await screen.findByText('Edit experiment');
    expect(editButton).toBeInTheDocument();
    expect(screen.queryByText('Rename')).not.toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();

    await userEvent.click(editButton);
    expect(setEditing).toHaveBeenCalledWith(true);
  });

  test('it should render the edit experiment button when only metadata modification is allowed', async () => {
    const setEditing = jest.fn();
    renderTestComponent({
      setEditing,
      experiment: { ...defaultExperiment, allowedActions: ['MODIFIY_PERMISSION'] },
    });

    const menuTrigger = screen.getByTestId('overflow-menu-trigger');
    await userEvent.click(menuTrigger);

    expect(await screen.findByText('Edit experiment')).toBeInTheDocument();
    expect(screen.queryByText('Delete')).not.toBeInTheDocument();
  });

  test('it should only render delete when editing permissions are not granted', async () => {
    renderTestComponent({
      setEditing: jest.fn(),
      experiment: { ...defaultExperiment, allowedActions: ['DELETE'] },
    });

    const menuTrigger = screen.getByTestId('overflow-menu-trigger');
    await userEvent.click(menuTrigger);

    expect(screen.queryByText('Edit experiment')).not.toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
  });

  test('it should hide the menu when no edit or delete actions are allowed', () => {
    renderTestComponent({
      setEditing: jest.fn(),
      experiment: { ...defaultExperiment, allowedActions: [] },
    });

    expect(screen.queryByTestId('overflow-menu-trigger')).not.toBeInTheDocument();
  });

  test('it should show menu actions when allowed actions are omitted', async () => {
    const setEditing = jest.fn();
    renderTestComponent({
      setEditing,
      experiment: { ...defaultExperiment, allowedActions: undefined },
    });

    const menuTrigger = screen.getByTestId('overflow-menu-trigger');
    await userEvent.click(menuTrigger);

    expect(await screen.findByText('Edit experiment')).toBeInTheDocument();
    expect(screen.getByText('Delete')).toBeInTheDocument();
  });

  describe('with the experiment Settings page', () => {
    test('offers Settings and Edit description instead of Edit experiment', async () => {
      jest.mocked(shouldEnableExperimentSettingsPage).mockReturnValue(true);
      const setEditing = jest.fn();
      renderTestComponent({
        setEditing,
        experiment: { ...defaultExperiment, allowedActions: ['RENAME', 'MODIFIY_PERMISSION', 'DELETE'] },
      });

      await userEvent.click(screen.getByTestId('overflow-menu-trigger'));

      expect(screen.getAllByRole('menuitem').map((item) => item.textContent)).toEqual([
        'Settings',
        'Edit description',
        'Delete',
      ]);
      await userEvent.click(screen.getByText('Edit description'));
      expect(setEditing).toHaveBeenCalledWith(true);
    });

    test('does not offer Edit description when only renaming is allowed', async () => {
      jest.mocked(shouldEnableExperimentSettingsPage).mockReturnValue(true);
      renderTestComponent({ setEditing: jest.fn(), experiment: { ...defaultExperiment, allowedActions: ['RENAME'] } });

      await userEvent.click(screen.getByTestId('overflow-menu-trigger'));

      expect(screen.getByText('Settings')).toBeInTheDocument();
      expect(screen.queryByText('Edit description')).not.toBeInTheDocument();
      expect(screen.queryByText('Edit experiment')).not.toBeInTheDocument();
    });
  });

  test('it should not offer Settings without the experiment Settings page', async () => {
    jest.mocked(shouldEnableExperimentSettingsPage).mockReturnValue(false);
    renderTestComponent();

    await userEvent.click(screen.getByTestId('overflow-menu-trigger'));

    expect(screen.queryByText('Settings')).not.toBeInTheDocument();
  });

  test('it should open the experiment Settings page from the menu', async () => {
    jest.mocked(shouldEnableExperimentSettingsPage).mockReturnValue(true);
    renderTestComponent();

    await userEvent.click(screen.getByTestId('overflow-menu-trigger'));
    await userEvent.click(screen.getByText('Settings'));

    expect(window.location.pathname).toBe('/experiments/123/settings');
  });

  test('it should keep the workflow type and workspace when opening Settings', async () => {
    jest.mocked(shouldEnableExperimentSettingsPage).mockReturnValue(true);
    const LocationDisplay = () => {
      const { pathname, search } = useLocation();
      return <div data-testid="location">{`${pathname}${search}`}</div>;
    };
    const mockStore = configureStore([thunk, promiseMiddleware()]);
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter initialEntries={['/experiments/123/runs?workflowType=machine_learning&workspace=team-a&x=1']}>
          <Provider store={mockStore({ entities: { experimentsById: {} } })}>
            <IntlProvider locale="en">
              <DesignSystemProvider>
                <ExperimentViewManagementMenu experiment={defaultExperiment} />
                <LocationDisplay />
              </DesignSystemProvider>
            </IntlProvider>
          </Provider>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    await userEvent.click(screen.getByTestId('overflow-menu-trigger'));
    await userEvent.click(screen.getByText('Settings'));

    expect(screen.getByTestId('location')).toHaveTextContent(
      '/experiments/123/settings?workflowType=machine_learning&workspace=team-a',
    );
  });
});
