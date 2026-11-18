import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { graphql } from 'msw';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { IntlProvider } from 'react-intl';
import { AccountApi } from '../../../account/api';
import { setupServer } from '../../../common/utils/setup-msw';
import { TestApolloProvider } from '../../../common/utils/TestApolloProvider';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';
import { ExperimentKind } from '../../constants';
import { MlflowService } from '../../sdk/MlflowService';
import { ExperimentPageRevampProvider } from '../experiment-page-tabs/ExperimentPageRevampContext';
import ExperimentSettingsPage from './ExperimentSettingsPage';

jest.mock('../../../account/api', () => ({
  AccountApi: {
    getCurrentUser: jest.fn(),
    listUserRoles: jest.fn(),
    listMyPermissions: jest.fn(),
  },
}));

jest.mock('../../hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../hooks/useServerInfo')>('../../hooks/useServerInfo'),
  useWorkspacesEnabled: () => ({ workspacesEnabled: false, loading: false }),
}));

const mockedAccountApi = jest.mocked(AccountApi);

const baseTags = [
  { __typename: 'MlflowExperimentTag' as const, key: 'mlflow.experimentKind', value: 'genai_development' },
];

const makeExperiment = (tags: { key: string; value: string }[] = []) => ({
  __typename: 'MlflowExperiment' as const,
  artifactLocation: 'mlflow-artifacts:/123',
  creationTime: '0',
  effectiveTraceArchivalRetention: null,
  experimentId: '123',
  lastUpdateTime: '0',
  lifecycleStage: 'active',
  name: 'team/Helpful experiment',
  tags: [...baseTags, ...tags.map((tag) => ({ __typename: 'MlflowExperimentTag' as const, ...tag }))],
});

const experimentQuery = (getExperiment: () => ReturnType<typeof makeExperiment>) =>
  graphql.query('MlflowGetExperimentQuery', (_req, res, ctx) =>
    res(
      ctx.data({
        mlflowGetExperiment: {
          __typename: 'MlflowGetExperimentResponse',
          apiError: null,
          experiment: getExperiment(),
        },
      }),
    ),
  );

const server = setupServer(experimentQuery(() => makeExperiment()));

const authDisabled = () => {
  mockedAccountApi.getCurrentUser.mockRejectedValue(new Error('Not found'));
};

const authEnabledAs = ({ isAdmin, permission }: { isAdmin: boolean; permission?: string }) => {
  mockedAccountApi.getCurrentUser.mockResolvedValue({
    user: { id: 1, username: 'alice', is_admin: isAdmin },
    is_basic_auth: true,
  });
  mockedAccountApi.listUserRoles.mockResolvedValue({ roles: [] });
  mockedAccountApi.listMyPermissions.mockResolvedValue({
    is_admin: isAdmin,
    permissions: permission
      ? [
          {
            role_id: 1,
            role_name: '__user_1__',
            workspace: 'default',
            resource_type: 'experiment',
            resource_pattern: '123',
            permission,
          },
        ]
      : [],
  });
};

describe('ExperimentSettingsPage', () => {
  const { history } = setupTestRouter();

  beforeEach(() => {
    authDisabled();
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const renderPage = (entry = createMLflowRoutePath('/experiments/123/settings')) => {
    const queryClient = new QueryClient();
    return render(<div />, {
      wrapper: () => (
        <TestApolloProvider disableCache>
          <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {}, experimentsById: {} } }}>
            <IntlProvider locale="en">
              <QueryClientProvider client={queryClient}>
                <DesignSystemProvider>
                  <TestRouter
                    history={history}
                    initialEntries={[entry]}
                    routes={[
                      testRoute(
                        <ExperimentPageRevampProvider enabled>
                          <ExperimentSettingsPage />
                        </ExperimentPageRevampProvider>,
                        createMLflowRoutePath('/experiments/:experimentId/settings'),
                      ),
                      testRoute(<div>Experiment root</div>, createMLflowRoutePath('/experiments/:experimentId')),
                      testRoute(<div>Experiments list</div>, createMLflowRoutePath('/experiments')),
                      testRoute(<div>Admin page</div>, createMLflowRoutePath('/admin')),
                    ]}
                  />
                </DesignSystemProvider>
              </QueryClientProvider>
            </IntlProvider>
          </MockedReduxStoreProvider>
        </TestApolloProvider>
      ),
    });
  };

  test('presents the About and Actions settings when the server has no authentication', async () => {
    renderPage();

    const aboutSection = await screen.findByRole('region', { name: 'About' });
    const actionsSection = screen.getByRole('region', { name: 'Actions' });
    expect(screen.getAllByRole('heading', { level: 3 }).map((heading) => heading.textContent)).toEqual([
      'About',
      'Actions',
    ]);
    expect(screen.queryByRole('region', { name: 'Permissions' })).not.toBeInTheDocument();
    expect(within(aboutSection).getByRole('textbox', { name: 'Experiment name' })).toHaveValue(
      'team/Helpful experiment',
    );
    expect(within(aboutSection).getByText('Experiment ID')).toBeInTheDocument();
    expect(within(aboutSection).getByText('123')).toBeInTheDocument();
    expect(within(aboutSection).getByText('The unique identifier used by MLflow APIs.')).toBeInTheDocument();
    expect(within(aboutSection).getByText('Artifact location')).toBeInTheDocument();
    expect(within(aboutSection).getByText('mlflow-artifacts:/123')).toBeInTheDocument();
    expect(within(aboutSection).getByText('The location where run artifacts are stored.')).toBeInTheDocument();
    expect(within(aboutSection).getByRole('button', { name: 'Copy experiment ID' })).toBeInTheDocument();
    expect(within(aboutSection).getByRole('button', { name: 'Copy artifact location' })).toBeInTheDocument();
    expect(within(aboutSection).getByText('Experiment type')).toBeInTheDocument();
    expect(within(aboutSection).getByText('GenAI apps & agents')).toBeInTheDocument();
    expect(within(aboutSection).queryByText('Path')).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Trace location' })).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Governance' })).not.toBeInTheDocument();
    expect(within(actionsSection).getByText('Delete experiment')).toBeInTheDocument();
    expect(
      within(actionsSection).getByText('Deletes this experiment. Deleted experiments can be restored.'),
    ).toBeInTheDocument();
    expect(
      within(actionsSection)
        .getAllByRole('button')
        .map((button) => button.textContent),
    ).toEqual(['Delete']);
    expect(within(actionsSection).getByRole('button', { name: 'Delete' })).toBeEnabled();
  });

  test('saves the full experiment name, including slashes', async () => {
    const updateExperiment = jest.spyOn(MlflowService, 'updateExperiment').mockResolvedValue({});
    renderPage();

    const input = await screen.findByRole('textbox', { name: 'Experiment name' });
    await userEvent.clear(input);
    await userEvent.paste('team/Renamed experiment');
    await userEvent.tab();

    await waitFor(() => {
      expect(updateExperiment).toHaveBeenCalledWith({ experiment_id: '123', new_name: 'team/Renamed experiment' });
    });
  });

  test('refetches the experiment before leaving Settings after changing to machine learning', async () => {
    let experimentQueryCount = 0;
    let refetchedBeforeNavigation = false;
    server.use(
      experimentQuery(() => {
        experimentQueryCount += 1;
        if (experimentQueryCount > 1 && screen.queryByText('Experiment root') === null) {
          refetchedBeforeNavigation = true;
        }
        return makeExperiment();
      }),
    );
    const updateKind = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
    renderPage();

    // Editing (and so the type dropdown) is enabled once the server reports no authentication.
    await waitFor(() => expect(screen.getByRole('textbox', { name: 'Experiment name' })).toBeEnabled());
    await userEvent.click(screen.getByText('GenAI apps & agents'));
    await userEvent.click(await screen.findByText('Machine learning'));

    await waitFor(() => {
      expect(updateKind).toHaveBeenCalledWith({
        experiment_id: '123',
        key: 'mlflow.experimentKind',
        value: ExperimentKind.CUSTOM_MODEL_DEVELOPMENT,
      });
    });
    expect(await screen.findByText('Experiment root')).toBeInTheDocument();
    expect(experimentQueryCount).toBeGreaterThanOrEqual(2);
    expect(refetchedBeforeNavigation).toBe(true);
  });

  test('deletes the experiment after confirmation and returns to the experiments list', async () => {
    const deleteExperiment = jest.spyOn(MlflowService, 'deleteExperiment').mockResolvedValue({});
    renderPage();

    const actionsSection = await screen.findByRole('region', { name: 'Actions' });
    await userEvent.click(within(actionsSection).getByRole('button', { name: 'Delete' }));
    const dialog = await screen.findByRole('dialog', { name: 'Delete Experiment "team/Helpful experiment"' });
    await userEvent.click(within(dialog).getByRole('button', { name: 'Delete' }));

    await waitFor(() => expect(deleteExperiment).toHaveBeenCalledWith({ experiment_id: '123' }));
    expect(await screen.findByText('Experiments list')).toBeInTheDocument();
  });

  describe('with authentication enabled', () => {
    test('lets an admin open the admin page to manage permissions', async () => {
      authEnabledAs({ isAdmin: true });
      renderPage();

      const permissionsSection = await screen.findByRole('region', { name: 'Permissions' });
      expect(screen.getAllByRole('heading', { level: 3 }).map((heading) => heading.textContent)).toEqual([
        'About',
        'Permissions',
        'Actions',
      ]);
      expect(within(permissionsSection).getByText('Experiment permissions')).toBeInTheDocument();
      const manageButton = within(permissionsSection).getByRole('button', { name: 'Manage permissions' });
      expect(manageButton).toBeEnabled();
      await userEvent.click(manageButton);
      expect(await screen.findByText('Admin page')).toBeInTheDocument();
    });

    test('disables managing permissions for users who are not admins', async () => {
      authEnabledAs({ isAdmin: false, permission: 'MANAGE' });
      renderPage();

      const permissionsSection = await screen.findByRole('region', { name: 'Permissions' });
      expect(within(permissionsSection).getByRole('button', { name: 'Manage permissions' })).toBeDisabled();
    });

    test('disables editing and deleting with only READ permission', async () => {
      authEnabledAs({ isAdmin: false, permission: 'READ' });
      server.use(experimentQuery(() => makeExperiment([{ key: 'team', value: 'search' }])));
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      await screen.findByRole('region', { name: 'Permissions' });
      expect(within(aboutSection).getByRole('textbox', { name: 'Experiment name' })).toBeDisabled();
      expect(within(aboutSection).getByRole('button', { name: 'Edit tags' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled();
    });

    test('allows editing but not deleting with EDIT permission', async () => {
      authEnabledAs({ isAdmin: false, permission: 'EDIT' });
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      await waitFor(() => expect(within(aboutSection).getByRole('textbox', { name: 'Experiment name' })).toBeEnabled());
      expect(within(aboutSection).getByRole('button', { name: 'Edit tags' })).toBeEnabled();
      expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled();
    });
  });

  describe('experiment tags', () => {
    const mockExperimentWithUserTag = () =>
      server.use(experimentQuery(() => makeExperiment([{ key: 'team', value: 'search' }])));

    const openTagEditor = async (aboutSection: HTMLElement) => {
      const editButton = within(aboutSection).getByRole('button', { name: 'Edit tags' });
      await waitFor(() => expect(editButton).toBeEnabled());
      await userEvent.click(editButton);
      return screen.findByRole('dialog', { name: 'Add tags' });
    };

    test('lists only user-facing tags and opens the tag editor', async () => {
      mockExperimentWithUserTag();
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      expect(within(aboutSection).getByText('Tags')).toBeVisible();
      expect(within(aboutSection).getByText('team')).toBeVisible();
      expect(within(aboutSection).getByText(': search')).toBeVisible();
      expect(within(aboutSection).queryByText('mlflow.experimentKind')).not.toBeInTheDocument();

      expect(await openTagEditor(aboutSection)).toBeVisible();
    });

    test('scrolls to the Tags row when opened with the tags hash', async () => {
      const scrollIntoView = jest.fn();
      Element.prototype.scrollIntoView = scrollIntoView;
      mockExperimentWithUserTag();
      renderPage(`${createMLflowRoutePath('/experiments/123/settings')}#tags`);

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      expect(within(aboutSection).getByRole('button', { name: 'Edit tags' })).toBeVisible();
      expect(scrollIntoView).toHaveBeenCalledWith({ block: 'center' });
    });

    test('enables Save only for a valid change', async () => {
      mockExperimentWithUserTag();
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      const dialog = await openTagEditor(aboutSection);
      expect(await within(dialog).findByDisplayValue('team')).toBeVisible();
      const saveButton = within(dialog).getByRole('button', { name: 'Save' });
      expect(saveButton).toBeDisabled();

      const textboxes = within(dialog).getAllByRole('textbox');
      await userEvent.click(textboxes[textboxes.length - 2]);
      await userEvent.paste('newkey');
      await waitFor(() => expect(saveButton).toBeEnabled());
    });

    test('keeps the tag editor open with an error when saving fails', async () => {
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockRejectedValue(new Error('Tag save failed'));
      mockExperimentWithUserTag();
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      const dialog = await openTagEditor(aboutSection);
      const textboxes = within(dialog).getAllByRole('textbox');
      await userEvent.click(textboxes[textboxes.length - 2]);
      await userEvent.paste('newkey');
      await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));

      expect(await within(dialog).findByText('Tag save failed')).toBeVisible();
      expect(dialog).toBeVisible();
      expect(setTag).toHaveBeenCalled();
    });

    test('saves tags from the editor and shows the refetched tags', async () => {
      let tagSaved = false;
      server.use(experimentQuery(() => makeExperiment(tagSaved ? [{ key: 'newkey', value: 'v' }] : [])));
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockImplementation(async () => {
        tagSaved = true;
        return {};
      });
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      const dialog = await openTagEditor(aboutSection);
      const textboxes = within(dialog).getAllByRole('textbox');
      await userEvent.click(textboxes[textboxes.length - 2]);
      await userEvent.paste('newkey');
      await userEvent.click(textboxes[textboxes.length - 1]);
      await userEvent.paste('v');
      await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));

      await waitFor(() => {
        expect(setTag).toHaveBeenCalledWith({ experiment_id: '123', key: 'newkey', value: 'v' });
      });
      expect(await within(aboutSection).findByText('newkey')).toBeVisible();
    });
  });
});
