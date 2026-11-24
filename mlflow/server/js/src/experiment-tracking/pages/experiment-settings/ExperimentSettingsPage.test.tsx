import { afterEach, beforeEach, describe, expect, jest, test } from '@jest/globals';
import { useState } from 'react';
import type { ReactNode } from 'react';
import { act, render, screen, waitFor, within } from '@testing-library/react';
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
import { createMLflowRoutePath, useNavigate } from '../../../common/utils/RoutingUtils';
import { MlflowService } from '../../sdk/MlflowService';
import { useGetExperimentQuery } from '../../hooks/useExperimentQuery';
import { ExperimentPageRevampProvider } from '../experiment-page-tabs/ExperimentPageRevampContext';
import { encodeTraceArchivalRetentionTag, TRACE_ARCHIVAL_RETENTION_TAG_KEY } from '../../../common/utils/traceArchival';
import ExperimentSettingsPage from './ExperimentSettingsPage';

// The client retries failed reads with growing delays before reporting a network error; pass
// requests straight through so a network failure surfaces at once.
jest.mock('@mlflow/mlflow/src/common/utils/graphQLHooks', () => {
  const actual = jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/graphQLHooks')>(
    '@mlflow/mlflow/src/common/utils/graphQLHooks',
  );
  // Takes the RetryLink options but ignores them: ApolloLink's constructor would store them as the request.
  class PassThroughLink extends actual.ApolloLink {
    constructor() {
      super((operation, forward) => forward(operation));
    }
  }
  return { ...actual, RetryLink: PassThroughLink };
});

jest.mock('../../../account/api', () => ({
  AccountApi: {
    getCurrentUser: jest.fn(),
    listUserRoles: jest.fn(),
    listMyPermissions: jest.fn(),
  },
}));

let mockTraceArchivalEnabled = false;
jest.mock('../../hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../hooks/useServerInfo')>('../../hooks/useServerInfo'),
  useWorkspacesEnabled: () => ({ workspacesEnabled: false, loading: false }),
  useTraceArchivalEnabled: () => mockTraceArchivalEnabled,
}));

const mockedAccountApi = jest.mocked(AccountApi);

const baseTags = [
  { __typename: 'MlflowExperimentTag' as const, key: 'mlflow.experimentKind', value: 'genai_development' },
];

const makeExperiment = (
  tags: { key: string; value: string }[] = [],
  overrides: { experimentId?: string; name?: string } = {},
) => ({
  __typename: 'MlflowExperiment' as const,
  artifactLocation: 'mlflow-artifacts:/123',
  creationTime: '0',
  effectiveTraceArchivalRetention: null,
  experimentId: '123',
  lastUpdateTime: '0',
  lifecycleStage: 'active',
  name: 'team/Helpful experiment',
  tags: [...baseTags, ...tags.map((tag) => ({ __typename: 'MlflowExperimentTag' as const, ...tag }))],
  ...overrides,
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

/** Serves the experiment until `failFromNow` is called, then fails every request at the network level. */
const failingAfterwards = (getExperiment = () => makeExperiment()) => {
  let failing = false;
  server.use(
    graphql.query('MlflowGetExperimentQuery', (_req, res, ctx) =>
      failing
        ? res.networkError('Failed to connect')
        : res(
            ctx.data({
              mlflowGetExperiment: {
                __typename: 'MlflowGetExperimentResponse',
                apiError: null,
                experiment: getExperiment(),
              },
            }),
          ),
    ),
  );
  return {
    failFromNow: () => {
      failing = true;
    },
  };
};

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
    mockTraceArchivalEnabled = false;
  });

  const renderPage = (
    entry = createMLflowRoutePath('/experiments/123/settings'),
    { cache = false, page = <ExperimentSettingsPage /> }: { cache?: boolean; page?: ReactNode } = {},
  ) => {
    const queryClient = new QueryClient();
    return render(<div />, {
      wrapper: () => (
        <TestApolloProvider disableCache={!cache}>
          <MockedReduxStoreProvider state={{ entities: { experimentTagsByExperimentId: {}, experimentsById: {} } }}>
            <IntlProvider locale="en">
              <QueryClientProvider client={queryClient}>
                <DesignSystemProvider>
                  <TestRouter
                    history={history}
                    initialEntries={[entry]}
                    routes={[
                      testRoute(
                        <ExperimentPageRevampProvider enabled>{page}</ExperimentPageRevampProvider>,
                        createMLflowRoutePath('/experiments/:experimentId/settings'),
                      ),
                      testRoute(<div>Experiment root</div>, createMLflowRoutePath('/experiments/:experimentId')),
                      testRoute(<div>Experiments list</div>, createMLflowRoutePath('/experiments')),
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
    // OSS hides the experiment type when the GenAI / Model training switch picks the layout.
    expect(within(aboutSection).queryByText('Experiment type')).not.toBeInTheDocument();
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

  test('shows its own load error when only Settings fails, after the experiment page loaded', async () => {
    // Requests succeed until the experiment page has loaded; Settings' own request then fails at the
    // network level, leaving that page's copy intact.
    let failRequests = false;
    server.use(
      graphql.query('MlflowGetExperimentQuery', (_req, res, ctx) =>
        failRequests
          ? res.networkError('Failed to connect')
          : res(
              ctx.data({
                mlflowGetExperiment: {
                  __typename: 'MlflowGetExperimentResponse',
                  apiError: null,
                  experiment: makeExperiment(),
                },
              }),
            ),
      ),
    );
    // Stands in for the experiment page: it loads the experiment first, then renders the tab.
    const ExperimentPage = () => {
      const { data } = useGetExperimentQuery({ experimentId: '123' });
      if (!data) {
        return null;
      }
      failRequests = true;
      return <ExperimentSettingsPage />;
    };
    renderPage(undefined, { cache: true, page: <ExperimentPage /> });

    expect(await screen.findByText(/^Experiment load error:/)).toBeVisible();
    expect(screen.queryByRole('region', { name: 'About' })).not.toBeInTheDocument();
  });

  test('leaves a load error to the experiment page instead of showing it again', async () => {
    let requested = false;
    server.use(
      graphql.query('MlflowGetExperimentQuery', (_req, res, ctx) => {
        requested = true;
        return res(
          ctx.data({
            mlflowGetExperiment: {
              __typename: 'MlflowGetExperimentResponse',
              apiError: {
                __typename: 'ApiError',
                helpUrl: null,
                code: 'RESOURCE_DOES_NOT_EXIST',
                message: 'No Experiment with id=123 exists',
              },
              experiment: null,
            },
          }),
        );
      }),
    );
    renderPage();

    await waitFor(() => expect(requested).toBe(true));
    // The container also wraps the loading skeleton, so its removal means the error has arrived.
    await waitFor(() => expect(screen.queryByTestId('experiment-settings-content')).not.toBeInTheDocument());
    expect(screen.queryByText(/No Experiment with id=123 exists/)).not.toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'About' })).not.toBeInTheDocument();
  });

  test('loads the latest experiment each time Settings opens, not a cached copy', async () => {
    const SettingsToggle = () => {
      const [shown, setShown] = useState(true);
      return (
        <>
          <button onClick={() => setShown((value) => !value)}>Toggle Settings</button>
          {shown && <ExperimentSettingsPage />}
        </>
      );
    };
    renderPage(undefined, { cache: true, page: <SettingsToggle /> });
    expect(await screen.findByRole('textbox', { name: 'Experiment name' })).toHaveValue('team/Helpful experiment');

    // Another page changes the experiment on the server without touching this page's cache.
    server.use(experimentQuery(() => ({ ...makeExperiment(), name: 'team/Renamed elsewhere' })));
    await userEvent.click(screen.getByRole('button', { name: 'Toggle Settings' }));
    await userEvent.click(screen.getByRole('button', { name: 'Toggle Settings' }));

    expect(await screen.findByRole('textbox', { name: 'Experiment name' })).toHaveValue('team/Renamed elsewhere');
  });

  test('does not report a rename as failed when only the refresh after it fails', async () => {
    const { failFromNow } = failingAfterwards();
    const updateExperiment = jest.spyOn(MlflowService, 'updateExperiment').mockImplementation(async () => {
      failFromNow();
      return {};
    });
    renderPage(undefined, { cache: true });

    const input = await screen.findByRole('textbox', { name: 'Experiment name' });
    await waitFor(() => expect(input).toBeEnabled());
    await userEvent.clear(input);
    await userEvent.paste('team/Renamed experiment');
    await userEvent.tab();
    await waitFor(() => expect(updateExperiment).toHaveBeenCalled());
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 300));
    });

    expect(screen.queryByText(/Unable to save the experiment name/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Experiment load error/)).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Experiment name' })).toHaveValue('team/Renamed experiment');
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

  test('shows the skeleton, not the previous experiment, after navigating to another experiment', async () => {
    server.use(
      graphql.query('MlflowGetExperimentQuery', (req, res, ctx) => {
        const { experimentId } = req.variables['input'] as { experimentId: string };
        const isSecond = experimentId === '456';
        return res(
          ctx.delay(isSecond ? 300 : 0),
          ctx.data({
            mlflowGetExperiment: {
              __typename: 'MlflowGetExperimentResponse',
              apiError: null,
              experiment: makeExperiment([], isSecond ? { experimentId, name: 'Second experiment' } : { experimentId }),
            },
          }),
        );
      }),
    );
    const GoToSecondExperiment = () => {
      const navigate = useNavigate();
      return (
        <button type="button" onClick={() => navigate(createMLflowRoutePath('/experiments/456/settings'))}>
          Go to the second experiment
        </button>
      );
    };
    renderPage(undefined, {
      page: (
        <>
          <GoToSecondExperiment />
          <ExperimentSettingsPage />
        </>
      ),
    });
    expect(await screen.findByRole('textbox', { name: 'Experiment name' })).toHaveValue('team/Helpful experiment');

    await userEvent.click(screen.getByRole('button', { name: 'Go to the second experiment' }));

    // While the second experiment loads, the first one's details are gone rather than shown as if they were its.
    await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Experiment name' })).not.toBeInTheDocument());
    expect(screen.queryByText('123')).not.toBeInTheDocument();
    expect(await screen.findByRole('textbox', { name: 'Experiment name' })).toHaveValue('Second experiment');
    expect(screen.getByText('456')).toBeInTheDocument();
  });

  describe('with authentication enabled', () => {
    const permissionsLoaded = async () => {
      await waitFor(() => expect(mockedAccountApi.listMyPermissions).toHaveBeenCalled());
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 200));
      });
    };

    test('has no Permissions section, even for an admin', async () => {
      authEnabledAs({ isAdmin: true });
      renderPage();

      await screen.findByRole('region', { name: 'About' });
      await permissionsLoaded();
      expect(screen.getAllByRole('heading', { level: 3 }).map((heading) => heading.textContent)).toEqual([
        'About',
        'Actions',
      ]);
      expect(screen.queryByRole('button', { name: 'Manage permissions' })).not.toBeInTheDocument();
    });

    test('disables editing and deleting with only READ permission', async () => {
      authEnabledAs({ isAdmin: false, permission: 'READ' });
      server.use(experimentQuery(() => makeExperiment([{ key: 'team', value: 'search' }])));
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      await permissionsLoaded();
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

    test('closes the tag editor and keeps showing Settings when the refresh after a save fails', async () => {
      const { failFromNow } = failingAfterwards(() => makeExperiment([{ key: 'team', value: 'search' }]));
      jest.spyOn(MlflowService, 'setExperimentTag').mockImplementation(async () => {
        failFromNow();
        return {};
      });
      renderPage(undefined, { cache: true });

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      const dialog = await openTagEditor(aboutSection);
      const textboxes = within(dialog).getAllByRole('textbox');
      await userEvent.click(textboxes[textboxes.length - 2]);
      await userEvent.paste('newkey');
      await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));

      await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Add tags' })).not.toBeInTheDocument());
      expect(screen.getByRole('region', { name: 'About' })).toBeVisible();
      expect(screen.queryByText(/Experiment load error/)).not.toBeInTheDocument();
    });

    test('keeps the failed edit after a partly failed save, and retries only what failed', async () => {
      // A server whose tags reflect each successful write, so a refetch would show the partial save.
      let serverTags = [{ key: 'team', value: 'search' }];
      server.use(experimentQuery(() => makeExperiment(serverTags)));
      let ownerAttempts = 0;
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockImplementation(async ({ key, value }) => {
        if (key === 'owner' && ++ownerAttempts === 1) {
          throw new Error('Owner tag failed');
        }
        serverTags = [...serverTags.filter((tag) => tag.key !== key), { key, value }];
        return {};
      });
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      const dialog = await openTagEditor(aboutSection);
      const textboxes = within(dialog).getAllByRole('textbox');
      await userEvent.clear(textboxes[1]);
      await userEvent.paste('platform');
      await userEvent.click(within(dialog).getAllByRole('textbox')[2]);
      await userEvent.paste('owner');
      await userEvent.click(within(dialog).getAllByRole('textbox')[3]);
      await userEvent.paste('alice');
      await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));

      expect(await within(dialog).findByText('Owner tag failed')).toBeVisible();
      // Give any refetch time to land and reset the form.
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 300));
      });
      const values = within(dialog)
        .getAllByRole('textbox')
        .map((box) => (box as HTMLInputElement).value);
      expect(values).toEqual(expect.arrayContaining(['team', 'platform', 'owner', 'alice']));

      await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
      await waitFor(() => expect(dialog).not.toBeInTheDocument());
      expect(setTag.mock.calls.filter(([args]) => args.key === 'team')).toHaveLength(1);
      expect(setTag.mock.calls.filter(([args]) => args.key === 'owner')).toHaveLength(2);
      expect(await within(aboutSection).findByText('owner')).toBeVisible();
    });

    test('saves a change that is undone after a partly failed save', async () => {
      let serverTags = [{ key: 'team', value: 'search' }];
      server.use(experimentQuery(() => makeExperiment(serverTags)));
      let ownerAttempts = 0;
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockImplementation(async ({ key, value }) => {
        if (key === 'owner' && ++ownerAttempts === 1) {
          throw new Error('Owner tag failed');
        }
        serverTags = [...serverTags.filter((tag) => tag.key !== key), { key, value }];
        return {};
      });
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      const dialog = await openTagEditor(aboutSection);
      // Change team's value and add an owner tag. The team change saves; the owner tag fails.
      const textboxes = within(dialog).getAllByRole('textbox');
      await userEvent.clear(textboxes[1]);
      await userEvent.paste('platform');
      await userEvent.click(within(dialog).getAllByRole('textbox')[2]);
      await userEvent.paste('owner');
      await userEvent.click(within(dialog).getAllByRole('textbox')[3]);
      await userEvent.paste('alice');
      await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));
      expect(await within(dialog).findByText('Owner tag failed')).toBeVisible();

      // Put team's value back and save again. The server has the changed value, so that goes back too.
      await userEvent.clear(within(dialog).getAllByRole('textbox')[1]);
      await userEvent.paste('search');
      await userEvent.click(within(dialog).getByRole('button', { name: 'Save' }));

      await waitFor(() => expect(dialog).not.toBeInTheDocument());
      expect(setTag.mock.calls.map(([args]) => [args.key, args.value])).toEqual([
        ['team', 'platform'],
        ['owner', 'alice'],
        ['team', 'search'],
        ['owner', 'alice'],
      ]);
      expect(serverTags).toEqual(
        expect.arrayContaining([
          { key: 'team', value: 'search' },
          { key: 'owner', value: 'alice' },
        ]),
      );
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

  describe('trace archival retention', () => {
    const retentionTag = (retention: string) => ({
      key: TRACE_ARCHIVAL_RETENTION_TAG_KEY,
      value: encodeTraceArchivalRetentionTag(retention),
    });
    const retentionSetTo = (retention: string) => ({
      experiment_id: '123',
      key: TRACE_ARCHIVAL_RETENTION_TAG_KEY,
      value: encodeTraceArchivalRetentionTag(retention),
    });
    const findRetentionInput = async () => {
      const input = await screen.findByRole('textbox', { name: 'Trace archival retention' });
      await waitFor(() => expect(input).toBeEnabled());
      return input;
    };
    const AUTOSAVE_WAIT = { timeout: 3000 };

    beforeEach(() => {
      mockTraceArchivalEnabled = true;
    });

    test('is hidden when the server has trace archival off', async () => {
      mockTraceArchivalEnabled = false;
      renderPage();

      const aboutSection = await screen.findByRole('region', { name: 'About' });
      expect(within(aboutSection).getByRole('textbox', { name: 'Experiment name' })).toBeInTheDocument();
      expect(within(aboutSection).queryByText('Trace archival retention')).not.toBeInTheDocument();
    });

    test('shows the saved override and saves a new one shortly after typing', async () => {
      server.use(experimentQuery(() => makeExperiment([retentionTag('30d')])));
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
      renderPage();

      const input = await findRetentionInput();
      expect(input).toHaveValue('30');
      expect(screen.getByText('Days')).toBeInTheDocument();
      await userEvent.clear(input);
      await userEvent.type(input, '14');

      await waitFor(() => expect(setTag).toHaveBeenCalledWith(retentionSetTo('14d')), AUTOSAVE_WAIT);
      expect(setTag).toHaveBeenCalledTimes(1);
    });

    test('removes the override when cleared', async () => {
      server.use(experimentQuery(() => makeExperiment([retentionTag('30d')])));
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
      const deleteTag = jest.spyOn(MlflowService, 'deleteExperimentTag').mockResolvedValue({});
      renderPage();

      await userEvent.clear(await findRetentionInput());

      await waitFor(
        () => expect(deleteTag).toHaveBeenCalledWith({ experiment_id: '123', key: TRACE_ARCHIVAL_RETENTION_TAG_KEY }),
        AUTOSAVE_WAIT,
      );
      expect(setTag).not.toHaveBeenCalled();
    });

    test('saves right away when the unit changes', async () => {
      server.use(experimentQuery(() => makeExperiment([retentionTag('30d')])));
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
      renderPage();

      await findRetentionInput();
      await userEvent.click(screen.getByText('Days'));
      await userEvent.click(await screen.findByRole('option', { name: 'Hours' }));

      await waitFor(() => expect(setTag).toHaveBeenCalledWith(retentionSetTo('30h')));
    });

    test('shows why a value is invalid and does not save it', async () => {
      const setTag = jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({});
      renderPage();

      await userEvent.type(await findRetentionInput(), '0');

      expect(
        await screen.findByText(
          "Trace archival retention must use the format <int><unit>, where unit is one of 'm', 'h', or 'd'.",
          undefined,
          AUTOSAVE_WAIT,
        ),
      ).toBeInTheDocument();
      expect(setTag).not.toHaveBeenCalled();
    });

    test("shows the server's reason when saving fails, and saves again on Retry", async () => {
      const setTag = jest
        .spyOn(MlflowService, 'setExperimentTag')
        .mockRejectedValueOnce(new Error('Retention exceeds the server maximum'))
        .mockResolvedValue({});
      renderPage();

      await userEvent.type(await findRetentionInput(), '400');
      expect(
        await screen.findByText(
          'Unable to save the trace archival retention: Retention exceeds the server maximum',
          undefined,
          AUTOSAVE_WAIT,
        ),
      ).toBeInTheDocument();

      await userEvent.click(screen.getByRole('button', { name: 'Retry' }));

      await waitFor(() => expect(setTag).toHaveBeenCalledTimes(2));
      expect(setTag).toHaveBeenLastCalledWith(retentionSetTo('400d'));
      await waitFor(() =>
        expect(screen.queryByText(/Unable to save the trace archival retention/)).not.toBeInTheDocument(),
      );
    });

    test('cannot be changed with only READ permission', async () => {
      authEnabledAs({ isAdmin: false, permission: 'READ' });
      server.use(experimentQuery(() => makeExperiment([retentionTag('30d')])));
      renderPage();

      const input = await screen.findByRole('textbox', { name: 'Trace archival retention' });
      await waitFor(() => expect(mockedAccountApi.listMyPermissions).toHaveBeenCalled());
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 200));
      });
      expect(input).toHaveValue('30');
      expect(input).toBeDisabled();
    });
  });
});
