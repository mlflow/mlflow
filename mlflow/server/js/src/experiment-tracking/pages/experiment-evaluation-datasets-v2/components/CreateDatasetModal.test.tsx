import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { QueryClient, QueryClientProvider } from '../../../../common/utils/reactQueryHooks';
import { setupServer } from '../../../../common/utils/setup-msw';
import { setActiveWorkspace } from '../../../../workspaces/utils/WorkspaceUtils';
import { useWorkspacesEnabled } from '../../../hooks/useServerInfo';
import { CreateEvaluationDatasetButton } from '../../experiment-evaluation-datasets/components/CreateEvaluationDatasetButton';
import { CreateDatasetButton } from './CreateDatasetModal';

jest.mock('../../../hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../../hooks/useServerInfo')>('../../../hooks/useServerInfo'),
  useWorkspacesEnabled: jest.fn(),
}));

describe('CreateDatasetButton permissions', () => {
  let permission = 'EDIT';
  let permissionStatus = 200;
  let userStatus = 200;
  let userIsAdmin = false;
  let createStatus = 200;
  let permissionGate = Promise.resolve();
  let permissionRequests: string[] = [];
  let permissionArguments: { username: string | null; resourceType: string | null }[] = [];
  const createdDataset = { dataset_id: 'new-dataset', name: 'New dataset' };

  const server = setupServer(
    rest.get('/ajax-api/2.0/mlflow/users/current', (_req, res, ctx) =>
      res(ctx.status(userStatus), ctx.json({ user: { username: 'dataset-user', is_admin: userIsAdmin } })),
    ),
    rest.get('/ajax-api/3.0/mlflow/users/permissions/get', async (req, res, ctx) => {
      permissionRequests.push(req.url.searchParams.get('resource_id') ?? '');
      permissionArguments.push({
        username: req.url.searchParams.get('username'),
        resourceType: req.url.searchParams.get('resource_type'),
      });
      await permissionGate;
      return res(ctx.status(permissionStatus), ctx.json({ permission, allowed: true }));
    }),
    rest.post('/ajax-api/3.0/mlflow/datasets/create', (_req, res, ctx) =>
      res(ctx.status(createStatus), ctx.json({ dataset: createdDataset, message: 'Dataset creation failed' })),
    ),
  );

  beforeEach(() => {
    permission = 'EDIT';
    permissionStatus = 200;
    userStatus = 200;
    userIsAdmin = false;
    createStatus = 200;
    permissionGate = Promise.resolve();
    permissionRequests = [];
    permissionArguments = [];
    jest.mocked(useWorkspacesEnabled).mockReturnValue({ workspacesEnabled: false, loading: false });
    setActiveWorkspace(null);
  });

  const renderButton = (props: Partial<React.ComponentProps<typeof CreateDatasetButton>> = {}) => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
      logger: { log: jest.fn(), warn: jest.fn(), error: jest.fn() },
    });
    const view = render(<CreateDatasetButton experimentId="exp-1" {...props} />, {
      wrapper: ({ children }) => (
        <QueryClientProvider client={queryClient}>
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        </QueryClientProvider>
      ),
    });
    return { ...view, queryClient };
  };

  test('disables creation while permission is pending, then enables on EDIT', async () => {
    let release = () => {};
    permissionGate = new Promise<void>((resolve) => {
      release = resolve;
    });
    renderButton();
    try {
      expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
      await waitFor(() => expect(permissionRequests).toEqual(['exp-1']));
    } finally {
      release();
    }
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
  });

  test.each(['EDIT', 'MANAGE'])('allows effective %s permission', async (level) => {
    permission = level;
    renderButton();
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1']));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    expect(permissionArguments).toEqual([{ username: 'dataset-user', resourceType: 'experiment' }]);
  });

  test('allows platform admins without an explicit experiment grant', async () => {
    userIsAdmin = true;
    permission = 'READ';
    renderButton();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    expect(permissionRequests).toEqual([]);
  });

  test.each(['READ', 'EDIT'])('legacy trace-export control uses the same %s gate', async (level) => {
    permission = level;
    const { rerender, queryClient } = renderButton();
    rerender(<CreateEvaluationDatasetButton experimentId="exp-1" />);
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1']));
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    const button = screen.getByRole('button', { name: 'Create dataset' });
    if (level === 'READ') {
      expect(button).toBeDisabled();
      await userEvent.click(button);
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    } else {
      expect(button).toBeEnabled();
      await userEvent.click(button);
      await userEvent.type(screen.getByRole('textbox', { name: 'Dataset name' }), 'discarded');
      await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
      await userEvent.click(button);
      expect(screen.getByRole('textbox', { name: 'Dataset name' })).toHaveValue('');
    }
  });

  test.each(['READ', 'USE', 'NO_PERMISSIONS'])('denies effective %s permission', async (level) => {
    permission = level;
    const { queryClient } = renderButton();
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1']));
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    const button = screen.getByRole('button', { name: 'Create dataset' });
    expect(button).toBeDisabled();
    await userEvent.click(button);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await userEvent.hover(button.parentElement as HTMLElement);
    expect(await screen.findByRole('tooltip')).toHaveTextContent('Edit permission');
  });

  test.each([403, 500])('stays disabled when permission lookup returns %s', async (status) => {
    permissionStatus = status;
    const { queryClient } = renderButton();
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1']));
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
  });

  test.each([401, 403, 500])('does not treat a current-user %s as auth disabled', async (status) => {
    userStatus = status;
    const { queryClient } = renderButton();
    await waitFor(() => expect(queryClient.getQueryState(['account_current_user'])?.status).toBe('error'));
    expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
  });

  test('keeps creation disabled after a network failure', async () => {
    server.use(rest.get('/ajax-api/3.0/mlflow/users/permissions/get', (_req, res) => res.networkError('offline')));
    const { queryClient } = renderButton();
    await waitFor(() =>
      expect(
        queryClient
          .getQueryCache()
          .getAll()
          .some((query) => query.state.status === 'error'),
      ).toBe(true),
    );
    expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
  });

  test('button props cannot override a permission denial', async () => {
    permission = 'READ';
    const onClick = jest.fn();
    const { queryClient } = renderButton({ buttonProps: { disabled: false, onClick } });
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1']));
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    await userEvent.click(screen.getByRole('button', { name: 'Create dataset' }));
    expect(onClick).not.toHaveBeenCalled();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('supports servers without auth', async () => {
    userStatus = 404;
    renderButton();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    expect(permissionRequests).toEqual([]);
  });

  test('checks the new experiment rather than reusing an allowed result', async () => {
    const { rerender } = renderButton();
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1']));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    permission = 'READ';
    rerender(<CreateDatasetButton experimentId="exp-2" />);
    expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1', 'exp-2']));
  });

  test('cancel closes the modal and clears its form before reopening', async () => {
    renderButton();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    const button = screen.getByRole('button', { name: 'Create dataset' });
    await userEvent.click(button);
    await userEvent.type(screen.getByRole('textbox', { name: 'Dataset name' }), 'discarded');
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    await userEvent.click(button);
    expect(screen.getByRole('textbox', { name: 'Dataset name' })).toHaveValue('');
  });

  test('checks permissions again when the active workspace changes', async () => {
    jest.mocked(useWorkspacesEnabled).mockReturnValue({ workspacesEnabled: true, loading: false });
    setActiveWorkspace('ws-a');
    const { queryClient } = renderButton();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    permission = 'READ';
    act(() => setActiveWorkspace('ws-b'));
    expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
    await waitFor(() => expect(permissionRequests).toEqual(['exp-1', 'exp-1']));
    await waitFor(() => expect(queryClient.isFetching()).toBe(0));
    expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
  });

  test.each([
    { workspacesEnabled: true, loading: false },
    { workspacesEnabled: false, loading: true },
  ])('waits for workspace context when configuration is %j', async (configuration) => {
    jest.mocked(useWorkspacesEnabled).mockReturnValue(configuration);
    const { queryClient } = renderButton();
    await waitFor(() => expect(queryClient.getQueryState(['account_current_user'])?.status).toBe('success'));
    expect(permissionRequests).toEqual([]);
    expect(screen.getByRole('button', { name: 'Create dataset' })).toBeDisabled();
    act(() => setActiveWorkspace('ws-a'));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    expect(permissionRequests).toEqual(['exp-1']);
  });

  test('keeps an open modal independent of a later permission lookup failure', async () => {
    const { queryClient } = renderButton();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    await userEvent.click(screen.getByRole('button', { name: 'Create dataset' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Dataset name' }), 'Work in progress');
    permissionStatus = 500;
    await act(async () => {
      await queryClient.refetchQueries({ queryKey: ['dataset-create-permission'] });
    });
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Dataset name' })).toHaveValue('Work in progress');
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });

  test('successful creation closes the modal', async () => {
    renderButton();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    const button = screen.getByRole('button', { name: 'Create dataset' });
    await userEvent.click(button);
    await userEvent.type(screen.getByRole('textbox', { name: 'Dataset name' }), 'New dataset');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });

  test('a rejected create leaves the modal open and cancellable', async () => {
    createStatus = 403;
    const { queryClient } = renderButton();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create dataset' })).toBeEnabled());
    const button = screen.getByRole('button', { name: 'Create dataset' });
    await userEvent.click(button);
    await userEvent.type(screen.getByRole('textbox', { name: 'Dataset name' }), 'Rejected dataset');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() =>
      expect(
        queryClient
          .getMutationCache()
          .getAll()
          .some((mutation) => mutation.state.status === 'error'),
      ).toBe(true),
    );
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
  });
});
