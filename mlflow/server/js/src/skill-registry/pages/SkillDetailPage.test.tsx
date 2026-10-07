import { describe, it, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import SkillDetailPage from './SkillDetailPage';
import SkillRegistryRoutes from '../routes';
import { SkillAction, SkillStatus } from '../types';
import {
  createMockSkill,
  createMockSkillVersion,
  getMockedGetSkillErrorResponse,
  getMockedGetSkillPermissionDeniedResponse,
  getMockedGetSkillVersionResponse,
  getMockedSearchSkillVersionsErrorResponse,
  getMockedSearchSkillVersionsResponse,
  getMockedSkillDetailHandlers,
} from '../test-utils';

const mockSkill = createMockSkill({
  description: 'Reviews pull requests',
  tags: { team: 'platform' },
  aliases: [{ alias: 'prod', version: 2 }],
});
const mockVersion2 = createMockSkillVersion({
  version: 2,
  status: SkillStatus.ACTIVE,
  source_type: 'git',
  source: 'https://github.com/acme/skills',
  ref: 'main',
  subpath: 'code-review',
  digest: 'sha256:abc123',
  aliases: ['prod'],
  tags: { env: 'prod' },
  created_by: 'alice@example.com',
});
const mockVersion1 = createMockSkillVersion({
  version: 1,
  status: SkillStatus.DRAFT,
  source_type: 'zip',
  source: 's3://private-bucket/skill.zip',
  ref: null,
  subpath: null,
  digest: 'sha256:def456',
  aliases: [],
  tags: { env: 'dev' },
  created_by: 'carol@example.com',
});

const changeSimpleSelect = async (componentId: string, optionLabel: string) => {
  const trigger = document.querySelector<HTMLElement>(`[data-component-id="${componentId}"]`);
  if (!trigger) throw new Error(`SimpleSelect "${componentId}" not found`);
  await userEvent.click(trigger);
  await userEvent.click(await screen.findByRole('option', { name: optionLabel }));
};

describe('SkillDetailPage', () => {
  const server = setupServer(...getMockedSkillDetailHandlers(mockSkill, [mockVersion2, mockVersion1]));

  beforeEach(() => {
    setActiveWorkspace(null);
  });

  const renderPage = (
    initialEntries = [SkillRegistryRoutes.getSkillDetailRoute(mockSkill.name, mockSkill.organization)],
  ) => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<SkillDetailPage />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <TestRouter
            routes={[
              testRoute(
                <DesignSystemProvider>
                  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
                </DesignSystemProvider>,
                '/skills/:organization/:skillName',
              ),
              testRoute(
                <DesignSystemProvider>
                  <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
                </DesignSystemProvider>,
                '/skills/:skillKey',
              ),
              testRoute(<div data-testid="skill-catalog" />, '/skills'),
            ]}
            initialEntries={initialEntries}
          />
        </IntlProvider>
      ),
    });
  };

  it('renders parent metadata, latest version, and encoded-route identity', async () => {
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });
    expect(screen.getAllByText('code-review').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('@acme')).toBeInTheDocument();
    expect(screen.getByText('Reviews pull requests')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Skills' })).toHaveAttribute('href', '/skills');
    expect(document.body.textContent).toContain('team');
    expect(document.body.textContent).toContain('platform');
    expect(document.body.textContent).toContain('alice@example.com');
  });

  it('loads a two-segment compatibility URL', async () => {
    renderPage(['/skills/@acme/code-review']);
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });
    expect(screen.getByText('@acme')).toBeInTheDocument();
  });

  it('loads a default-organization skill', async () => {
    const unscoped = createMockSkill({ name: 'prompt-style-guide', organization: '', latest_version: 1 });
    const version = createMockSkillVersion({ name: 'prompt-style-guide', organization: '', version: 1 });
    server.use(...getMockedSkillDetailHandlers(unscoped, [version]));
    renderPage([SkillRegistryRoutes.getSkillDetailRoute('prompt-style-guide')]);

    await waitFor(() => {
      expect(screen.getByText('prompt-style-guide')).toBeInTheDocument();
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });
    expect(screen.queryByText('@acme')).not.toBeInTheDocument();
  });

  it('keeps parent fields stable while switching versions', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });

    expect(screen.getByText('https://github.com/acme/skills')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'https://github.com/acme/skills' })).toHaveAttribute(
      'href',
      'https://github.com/acme/skills',
    );

    await userEvent.click(screen.getByText('Version 1'));

    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });
    expect(screen.getByText('@acme')).toBeInTheDocument();
    expect(screen.getByText('Reviews pull requests')).toBeInTheDocument();
    expect(screen.getByText('s3://private-bucket/skill.zip')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 's3://private-bucket/skill.zip' })).not.toBeInTheDocument();
    expect(screen.getByText('carol@example.com')).toBeInTheDocument();
    expect(screen.getAllByText('Draft').length).toBeGreaterThanOrEqual(1);
  });

  it('restores the selected version from the URL', async () => {
    renderPage([`${SkillRegistryRoutes.getSkillDetailRoute(mockSkill.name, mockSkill.organization)}?version=1`]);

    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });
    expect(screen.getByRole('row', { selected: true })).toHaveTextContent('Version 1');
  });

  it('hides deleted versions from the version list', async () => {
    server.use(
      getMockedSearchSkillVersionsResponse([
        mockVersion2,
        createMockSkillVersion({ version: 1, status: SkillStatus.DELETED }),
      ]),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Version 2')).toBeInTheDocument();
    });
    expect(screen.queryByText('Version 1')).not.toBeInTheDocument();
  });

  it('pins the selected version in the Use modal and updates the install destination', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Use' }));
    expect(await screen.findByText('Use @acme/code-review')).toBeInTheDocument();
    expect(screen.getByText('Pinned version: v2')).toBeInTheDocument();
    expect(document.body.textContent).toContain('skills:/@acme/code-review/2');
    expect(document.body.textContent).toContain('mlflow skills pull skills:/@acme/code-review/2');
    expect(document.body.textContent).toContain('--destination .claude/skills');
    expect(document.body.textContent).not.toContain('skills:/@acme/code-review/latest');

    await userEvent.click(screen.getByRole('radio', { name: 'Python' }));
    expect(document.body.textContent).toContain('version=2');

    await userEvent.click(screen.getByRole('radio', { name: 'CLI' }));
    await changeSimpleSelect('mlflow.skill_registry.use_modal.install_target', 'Cursor');
    expect(document.body.textContent).toContain('--destination .cursor/skills');
    expect(document.body.textContent).toContain('skills:/@acme/code-review/2');
  });

  it('changes a version status from the inline editor', async () => {
    let requestBody: unknown;
    server.use(
      rest.patch(/\/versions\/2$/, async (req, res, ctx) => {
        requestBody = await req.json();
        return res(ctx.json({ skill_version: { ...mockVersion2, status: SkillStatus.DEPRECATED } }));
      }),
    );
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'Edit version status' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Deprecated' }));

    await waitFor(() => {
      expect(requestBody).toEqual({ status: SkillStatus.DEPRECATED });
    });
  });

  it('surfaces a failed status update', async () => {
    server.use(
      rest.patch(/\/versions\/2$/, (_req, res, ctx) =>
        res(ctx.status(400), ctx.json({ error_code: 'INVALID_PARAMETER_VALUE', message: 'Invalid status transition' })),
      ),
    );
    renderPage();

    await userEvent.click(await screen.findByRole('button', { name: 'Edit version status' }));
    await userEvent.click(await screen.findByRole('option', { name: 'Draft' }));

    expect(await screen.findByText('Invalid status transition')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Edit version status' })).toBeEnabled();
  });

  it('settles on the empty state after deleting the last version', async () => {
    let deleted = false;
    const lastVersion = { ...mockVersion1, status: SkillStatus.DEPRECATED };
    server.use(
      // The refetch after the delete takes a moment, as it does against a real server.
      rest.get(/skills\/(?:@[^/]+\/)?[^/]+$/, (_req, res, ctx) =>
        res(ctx.delay(deleted ? 200 : 0), ctx.json({ ...mockSkill, aliases: [], latest_version: deleted ? null : 1 })),
      ),
      rest.get(/skills\/(?:@[^/]+\/)?[^/]+\/versions$/, (_req, res, ctx) =>
        res(
          ctx.delay(deleted ? 200 : 0),
          ctx.json({ skill_versions: deleted ? [] : [lastVersion], next_page_token: null }),
        ),
      ),
      rest.get(/skills\/(?:@[^/]+\/)?[^/]+\/versions\/\d+$/, (_req, res, ctx) =>
        deleted
          ? res(ctx.status(404), ctx.json({ error_code: 'RESOURCE_DOES_NOT_EXIST' }))
          : res(ctx.json(lastVersion)),
      ),
      rest.delete(/skills\/(?:@[^/]+\/)?[^/]+\/versions\/\d+$/, (_req, res, ctx) => {
        deleted = true;
        return res(ctx.json({}));
      }),
    );
    renderPage();

    expect(await screen.findByText('Viewing version 1')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Delete version' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Delete version' })).getByRole('button', { name: 'Delete' }),
    );

    expect(await screen.findByText('Select a version to view details.')).toBeInTheDocument();
    expect(screen.queryByText('This version is no longer available.')).not.toBeInTheDocument();
  });

  it('returns to the catalog after deleting the skill without waiting on its own queries', async () => {
    let deleted = false;
    server.use(
      rest.get(/skills\/(?:@[^/]+\/)?[^/]+$/, (_req, res, ctx) =>
        deleted ? res(ctx.status(404), ctx.json({ error_code: 'RESOURCE_DOES_NOT_EXIST' })) : res(ctx.json(mockSkill)),
      ),
      // Refetching a deleted skill's versions is slow and fails; the delete must not wait for it.
      rest.get(/skills\/(?:@[^/]+\/)?[^/]+\/versions$/, (_req, res, ctx) =>
        deleted
          ? res(ctx.delay(2000), ctx.status(404), ctx.json({}))
          : res(ctx.json({ skill_versions: [mockVersion2, mockVersion1], next_page_token: null })),
      ),
      rest.delete(/skills\/(?:@[^/]+\/)?[^/]+$/, (_req, res, ctx) => {
        deleted = true;
        return res(ctx.json({}));
      }),
    );
    renderPage();

    expect(await screen.findByText('Viewing version 2')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'More actions' }));
    await userEvent.click(await screen.findByRole('menuitem', { name: 'Delete' }));
    await userEvent.click(
      within(await screen.findByRole('dialog', { name: 'Delete skill' })).getByRole('button', { name: 'Delete' }),
    );

    expect(await screen.findByTestId('skill-catalog', {}, { timeout: 1000 })).toBeInTheDocument();
  });

  it('shows a deprecated-only skill as deprecated, since it still resolves', async () => {
    server.use(
      ...getMockedSkillDetailHandlers({ ...mockSkill, status: SkillStatus.DEPRECATED, latest_version: 2 }, [
        { ...mockVersion2, status: SkillStatus.DEPRECATED },
      ]),
    );
    renderPage();

    expect(await screen.findByText('Viewing version 2')).toBeInTheDocument();
    const header = screen.getByRole('heading', { level: 2 });
    expect(within(header).getByText('Deprecated')).toBeInTheDocument();
    expect(screen.queryByText('Unavailable')).not.toBeInTheDocument();
  });

  it('shows a skill without versions as unavailable', async () => {
    server.use(...getMockedSkillDetailHandlers({ ...mockSkill, status: null, latest_version: null }, []));
    renderPage();

    expect(await screen.findByText('Unavailable')).toBeInTheDocument();
  });

  it('hides delete actions from a user who can edit but not delete', async () => {
    server.use(
      ...getMockedSkillDetailHandlers({ ...mockSkill, allowed_actions: [SkillAction.USE, SkillAction.UPDATE] }, [
        { ...mockVersion2, status: SkillStatus.DEPRECATED },
        mockVersion1,
      ]),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });
    expect(screen.queryByRole('button', { name: 'Delete version' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'More actions' }));
    expect(await screen.findByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitem', { name: 'Delete' })).not.toBeInTheDocument();
  });

  it('renders a permission-denied empty state', async () => {
    server.use(getMockedGetSkillPermissionDeniedResponse());
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Permission denied')).toBeInTheDocument();
    });
    expect(screen.getByText('Not allowed to view this skill')).toBeInTheDocument();
  });

  it('renders a missing skill empty state', async () => {
    server.use(getMockedGetSkillErrorResponse(404, 'Skill not found'));
    renderPage();

    await waitFor(() => {
      expect(screen.getAllByText('Skill not found').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders a recoverable skill load error', async () => {
    server.use(getMockedGetSkillErrorResponse(500, 'Something broke'));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Failed to load skill')).toBeInTheDocument();
    });
    expect(screen.getByText('Something broke')).toBeInTheDocument();
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('renders a versions load error', async () => {
    server.use(getMockedSearchSkillVersionsErrorResponse(500, 'Versions unavailable'));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Versions unavailable')).toBeInTheDocument();
    });
  });

  it('renders an empty-version parent', async () => {
    server.use(...getMockedSkillDetailHandlers(createMockSkill({ latest_version: null }), []));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('No versions')).toBeInTheDocument();
      expect(screen.getByText('Select a version to view details.')).toBeInTheDocument();
    });
  });

  it('renders a missing selected version', async () => {
    renderPage([`${SkillRegistryRoutes.getSkillDetailRoute(mockSkill.name, mockSkill.organization)}?version=9`]);

    await waitFor(() => {
      expect(screen.getByText('This version is no longer available.')).toBeInTheDocument();
    });
    expect(screen.getByText('code-review')).toBeInTheDocument();
  });

  it('opens the folder picker instead of copying an uploaded version source', async () => {
    const uploaded = createMockSkillVersion({
      version: 2,
      status: SkillStatus.ACTIVE,
      source_type: 'mlflow',
      source: 'mlflow-artifacts:/skills/@acme/code-review/0123456789abcdef0123456789abcdef',
      ref: null,
      subpath: null,
    });
    server.use(...getMockedSkillDetailHandlers(mockSkill, [uploaded, mockVersion1]));
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Create skill version' }));
    expect(screen.getByRole('radio', { name: /Upload a folder/ })).toBeChecked();
    expect(screen.queryByLabelText('Location')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('radio', { name: /Import from existing source/ }));
    expect(screen.getByLabelText('Location')).toHaveValue('');
  });

  it('starts a version from an uploaded one in Import mode when the server cannot store uploads', async () => {
    const uploaded = createMockSkillVersion({
      version: 2,
      status: SkillStatus.ACTIVE,
      source_type: 'mlflow',
      source: 'mlflow-artifacts:/skills/@acme/code-review/0123456789abcdef0123456789abcdef',
      ref: null,
      subpath: null,
    });
    server.use(
      ...getMockedSkillDetailHandlers(mockSkill, [uploaded, mockVersion1]),
      rest.get(getAjaxUrl('ajax-api/3.0/mlflow/server-info'), (_req, res, ctx) =>
        res(ctx.json({ store_type: 'SqlStore', artifact_serving_enabled: false })),
      ),
    );
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Create skill version' }));
    await waitFor(() => {
      expect(screen.queryByRole('radio', { name: /Upload a folder/ })).not.toBeInTheDocument();
    });
    expect(screen.getByRole('radio', { name: /Import from existing source/ })).toBeChecked();
    expect(screen.getByLabelText('Location')).toHaveValue('');
  });

  it('adds an external version without changing the skill identity', async () => {
    const created = createMockSkillVersion({
      version: 3,
      source: 'https://github.com/acme/skills.git',
      ref: null,
      subpath: null,
    });
    let requestUrl = '';
    let requestBody: unknown;
    server.use(
      rest.post(getAjaxUrl('ajax-api/3.0/mlflow/skills/@acme/code-review/versions'), async (req, res, ctx) => {
        requestUrl = req.url.toString();
        requestBody = await req.json();
        return res(ctx.json(created));
      }),
      getMockedSearchSkillVersionsResponse([created, mockVersion2, mockVersion1]),
      getMockedGetSkillVersionResponse([created, mockVersion2, mockVersion1]),
    );
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Create skill version' }));
    expect(screen.getByRole('dialog', { name: 'Create skill version' })).toBeInTheDocument();
    expect(screen.getByText(/Adding a version to @acme\/code-review, or/)).toBeInTheDocument();
    expect(screen.queryByLabelText('Name')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Location')).toHaveValue('https://github.com/acme/skills');
    expect(screen.getByLabelText('Branch, tag or commit')).toHaveValue('main');
    expect(
      screen.getByText(/If this is a branch, pulls of this version get whatever it points to then/),
    ).toBeInTheDocument();
    await userEvent.clear(screen.getByLabelText('Branch, tag or commit'));
    await userEvent.type(screen.getByLabelText('Branch, tag or commit'), '0123abcd');
    expect(screen.getByText(/Pinned to this commit/)).toBeInTheDocument();
    await userEvent.clear(screen.getByLabelText('Branch, tag or commit'));
    await userEvent.type(screen.getByLabelText('Branch, tag or commit'), 'main');
    expect(screen.getByLabelText('Path within the source')).toHaveValue('code-review');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(screen.getByText('Viewing version 3')).toBeInTheDocument();
    });
    expect(requestUrl).toContain('/skills/@acme/code-review/versions');
    expect(requestBody).toMatchObject({
      source: 'https://github.com/acme/skills',
      source_type: 'git',
      ref: 'main',
      subpath: 'code-review',
      status: 'active',
    });
    expect(requestBody).not.toHaveProperty('name');
    expect(requestBody).not.toHaveProperty('organization');
  });
});
