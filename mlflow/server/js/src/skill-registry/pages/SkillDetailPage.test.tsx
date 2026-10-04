import { describe, it, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
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

  it('shows a deleted version as a disabled row and does not open it', async () => {
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
    expect(screen.getByText('Version 1')).toBeInTheDocument();
    expect(screen.getByRole('row', { name: /Version 1/ })).toHaveAttribute('aria-disabled', 'true');
    await userEvent.click(screen.getByText('Version 1'));
    expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
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
    expect(
      screen.getByText(/Adding a version to @acme\/code-review. Its content can come from anywhere/),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText('Name')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Location')).toHaveValue('https://github.com/acme/skills');
    expect(screen.getByLabelText('Branch, tag or commit')).toHaveValue('main');
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
