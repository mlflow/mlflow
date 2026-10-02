import { describe, it, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import SkillDetailPage from './SkillDetailPage';
import SkillRegistryRoutes from '../routes';
import { SkillStatus } from '../types';
import {
  createMockSkill,
  createMockSkillVersion,
  getMockedGetSkillErrorResponse,
  getMockedGetSkillPermissionDeniedResponse,
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
    expect(screen.getAllByText('draft').length).toBeGreaterThanOrEqual(1);
  });

  it('restores the selected version from the URL', async () => {
    renderPage([`${SkillRegistryRoutes.getSkillDetailRoute(mockSkill.name, mockSkill.organization)}?version=1`]);

    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });
    expect(screen.getByRole('row', { selected: true })).toHaveTextContent('Version 1');
  });

  it('omits deleted versions from ordinary search results', async () => {
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
});
