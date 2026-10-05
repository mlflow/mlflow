import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { setupServer } from '../../common/utils/setup-msw';
import SkillRegistryPage from './SkillRegistryPage';
import SkillDetailPage from './SkillDetailPage';
import { rest } from 'msw';
import { getAjaxUrl } from '@mlflow/mlflow/src/common/utils/FetchUtils';
import { setActiveWorkspace } from '../../workspaces/utils/WorkspaceUtils';
import {
  createMockSkill,
  createMockSkillVersion,
  getMockedGetSkillResponse,
  getMockedSearchSkillsErrorResponse,
  getMockedSearchSkillsPermissionDeniedResponse,
  getMockedSearchSkillsResponse,
  getMockedSkillDetailHandlers,
} from '../test-utils';
import { SKILL_QUERY_KEYS } from '../utils';

const BASE_URL = 'ajax-api/3.0/mlflow/skills';

const changeSimpleSelect = async (componentId: string, optionLabel: string) => {
  const trigger = document.querySelector<HTMLElement>(`[data-component-id="${componentId}"]`);
  if (!trigger) throw new Error(`SimpleSelect "${componentId}" not found`);
  await userEvent.click(trigger);
  await userEvent.click(await screen.findByRole('option', { name: optionLabel }));
};

describe('SkillRegistryPage', () => {
  const server = setupServer(getMockedSearchSkillsResponse([]));

  beforeEach(() => {
    setActiveWorkspace(null);
  });

  // An empty catalog shows Create skill in its empty state once the list has loaded.
  const openCreateSkillDialog = async () => {
    await screen.findByText('Register and catalog skills for your organization.');
    await userEvent.click(screen.getByRole('button', { name: 'Create skill' }));
  };

  const renderPage = (initialEntries = ['/skills']) => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={queryClient}>
            <TestRouter
              routes={[
                testRoute(<SkillRegistryPage />, '/skills'),
                testRoute(<SkillDetailPage />, '/skills/:organization/:skillName'),
                testRoute(<SkillDetailPage />, '/skills/:skillKey'),
              ]}
              initialEntries={initialEntries}
            />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );
    return queryClient;
  };

  it('renders the catalog title and empty-registry state', async () => {
    renderPage();
    expect(await screen.findByText('Register and catalog skills for your organization.')).toBeInTheDocument();
    expect(screen.getByText('Skills')).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Tag key')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Tag value')).not.toBeInTheDocument();
  });

  it('keeps the header create button hidden while the first page loads', async () => {
    server.use(rest.get(getAjaxUrl(BASE_URL), (_req, res, ctx) => res(ctx.delay(300), ctx.json({ skills: [] }))));
    renderPage();

    expect(await screen.findByText('Loading skills...')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Create skill' })).not.toBeInTheDocument();
    await openCreateSkillDialog();
    expect(screen.getByRole('dialog', { name: 'Create skill' })).toBeInTheDocument();
  });

  it('orders catalog filters like the prototype', async () => {
    renderPage();

    const search = screen.getByPlaceholderText('Search skills');
    const organization = screen.getByPlaceholderText('All organizations');
    const source = document.querySelector<HTMLElement>(
      '[data-component-id="mlflow.skill_registry.filter.source_type"]',
    );
    const active = screen.getByRole('button', { name: 'Filter by active status' });
    if (!source) throw new Error('Source filter not found');

    expect(search.compareDocumentPosition(active) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(active.compareDocumentPosition(organization) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(organization.compareDocumentPosition(source) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('renders skill cards when data is available', async () => {
    server.use(
      getMockedSearchSkillsResponse([
        createMockSkill({ name: 'code-review', organization: 'acme' }),
        createMockSkill({ name: 'prompt-style-guide', organization: '' }),
      ]),
    );
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('code-review')).toBeInTheDocument();
      expect(screen.getByText('prompt-style-guide')).toBeInTheDocument();
    });
    expect(screen.getByText('@acme')).toBeInTheDocument();
  });

  it('renders a recoverable error with retry', async () => {
    server.use(getMockedSearchSkillsErrorResponse(500, 'Something broke'));
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Something broke')).toBeInTheDocument();
    });
    expect(screen.getByText('Retry')).toBeInTheDocument();
  });

  it('renders a dedicated permission-denied empty state', async () => {
    server.use(getMockedSearchSkillsPermissionDeniedResponse());
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Permission denied')).toBeInTheDocument();
    });
    expect(screen.getByText('Not allowed to search skills')).toBeInTheDocument();
    expect(screen.queryByText('Retry')).not.toBeInTheDocument();
  });

  it('moves Create skill from the header into the empty state when the catalog is empty', async () => {
    renderPage();

    expect(await screen.findByText('Register and catalog skills for your organization.')).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Create skill' })).toHaveLength(1);
    await userEvent.click(screen.getByRole('button', { name: 'Create skill' }));
    expect(screen.getByRole('dialog', { name: 'Create skill' })).toBeInTheDocument();
  });

  it('drops a Git ref when the location changes to another source type', async () => {
    let requestBody: Record<string, unknown> | undefined;
    server.use(
      rest.get(/skills\/@acme\/oci-skill$/, (_req, res, ctx) => res(ctx.status(404), ctx.json({}))),
      rest.post(getAjaxUrl(`${BASE_URL}/register`), async (req, res, ctx) => {
        requestBody = await req.json();
        return res(ctx.status(500), ctx.json({ message: 'stop here' }));
      }),
    );
    renderPage();

    await openCreateSkillDialog();
    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/acme/skills/tree/dev/code-review');
    await userEvent.click(screen.getByRole('button', { name: 'Advanced settings (optional)' }));
    expect(screen.getByLabelText('Branch, tag or commit')).toHaveValue('dev');

    await userEvent.clear(screen.getByLabelText('Location'));
    await userEvent.type(screen.getByLabelText('Location'), 'oci://ghcr.io/acme/oci-skill:1');
    expect(screen.queryByLabelText('Branch, tag or commit')).not.toBeInTheDocument();
    await userEvent.clear(screen.getByLabelText('Name'));
    await userEvent.type(screen.getByLabelText('Name'), '@acme/oci-skill');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => expect(requestBody).toBeDefined());
    expect(requestBody).toMatchObject({ source_type: 'oci', source: 'ghcr.io/acme/oci-skill:1' });
    expect(requestBody).not.toHaveProperty('ref');
  });

  it('shows empty-search copy when filters return no results', async () => {
    let capturedFilter: string | null = null;
    server.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ skills: [], next_page_token: null }));
      }),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Search skills')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByPlaceholderText('Search skills'), 'review');

    await waitFor(
      () => {
        expect(capturedFilter).toBe("search_text ILIKE '%review%'");
        expect(screen.getByText('No skills found')).toBeInTheDocument();
      },
      { timeout: 2000 },
    );
  });

  it('preserves search and filters when switching between grid and table views', async () => {
    const skills = [createMockSkill({ name: 'code-review', organization: 'acme', latest_version: 2 })];
    let capturedFilter: string | null = null;
    server.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ skills, next_page_token: null }));
      }),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('code-review')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByPlaceholderText('Search skills'), 'review');
    await userEvent.click(screen.getByPlaceholderText('All organizations'));
    await userEvent.click(await screen.findByRole('option', { name: '@acme' }));
    await userEvent.click(screen.getByRole('button', { name: 'Filter by active status' }));
    await changeSimpleSelect('mlflow.skill_registry.filter.source_type', 'Git');

    await waitFor(
      () => {
        expect(capturedFilter).toContain("search_text ILIKE '%review%'");
        expect(capturedFilter).toContain("status = 'active'");
        expect(capturedFilter).toContain("organization = 'acme'");
        expect(capturedFilter).toContain("source_type = 'git'");
      },
      { timeout: 2000 },
    );

    await userEvent.click(screen.getByLabelText('List view'));

    await waitFor(() => {
      expect(screen.getByText('Latest version')).toBeInTheDocument();
      expect(screen.getByRole('link', { name: 'code-review' })).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText('Search skills')).toHaveValue('review');
    expect(capturedFilter).toContain("source_type = 'git'");

    await userEvent.click(screen.getByLabelText('Grid view'));

    await waitFor(() => {
      expect(screen.queryByText('Latest version')).not.toBeInTheDocument();
      expect(screen.getByText('code-review')).toBeInTheDocument();
    });
    expect(screen.getByPlaceholderText('Search skills')).toHaveValue('review');
  });

  it('clears the organization typeahead back to all organizations', async () => {
    const skills = [createMockSkill({ name: 'code-review', organization: 'acme' })];
    let capturedFilter: string | null = null;
    server.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilter = req.url.searchParams.get('filter_string');
        return res(ctx.json({ skills, next_page_token: null }));
      }),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('code-review')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByPlaceholderText('All organizations'));
    await userEvent.click(await screen.findByRole('option', { name: '@acme' }));

    await waitFor(() => {
      expect(capturedFilter).toContain("organization = 'acme'");
      expect(screen.getByPlaceholderText('All organizations')).toHaveValue('@acme');
    });

    await userEvent.click(screen.getByRole('button', { name: /clear/i }));

    await waitFor(() => {
      expect(capturedFilter ?? '').not.toContain('organization =');
      expect(screen.getByPlaceholderText('All organizations')).toHaveValue('');
    });
  });

  it('filters organizations that are not in the loaded suggestions', async () => {
    const capturedFilters: (string | null)[] = [];
    server.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        capturedFilters.push(req.url.searchParams.get('filter_string'));
        return res(
          ctx.json({ skills: [createMockSkill({ name: 'code-review', organization: 'acme' })], next_page_token: null }),
        );
      }),
    );
    renderPage();

    await screen.findByText('code-review');
    const organization = screen.getByPlaceholderText('All organizations');
    await userEvent.type(organization, '@other-org');
    await userEvent.tab();

    await waitFor(() => {
      expect(capturedFilters).toContain("organization = 'other-org'");
      expect(organization).toHaveValue('@other-org');
    });
  });

  it('renders list view columns without organization in the name', async () => {
    server.use(
      getMockedSearchSkillsResponse([
        createMockSkill({
          name: 'cluster-inventory',
          organization: 'ocp-admin',
          latest_version: 4,
          source_type: 'mlflow',
        }),
      ]),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('cluster-inventory')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByLabelText('List view'));

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'cluster-inventory' })).toBeInTheDocument();
    });
    expect(screen.queryByRole('link', { name: '@ocp-admin/cluster-inventory' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'cluster-inventory' })).toHaveAttribute(
      'href',
      '/skills/%40ocp-admin%2Fcluster-inventory',
    );
    expect(screen.getByText('@ocp-admin')).toBeInTheDocument();
    expect(screen.getByText('v4')).toBeInTheDocument();
    expect(screen.getByText('MLflow artifacts')).toBeInTheDocument();
    expect(screen.getByText('Last modified')).toBeInTheDocument();
    expect(screen.getByText('Source')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Use' })).toBeInTheDocument();

    const descriptionHeader = screen.getByRole('columnheader', { name: 'Description' });
    const useHeader = screen.getByRole('columnheader', { name: 'Use' });
    const useCell = screen.getByRole('button', { name: 'Use' }).closest('[role="cell"]') as HTMLElement | null;
    expect(descriptionHeader.style.flex).toBe('2');
    expect(useHeader.style.flex).toBe('0 0 72px');
    expect(useHeader.style.justifyContent).toBe('left');
    expect(useCell?.style.flex).toBe('0 0 72px');
    expect(useCell?.style.textAlign).toBe('left');
  });

  it('paginates with next/previous tokens from the catalog page', async () => {
    server.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        const token = req.url.searchParams.get('page_token');
        if (token === 'page-2') {
          return res(ctx.json({ skills: [createMockSkill({ name: 'page-two' })], next_page_token: null }));
        }
        return res(ctx.json({ skills: [createMockSkill({ name: 'page-one' })], next_page_token: 'page-2' }));
      }),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('page-one')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Next'));
    await waitFor(() => {
      expect(screen.getByText('page-two')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Previous'));
    await waitFor(() => {
      expect(screen.getByText('page-one')).toBeInTheDocument();
    });
  });

  it('keeps Previous available when the current card page is empty', async () => {
    server.use(
      rest.get(getAjaxUrl(BASE_URL), (req, res, ctx) => {
        if (req.url.searchParams.get('page_token') === 'page-2') {
          return res(ctx.json({ skills: [], next_page_token: null }));
        }
        return res(ctx.json({ skills: [createMockSkill({ name: 'page-one' })], next_page_token: 'page-2' }));
      }),
    );
    renderPage();

    await screen.findByText('page-one');
    await userEvent.click(screen.getByText('Next'));

    await screen.findByText('No skills yet');
    expect(screen.getByText('Previous')).toBeInTheDocument();
  });

  it('scopes results and cursor history to the active workspace', async () => {
    const requests: { workspace: string | null; pageToken: string | null }[] = [];
    const unblockWorkspaceRequests: (() => void)[] = [];
    server.use(
      rest.get(getAjaxUrl(BASE_URL), async (req, res, ctx) => {
        const workspace = req.headers.get('X-MLFLOW-WORKSPACE');
        const pageToken = req.url.searchParams.get('page_token');
        requests.push({ workspace, pageToken });

        if (workspace === 'team-b') {
          await new Promise<void>((resolve) => unblockWorkspaceRequests.push(resolve));
          return res(ctx.json({ skills: [createMockSkill({ name: 'team-b-page-one' })], next_page_token: null }));
        }

        if (pageToken === 'team-a-page-2') {
          return res(ctx.json({ skills: [createMockSkill({ name: 'team-a-page-two' })], next_page_token: null }));
        }
        return res(
          ctx.json({ skills: [createMockSkill({ name: 'team-a-page-one' })], next_page_token: 'team-a-page-2' }),
        );
      }),
    );

    act(() => setActiveWorkspace('team-a'));
    renderPage();

    try {
      await screen.findByText('team-a-page-one');
      await userEvent.click(screen.getByText('Next'));
      await screen.findByText('team-a-page-two');

      act(() => setActiveWorkspace('team-b'));
      await waitFor(() => expect(requests.some(({ workspace }) => workspace === 'team-b')).toBe(true));

      expect(requests.filter(({ workspace }) => workspace === 'team-b')).toEqual([
        { workspace: 'team-b', pageToken: null },
      ]);
      expect(screen.queryByText('team-a-page-two')).not.toBeInTheDocument();
      expect(screen.getByText('Loading skills...')).toBeInTheDocument();

      unblockWorkspaceRequests.forEach((unblock) => unblock());
      await screen.findByText('team-b-page-one');
    } finally {
      unblockWorkspaceRequests.forEach((unblock) => unblock());
      act(() => setActiveWorkspace(null));
    }
  });

  it('navigates from a catalog card to the Skill detail route', async () => {
    const skill = createMockSkill({ name: 'code-review', organization: 'acme' });
    server.use(
      getMockedSearchSkillsResponse([skill]),
      ...getMockedSkillDetailHandlers(skill, [
        createMockSkillVersion({ version: 2 }),
        createMockSkillVersion({ version: 1 }),
      ]),
    );
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('code-review')).toBeInTheDocument();
    });

    await userEvent.click(document.querySelector('[data-component-id="mlflow.skill_registry.card"]') as HTMLElement);

    await waitFor(() => {
      expect(screen.getByText('Viewing version 2')).toBeInTheDocument();
      expect(screen.getByText('@acme')).toBeInTheDocument();
    });
  });

  it('registers an external source from the catalog and opens the returned version', async () => {
    const skill = createMockSkill({
      name: 'network-policy-architect',
      organization: 'acme',
      latest_version: 1,
    });
    const version = createMockSkillVersion({
      name: 'network-policy-architect',
      organization: 'acme',
      version: 1,
      ref: 'main',
      subpath: 'network-policy-architect',
    });
    let requestBody: unknown;
    let registered = false;
    server.use(
      // The skill does not exist until it is registered, so the name check passes.
      rest.get(/skills\/@acme\/network-policy-architect$/, (_req, res, ctx) =>
        registered ? undefined : res(ctx.status(404), ctx.json({ error_code: 'RESOURCE_DOES_NOT_EXIST' })),
      ),
      rest.post(getAjaxUrl(`${BASE_URL}/register`), async (req, res, ctx) => {
        requestBody = await req.json();
        registered = true;
        return res(ctx.json(version));
      }),
      ...getMockedSkillDetailHandlers(skill, [version]),
    );
    const queryClient = renderPage();
    const invalidateQueries = jest.spyOn(queryClient, 'invalidateQueries');

    await openCreateSkillDialog();
    expect(screen.getByRole('radio', { name: /Import from existing source, e.g. Git, OCI/ })).toBeChecked();
    expect(
      screen.getByLabelText('Location').compareDocumentPosition(screen.getByLabelText('Name')) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    await userEvent.click(screen.getByRole('radio', { name: /Upload a folder/ }));
    expect(screen.getByText('Select the directory containing SKILL.md.')).toBeInTheDocument();
    expect(screen.getByText('Up to 25 MB of files, unless your server sets a different limit.')).toBeInTheDocument();
    expect(screen.getByLabelText('Name')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Advanced settings (optional)' })).toBeInTheDocument();
    expect(document.querySelector('input[type="file"]')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();

    await userEvent.click(screen.getByRole('radio', { name: /Import from existing source, e.g. Git, OCI/ }));
    await userEvent.type(
      screen.getByLabelText('Location'),
      'https://github.com/acme/skills/tree/main/network-policy-architect',
    );
    expect(screen.getByLabelText('Name')).toHaveValue('@acme/network-policy-architect');
    expect(screen.getByText(/Filled in from the source/)).toBeInTheDocument();
    expect(
      screen.getByText('Registers Git https://github.com/acme/skills · branch main · path network-policy-architect'),
    ).toBeInTheDocument();
    expect(screen.getByText('Registering every skill under this folder? Run this instead:')).toBeInTheDocument();
    expect(document.body.textContent).toContain("--subpath 'network-policy-architect'");
    await userEvent.click(screen.getByRole('button', { name: /create through API/ }));
    const snippet = document.body.textContent ?? '';
    expect(snippet).toContain('mlflow skills register git');
    expect(snippet).toContain("--name 'network-policy-architect'");
    expect(snippet).toContain("--organization 'acme'");
    expect(snippet).toContain("--url 'https://github.com/acme/skills'");
    expect(snippet).toContain("--ref 'main'");
    expect(snippet).toContain("--subpath 'network-policy-architect'");
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: /Back to form/ }));
    expect(screen.getByLabelText('Location')).toHaveValue(
      'https://github.com/acme/skills/tree/main/network-policy-architect',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(screen.getByText('Viewing version 1')).toBeInTheDocument();
    });
    expect(requestBody).toMatchObject({
      name: 'network-policy-architect',
      organization: 'acme',
      source: 'https://github.com/acme/skills',
      source_type: 'git',
      ref: 'main',
      subpath: 'network-policy-architect',
      status: 'active',
    });
    expect(invalidateQueries).toHaveBeenCalledWith([SKILL_QUERY_KEYS.SKILLS_LIST]);
    expect(invalidateQueries).toHaveBeenCalledWith([SKILL_QUERY_KEYS.SKILL]);
    expect(invalidateQueries).toHaveBeenCalledWith([SKILL_QUERY_KEYS.SKILL_VERSIONS]);
    expect(invalidateQueries).toHaveBeenCalledWith([SKILL_QUERY_KEYS.SKILL_VERSION]);
  });

  it('refuses to register a new skill under a name that is already taken', async () => {
    let registerCalled = false;
    server.use(
      getMockedGetSkillResponse(createMockSkill({ name: 'skills-developer', organization: 'redhat-ai' })),
      rest.post(getAjaxUrl(`${BASE_URL}/register`), (_req, res, ctx) => {
        registerCalled = true;
        return res(ctx.json({}));
      }),
    );
    renderPage();

    await openCreateSkillDialog();
    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/redhat-ai/skills-developer');
    await userEvent.click(screen.getByLabelText('Name'));
    await userEvent.tab();
    expect(
      await screen.findByText('A skill named "@redhat-ai/skills-developer" is already registered.'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();

    await userEvent.type(screen.getByLabelText('Name'), '-v2');
    expect(screen.queryByText(/is already registered/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Create' })).toBeEnabled();
    await userEvent.clear(screen.getByLabelText('Name'));
    await userEvent.type(screen.getByLabelText('Name'), '@redhat-ai/skills-developer');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(
      await screen.findByText('A skill named "@redhat-ai/skills-developer" is already registered.'),
    ).toBeInTheDocument();
    expect(registerCalled).toBe(false);
  });

  it('offers only Import when the server cannot store uploads', async () => {
    server.use(
      rest.get(getAjaxUrl('ajax-api/3.0/mlflow/server-info'), (_req, res, ctx) =>
        res(ctx.json({ store_type: 'SqlStore', artifact_serving_enabled: false })),
      ),
    );
    renderPage();

    await openCreateSkillDialog();
    expect(screen.getByRole('radio', { name: /Import from existing source/ })).toBeChecked();
    expect(screen.queryByRole('radio', { name: /Upload a folder/ })).not.toBeInTheDocument();
  });

  it('forgets a selected folder when switching away from Upload', async () => {
    renderPage();

    await openCreateSkillDialog();
    await userEvent.click(screen.getByRole('radio', { name: /Upload a folder/ }));
    const content = '---\nname: demo\n---\n# Demo\n';
    const manifest = new File([content], 'SKILL.md');
    // jsdom's File has no text().
    Object.defineProperties(manifest, {
      webkitRelativePath: { value: 'demo/SKILL.md' },
      text: { value: async () => content },
    });
    await userEvent.upload(screen.getByLabelText('Skill folder'), manifest);
    await waitFor(() => expect(screen.getByRole('button', { name: 'Create' })).toBeEnabled());

    await userEvent.click(screen.getByRole('radio', { name: /Import from existing source, e.g. Git, OCI/ }));
    await userEvent.click(screen.getByRole('radio', { name: /Upload a folder/ }));
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('warns that a GitHub link may split a branch name containing a slash', async () => {
    renderPage();

    await openCreateSkillDialog();
    await userEvent.type(
      screen.getByLabelText('Location'),
      'https://github.com/acme/skills/tree/feature/review/skills/code-review',
    );
    expect(screen.getByText(/GitHub links don't show where a branch name ends/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /Advanced settings/ }));
    await userEvent.clear(screen.getByLabelText('Branch, tag or commit'));
    await userEvent.type(screen.getByLabelText('Branch, tag or commit'), 'feature/review');
    expect(screen.queryByText(/GitHub links don't show where a branch name ends/)).not.toBeInTheDocument();
  });

  it('does not register until the name check succeeds', async () => {
    let registerCalled = false;
    server.use(
      rest.get(/skills\/@redhat-ai\/skills-developer$/, (_req, res, ctx) =>
        res(ctx.status(500), ctx.json({ error_code: 'INTERNAL_ERROR', message: 'Database unavailable' })),
      ),
      rest.post(getAjaxUrl(`${BASE_URL}/register`), (_req, res, ctx) => {
        registerCalled = true;
        return res(ctx.json({}));
      }),
    );
    renderPage();

    await openCreateSkillDialog();
    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/redhat-ai/skills-developer');
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(await screen.findByText(/Couldn't check whether this name is already registered/)).toBeInTheDocument();
    expect(registerCalled).toBe(false);
  });

  it('points a whole-repository location at the repository import command', async () => {
    renderPage();

    await openCreateSkillDialog();
    await userEvent.type(screen.getByLabelText('Location'), 'https://github.com/redhat-ai/skills-developer');

    expect(screen.getByText('Registers Git https://github.com/redhat-ai/skills-developer')).toBeInTheDocument();
    expect(screen.getByText('Registering every skill in this repository? Run this instead:')).toBeInTheDocument();
    expect(document.body.textContent).toContain(
      "mlflow skills import --source 'https://github.com/redhat-ai/skills-developer'",
    );
    expect(document.body.textContent).toContain("--organization 'redhat-ai'");
    expect(screen.getByRole('button', { name: 'Copy repository import command' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /create through API/ }));
    await userEvent.click(screen.getByText('Python'));
    expect(document.body.textContent).toContain('mlflow.genai.import_skills(');
    expect(document.body.textContent).toContain('organization="redhat-ai"');
  });

  it('keeps the form and shows the server error when registration is denied', async () => {
    server.use(
      rest.get(/skills\/@redhat-ai\/skills-developer$/, (_req, res, ctx) => res(ctx.status(404), ctx.json({}))),
      rest.post(getAjaxUrl(`${BASE_URL}/register`), (_req, res, ctx) =>
        res(ctx.status(403), ctx.json({ error_code: 'PERMISSION_DENIED', message: 'Not allowed to create skills' })),
      ),
    );
    renderPage();

    await openCreateSkillDialog();
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    expect(screen.getByText('Enter a source location.')).toBeInTheDocument();

    const location = 'https://github.com/redhat-ai/skills-developer.git';
    await userEvent.type(screen.getByLabelText('Location'), location);
    expect(screen.getByLabelText('Name')).toHaveValue('@redhat-ai/skills-developer');

    await userEvent.click(screen.getByRole('button', { name: 'Create' }));
    await waitFor(() => {
      expect(screen.getByText(/Not allowed to create skills/)).toBeInTheDocument();
    });
    expect(screen.getByLabelText('Location')).toHaveValue(location);
    expect(screen.getByLabelText('Name')).toHaveValue('@redhat-ai/skills-developer');
    expect(screen.queryByText(/Viewing version/)).not.toBeInTheDocument();
  });
});
