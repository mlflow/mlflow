import { describe, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
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
import {
  createMockSkill,
  getMockedSearchSkillsErrorResponse,
  getMockedSearchSkillsPermissionDeniedResponse,
  getMockedSearchSkillsResponse,
} from '../test-utils';

const BASE_URL = 'ajax-api/3.0/mlflow/skills';

const changeSimpleSelect = async (componentId: string, optionLabel: string) => {
  const trigger = document.querySelector<HTMLElement>(`[data-component-id="${componentId}"]`);
  if (!trigger) throw new Error(`SimpleSelect "${componentId}" not found`);
  await userEvent.click(trigger);
  await userEvent.click(await screen.findByRole('option', { name: optionLabel }));
};

describe('SkillRegistryPage', () => {
  const server = setupServer(getMockedSearchSkillsResponse([]));

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
                testRoute(<SkillDetailPage />, '/skills/:skillName'),
              ]}
              initialEntries={initialEntries}
            />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );
  };

  it('renders the catalog title and empty-registry state', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('No skills yet')).toBeInTheDocument();
    });
    expect(screen.getByText('Skills')).toBeInTheDocument();
    expect(screen.getByText('Skills you can read will appear here once they are registered.')).toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Tag key')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('Tag value')).not.toBeInTheDocument();
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

  it('navigates from a catalog card to the Skill detail route', async () => {
    server.use(getMockedSearchSkillsResponse([createMockSkill({ name: 'code-review', organization: 'acme' })]));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('code-review')).toBeInTheDocument();
    });

    await userEvent.click(document.querySelector('[data-component-id="mlflow.skill_registry.card"]') as HTMLElement);

    await waitFor(() => {
      expect(screen.getByText('Skill details will appear here.')).toBeInTheDocument();
      expect(screen.getByText('@acme/code-review')).toBeInTheDocument();
    });
  });
});
