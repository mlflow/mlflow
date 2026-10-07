import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import { SkillCard } from './SkillCard';
import { createMockSkill } from '../test-utils';
import type { Skill } from '../types';

const renderCard = (skill: Skill) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <IntlProvider locale="en">
      <TestRouter
        routes={[
          testRoute(
            <QueryClientProvider client={queryClient}>
              <DesignSystemProvider>
                <SkillCard skill={skill} />
              </DesignSystemProvider>
            </QueryClientProvider>,
            '/',
          ),
        ]}
      />
    </IntlProvider>,
  );
};

describe('SkillCard', () => {
  it('renders the skill name, organization footer, description, version, and tags', () => {
    renderCard(
      createMockSkill({
        name: 'cluster-inventory',
        organization: 'ocp-admin',
        description: 'List and inspect clusters',
        latest_version: 4,
        tags: { category: 'monitoring' },
      }),
    );

    expect(screen.getByText('cluster-inventory')).toBeInTheDocument();
    expect(screen.queryByText('@ocp-admin/cluster-inventory')).not.toBeInTheDocument();
    expect(screen.getByText('@ocp-admin')).toBeInTheDocument();
    expect(screen.getByText('List and inspect clusters')).toBeInTheDocument();
    expect(screen.getByText('v4')).toBeInTheDocument();
    expect(document.body.textContent).toContain('category');
    expect(document.body.textContent).toContain('monitoring');
  });

  it('omits the organization footer when the skill has no organization', () => {
    renderCard(createMockSkill({ name: 'prompt-style-guide', organization: '' }));
    expect(screen.getByText('prompt-style-guide')).toBeInTheDocument();
    expect(screen.queryByText(/^@/)).not.toBeInTheDocument();
  });

  it('opens the use modal from the card footer without leaving the catalog', async () => {
    renderCard(createMockSkill({ name: 'cluster-inventory', organization: 'ocp-admin' }));

    await userEvent.click(screen.getByRole('button', { name: 'Use' }));

    expect(await screen.findByText('Use @ocp-admin/cluster-inventory')).toBeInTheDocument();
    expect(screen.getByText('Pinned version: v2')).toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
    expect(document.body.textContent).toContain('skills:/@ocp-admin/cluster-inventory/2');
    expect(document.body.textContent).toContain('mlflow skills pull skills:/@ocp-admin/cluster-inventory/2');
    expect(document.body.textContent).toContain('--destination .claude/skills');
  });

  it('closes the use modal without navigating to the skill details page', async () => {
    renderCard(createMockSkill({ name: 'cluster-inventory', organization: 'ocp-admin' }));

    await userEvent.click(screen.getByRole('button', { name: 'Use' }));
    expect(await screen.findByText('Use @ocp-admin/cluster-inventory')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Use' }).parentElement?.parentElement?.children).toHaveLength(2);

    await userEvent.click(await screen.findByRole('button', { name: /close/i }));

    expect(screen.queryByText('Use @ocp-admin/cluster-inventory')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Use' })).toBeInTheDocument();
  });

  describe('name tooltip', () => {
    const hoverName = async (
      name: string,
      { scrollWidth, clientWidth }: { scrollWidth: number; clientWidth: number },
    ) => {
      const nameElement = screen.getByText(name).parentElement as HTMLElement;
      Object.defineProperty(nameElement, 'scrollWidth', { configurable: true, value: scrollWidth });
      Object.defineProperty(nameElement, 'clientWidth', { configurable: true, value: clientWidth });
      await userEvent.hover(nameElement);
    };

    it('shows no tooltip for a name that fits', async () => {
      renderCard(createMockSkill({ name: 'short' }));
      await hoverName('short', { scrollWidth: 50, clientWidth: 100 });
      await new Promise((resolve) => setTimeout(resolve, 500));
      expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    });

    it('shows the full name when the ellipsis cuts it off', async () => {
      const name = 'a-very-long-skill-name-that-does-not-fit-on-the-card';
      renderCard(createMockSkill({ name }));
      await hoverName(name, { scrollWidth: 400, clientWidth: 100 });
      expect(await screen.findByRole('tooltip')).toHaveTextContent(name);
    });
  });
});
