import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { testRoute, TestRouter } from '../../common/utils/RoutingTestUtils';
import SkillDetailPage from './SkillDetailPage';

const renderDetail = (initialEntries: string[]) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TestRouter
          routes={[
            testRoute(<SkillDetailPage />, '/skills/:organization/:skillName'),
            testRoute(<SkillDetailPage />, '/skills/:skillName'),
            testRoute(<div>Catalog</div>, '/skills'),
          ]}
          initialEntries={initialEntries}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('SkillDetailPage', () => {
  it('renders an organization-qualified identity and a catalog breadcrumb', () => {
    renderDetail(['/skills/@acme/code-review']);
    expect(screen.getByText('@acme/code-review')).toBeInTheDocument();
    expect(screen.getByText('Skill details will appear here.')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Skills' })).toHaveAttribute('href', '/skills');
  });

  it('renders a default-organization identity from the unprefixed route', () => {
    renderDetail(['/skills/prompt-style-guide']);
    expect(screen.getByText('prompt-style-guide')).toBeInTheDocument();
  });
});
