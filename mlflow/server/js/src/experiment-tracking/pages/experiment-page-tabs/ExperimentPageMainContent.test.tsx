import { describe, expect, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentPageMainContent } from './ExperimentPageMainContent';

describe('ExperimentPageMainContent', () => {
  test('groups shared breadcrumbs, page actions, and routed content in the main landmark', () => {
    render(
      <DesignSystemProvider>
        <ExperimentPageMainContent breadcrumbs={<div>Breadcrumbs</div>} pageActions={<button>Page action</button>}>
          <div>Tab content</div>
        </ExperimentPageMainContent>
      </DesignSystemProvider>,
    );

    const main = screen.getByRole('main');
    expect(main).toContainElement(screen.getByText('Breadcrumbs'));
    expect(main).toContainElement(screen.getByRole('button', { name: 'Page action' }));
    expect(main).toContainElement(screen.getByText('Tab content'));
    expect(getComputedStyle(main).paddingBottom).toBe(getComputedStyle(main).paddingTop);
  });
});
