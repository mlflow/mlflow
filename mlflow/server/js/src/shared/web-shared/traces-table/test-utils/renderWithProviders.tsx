import { render, type RenderResult } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '../../genai-traces-table/utils/RoutingTestUtils';

/**
 * Render a traces-table component inside the providers it needs: DesignSystem, i18n, and a test
 * router (the session cell's `Link` needs a router context). Reuses the genai-traces-table routing
 * test helper so the `Link`'s `href` resolves in assertions. Awaits route rendering so callers can
 * assert synchronously afterward (the test router renders its routes via Suspense).
 */
export const renderWithProviders = async (ui: React.ReactElement): Promise<RenderResult> => {
  const result = render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TestRouter routes={[testRoute(ui)]} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  await waitForRoutesToBeRendered();
  return result;
};
