import { afterEach, describe, expect, it } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { OpenInPlaygroundButton } from './OpenInPlaygroundButton';
import { PLAYGROUND_SPAN_ID_QUERY_PARAM, PLAYGROUND_TRACE_ID_QUERY_PARAM } from './routes';

const renderOnTracesPage = (element: React.ReactElement) => {
  const router = createMemoryRouter([{ path: '/experiments/:experimentId/traces', element }], {
    initialEntries: ['/experiments/1/traces'],
  });
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <RouterProvider router={router} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return router;
};

describe('OpenInPlaygroundButton', () => {
  afterEach(() => {
    window.localStorage.removeItem('mlflow.activeWorkspace');
  });

  it('opens the playground of the current experiment with the trace and span ids', async () => {
    const router = renderOnTracesPage(<OpenInPlaygroundButton traceId="tr-1" spanId="span-1" />);

    await userEvent.click(screen.getByRole('button', { name: 'Open in Playground' }));

    expect(router.state.location.pathname).toBe('/experiments/1/playground');
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get(PLAYGROUND_TRACE_ID_QUERY_PARAM)).toBe('tr-1');
    expect(params.get(PLAYGROUND_SPAN_ID_QUERY_PARAM)).toBe('span-1');
  });

  it('keeps the active workspace in the playground URL', async () => {
    window.localStorage.setItem('mlflow.activeWorkspace', 'workspace-a');
    const router = renderOnTracesPage(<OpenInPlaygroundButton traceId="tr-1" />);

    await userEvent.click(screen.getByRole('button', { name: 'Open in Playground' }));

    expect(new URLSearchParams(router.state.location.search).get('workspace')).toBe('workspace-a');
  });

  it('renders nothing without a trace id', () => {
    renderOnTracesPage(<OpenInPlaygroundButton traceId={undefined} />);

    expect(screen.queryByRole('button', { name: 'Open in Playground' })).not.toBeInTheDocument();
  });

  it('renders nothing outside a router', () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <OpenInPlaygroundButton experimentId="1" traceId="tr-1" />
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(screen.queryByRole('button', { name: 'Open in Playground' })).not.toBeInTheDocument();
  });
});
