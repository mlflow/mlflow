import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { MOCK_CHAT_SPAN } from '../../ModelTraceExplorer.test-utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { PLAYGROUND_SPAN_ID_QUERY_PARAM, PLAYGROUND_TRACE_ID_QUERY_PARAM } from '../../routes';
import { ModelTraceExplorerRightPaneHeader } from './ModelTraceExplorerRightPaneHeader';

const renderHeader = (activeSpan: ModelTraceSpanNode) => {
  const router = createMemoryRouter(
    [
      {
        path: '/experiments/:experimentId/traces',
        element: <ModelTraceExplorerRightPaneHeader activeSpan={activeSpan} showAssessmentsToggle={false} />,
      },
    ],
    { initialEntries: ['/experiments/1/traces'] },
  );
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <RouterProvider router={router} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return router;
};

describe('ModelTraceExplorerRightPaneHeader', () => {
  it('opens a chat span in the playground', async () => {
    const router = renderHeader({ ...MOCK_CHAT_SPAN, traceId: 'tr-chat' });

    await userEvent.click(screen.getByRole('button', { name: 'Open span in Playground' }));

    expect(router.state.location.pathname).toBe('/experiments/1/playground');
    const params = new URLSearchParams(router.state.location.search);
    expect(params.get(PLAYGROUND_TRACE_ID_QUERY_PARAM)).toBe('tr-chat');
    expect(params.get(PLAYGROUND_SPAN_ID_QUERY_PARAM)).toBe(String(MOCK_CHAT_SPAN.key));
  });

  it('does not offer the playground for a span without chat messages', () => {
    renderHeader({ ...MOCK_CHAT_SPAN, traceId: 'tr-chat', chatMessages: undefined });

    expect(screen.queryByRole('button', { name: 'Open span in Playground' })).not.toBeInTheDocument();
  });
});
