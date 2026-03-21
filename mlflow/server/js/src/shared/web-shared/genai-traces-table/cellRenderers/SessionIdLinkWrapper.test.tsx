import { describe, test, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { IntlProvider } from '@databricks/i18n';

import { SessionIdLinkWrapper } from './SessionIdLinkWrapper';
import { TestRouter, testRoute, waitForRoutesToBeRendered } from '../utils/RoutingTestUtils';

describe('SessionIdLinkWrapper', () => {
  const renderComponent = (props: { sessionId: string; experimentId: string; traceId?: string }) => {
    return render(
      <IntlProvider locale="en">
        <TestRouter
          routes={[
            testRoute(
              <SessionIdLinkWrapper {...props}>
                <span>Test Content</span>
              </SessionIdLinkWrapper>,
            ),
          ]}
        />
      </IntlProvider>,
    );
  };

  test('should render a link to the session page without traceId when traceId is not provided', async () => {
    renderComponent({
      sessionId: 'test-session-123',
      experimentId: 'exp-456',
    });
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute(
      'href',
      expect.stringMatching(/^(\/ml)?\/experiments\/exp-456\/chat-sessions\/test-session-123$/),
    );
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });
  test('should render a link to the session page with selectedTraceId query parameter when traceId is provided', async () => {
    renderComponent({
      sessionId: 'test-session-123',
      experimentId: 'exp-456',
      traceId: 'trace-789',
    });
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute(
      'href',
      expect.stringMatching(
        /^(\/ml)?\/experiments\/exp-456\/chat-sessions\/test-session-123\?selectedTraceId=trace-789$/,
      ),
    );
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  test('should encode special characters in traceId', async () => {
    renderComponent({
      sessionId: 'test-session-123',
      experimentId: 'exp-456',
      traceId: 'trace/with/special?chars',
    });
    const link = screen.getByRole('link');
    expect(link).toHaveAttribute(
      'href',
      expect.stringMatching(
        /^(\/ml)?\/experiments\/exp-456\/chat-sessions\/test-session-123\?selectedTraceId=trace%2Fwith%2Fspecial%3Fchars$/,
      ),
    );
  });
});
