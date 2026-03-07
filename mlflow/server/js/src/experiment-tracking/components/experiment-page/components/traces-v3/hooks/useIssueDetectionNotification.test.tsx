import { describe, test, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { renderHook, act, waitFor, screen } from '@testing-library/react';
import { renderWithDesignSystem } from '../../../../../../common/utils/TestUtils.react18';
import { useIssueDetectionNotification } from './useIssueDetectionNotification';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from '../../../../../../common/utils/RoutingUtils';
import React from 'react';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <MemoryRouter>
    <IntlProvider locale="en" messages={{}}>
      <DesignSystemProvider>{children}</DesignSystemProvider>
    </IntlProvider>
  </MemoryRouter>
);

describe('useIssueDetectionNotification', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('returns showIssueDetectionNotification function and notificationContextHolder', () => {
    const { result } = renderHook(() => useIssueDetectionNotification('exp-123'), { wrapper });

    expect(result.current.showIssueDetectionNotification).toBeInstanceOf(Function);
    expect(result.current.notificationContextHolder).toBeDefined();
  });

  test('notification is not visible initially', () => {
    const { result } = renderHook(() => useIssueDetectionNotification('exp-123'), { wrapper });

    const TestComponent = () => <>{result.current.notificationContextHolder}</>;
    renderWithDesignSystem(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    expect(screen.queryByText('Issue detection job triggered')).not.toBeInTheDocument();
  });

  test('notification becomes visible when showIssueDetectionNotification is called', async () => {
    const { result } = renderHook(() => useIssueDetectionNotification('exp-123'), { wrapper });

    const TestComponent = () => <>{result.current.notificationContextHolder}</>;
    const { rerender } = renderWithDesignSystem(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    act(() => {
      result.current.showIssueDetectionNotification();
    });

    rerender(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText('Issue detection job triggered')).toBeInTheDocument();
    });
  });

  test('notification shows View status link when experimentId is provided', async () => {
    const { result } = renderHook(() => useIssueDetectionNotification('exp-123'), { wrapper });

    const TestComponent = () => <>{result.current.notificationContextHolder}</>;
    const { rerender } = renderWithDesignSystem(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    act(() => {
      result.current.showIssueDetectionNotification();
    });

    rerender(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText('View status')).toBeInTheDocument();
    });

    const link = screen.getByText('View status').closest('a');
    expect(link).toHaveAttribute('href', expect.stringContaining('/experiments/exp-123/evaluation-runs'));
  });

  test('notification does not show View status link when experimentId is not provided', async () => {
    const { result } = renderHook(() => useIssueDetectionNotification(undefined), { wrapper });

    const TestComponent = () => <>{result.current.notificationContextHolder}</>;
    const { rerender } = renderWithDesignSystem(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    act(() => {
      result.current.showIssueDetectionNotification();
    });

    rerender(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText('Issue detection job triggered')).toBeInTheDocument();
    });

    expect(screen.queryByText('View status')).not.toBeInTheDocument();
  });

  test('notification auto-dismisses after 10 seconds', async () => {
    const { result } = renderHook(() => useIssueDetectionNotification('exp-123'), { wrapper });

    const TestComponent = () => <>{result.current.notificationContextHolder}</>;
    const { rerender } = renderWithDesignSystem(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    act(() => {
      result.current.showIssueDetectionNotification();
    });

    rerender(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.getByText('Issue detection job triggered')).toBeInTheDocument();
    });

    act(() => {
      jest.advanceTimersByTime(10000);
    });

    rerender(
      <MemoryRouter>
        <TestComponent />
      </MemoryRouter>,
    );

    await waitFor(() => {
      expect(screen.queryByText('Issue detection job triggered')).not.toBeInTheDocument();
    });
  });
});
