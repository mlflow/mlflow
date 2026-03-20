import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { IntlProvider } from '@databricks/i18n';

import { DetectIssuesButton } from './DetectIssuesButton';

describe('DetectIssuesButton', () => {
  const onClickMock = jest.fn();
  const componentId = 'test-detect-issues-button';
  const storageKey = 'test.detectIssues.guidanceShown';

  beforeEach(() => {
    onClickMock.mockClear();
    localStorage.clear();
  });

  afterEach(() => {
    jest.restoreAllMocks();
    localStorage.clear();
  });

  const renderTestComponent = (guidanceStorageKey?: string) =>
    render(
      <DetectIssuesButton componentId={componentId} onClick={onClickMock} guidanceStorageKey={guidanceStorageKey} />,
      { wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider> },
    );

  test('renders the Detect Issues button', () => {
    renderTestComponent(storageKey);
    expect(screen.getByText('Detect Issues')).toBeInTheDocument();
  });

  test('shows guidance popover on first visit when hasSeenGuidance is false', async () => {
    renderTestComponent(storageKey);

    // Popover should appear immediately
    await waitFor(() => {
      expect(screen.getByText('Detect Issues in Your Traces')).toBeInTheDocument();
    });

    // Verify the guidance content is displayed
    expect(
      screen.getByText(/Use this button to automatically detect quality issues in your traces using AI/),
    ).toBeInTheDocument();
  });

  test('does not show guidance popover when hasSeenGuidance is true', async () => {
    // Set the localStorage value to indicate guidance has been seen
    localStorage.setItem(`${storageKey}_v1`, JSON.stringify(true));

    renderTestComponent(storageKey);

    // Verify the guidance popover is not displayed
    expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
  });

  test('dismissing guidance via close button persists the flag and hides popover', async () => {
    renderTestComponent(storageKey);

    // Popover should appear immediately
    await waitFor(() => {
      expect(screen.getByText('Detect Issues in Your Traces')).toBeInTheDocument();
    });

    // Click the close button
    const closeButton = screen.getByLabelText('Close guidance');
    await userEvent.click(closeButton);

    // Verify the popover is hidden
    await waitFor(() => {
      expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
    });

    // Verify the flag was persisted to localStorage
    const storedValue = localStorage.getItem(`${storageKey}_v1`);
    expect(storedValue).toBe('true');
  });

  test('dismissing guidance via "Got it" button persists the flag and hides popover', async () => {
    renderTestComponent(storageKey);

    // Popover should appear immediately
    await waitFor(() => {
      expect(screen.getByText('Detect Issues in Your Traces')).toBeInTheDocument();
    });

    // Click the "Got it" button
    const gotItButton = screen.getByText('Got it');
    await userEvent.click(gotItButton);

    // Verify the popover is hidden
    await waitFor(() => {
      expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
    });

    // Verify the flag was persisted to localStorage
    const storedValue = localStorage.getItem(`${storageKey}_v1`);
    expect(storedValue).toBe('true');
  });

  test('prevents re-showing guidance after it has been dismissed', async () => {
    // First render: guidance should show
    const { unmount } = renderTestComponent(storageKey);

    // Popover should appear immediately
    await waitFor(() => {
      expect(screen.getByText('Detect Issues in Your Traces')).toBeInTheDocument();
    });

    // Dismiss the guidance
    const gotItButton = screen.getByText('Got it');
    await userEvent.click(gotItButton);

    // Verify it's hidden
    await waitFor(() => {
      expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
    });

    // Unmount and re-render the component
    unmount();
    renderTestComponent(storageKey);

    // Verify the guidance does not show again
    expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
  });

  test('button onClick handler is called when button is clicked', async () => {
    // Set guidance as seen to avoid popover
    localStorage.setItem(`${storageKey}_v1`, JSON.stringify(true));

    renderTestComponent(storageKey);

    const button = screen.getByText('Detect Issues');
    await userEvent.click(button);

    expect(onClickMock).toHaveBeenCalledTimes(1);
  });

  test('uses default storage key when guidanceStorageKey is not provided', async () => {
    const defaultStorageKey = 'mlflow.detectIssues.guidanceShown';

    renderTestComponent();

    // Popover should appear immediately
    await waitFor(() => {
      expect(screen.getByText('Detect Issues in Your Traces')).toBeInTheDocument();
    });

    // Dismiss the guidance
    const gotItButton = screen.getByText('Got it');
    await userEvent.click(gotItButton);

    // Verify the default storage key was used
    const storedValue = localStorage.getItem(`${defaultStorageKey}_v1`);
    expect(storedValue).toBe('true');
  });

  test('different instances with different storage keys show guidance independently', async () => {
    const storageKey1 = 'test.instance1.guidanceShown';
    const storageKey2 = 'test.instance2.guidanceShown';

    // Mark first instance as seen
    localStorage.setItem(`${storageKey1}_v1`, JSON.stringify(true));

    // Render first instance - should not show guidance
    const { unmount: unmount1 } = render(
      <DetectIssuesButton componentId="button1" onClick={onClickMock} guidanceStorageKey={storageKey1} />,
      { wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider> },
    );

    expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();

    unmount1();

    // Render second instance - should show guidance
    render(<DetectIssuesButton componentId="button2" onClick={onClickMock} guidanceStorageKey={storageKey2} />, {
      wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
    });

    await waitFor(() => {
      expect(screen.getByText('Detect Issues in Your Traces')).toBeInTheDocument();
    });
  });
});
