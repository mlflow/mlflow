import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { IntlProvider } from '@databricks/i18n';

import {
  DetectIssuesButton,
  markDetectIssuesGuidanceSeen,
  DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY,
} from './DetectIssuesButton';

describe('DetectIssuesButton', () => {
  const onClickMock = jest.fn();
  const componentId = 'test-detect-issues-button';

  beforeEach(() => {
    onClickMock.mockClear();
    localStorage.clear();
  });

  afterEach(() => {
    jest.restoreAllMocks();
    localStorage.clear();
  });

  const renderTestComponent = (props: { loading?: boolean; disabled?: boolean } = {}) =>
    render(<DetectIssuesButton componentId={componentId} onClick={onClickMock} {...props} />, {
      wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
    });

  test.each([false, true])(
    'synchronizes guidance dismissal from another surface (storage denied: %s)',
    async (denied) => {
      renderTestComponent();
      expect(await screen.findByText('Detect Issues in Your Traces')).toBeInTheDocument();
      if (denied) {
        jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
          throw new Error('Storage denied');
        });
      }
      act(() => markDetectIssuesGuidanceSeen());
      await waitFor(() => expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument());
      await userEvent.click(screen.getByRole('button', { name: 'Detect issues in traces' }));
      expect(onClickMock).toHaveBeenCalledTimes(1);
    },
  );

  test('renders the Detect Issues button', () => {
    renderTestComponent();
    expect(screen.getByText('Detect Issues')).toBeInTheDocument();
  });

  test.each([{ loading: true }, { disabled: true }])(
    'prevents detection and guidance when unavailable (%o)',
    async (props) => {
      renderTestComponent(props);
      const button = screen.getByRole('button', { name: 'Detect issues in traces' });
      expect(button).toBeDisabled();
      expect(button).toHaveAttribute('aria-busy', String(Boolean(props.loading)));
      expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
      await userEvent.click(button);
      expect(onClickMock).not.toHaveBeenCalled();
    },
  );

  test('shows guidance popover on first visit when hasSeenGuidance is false', async () => {
    renderTestComponent();

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
    localStorage.setItem(`${DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY}_v1`, JSON.stringify(true));

    renderTestComponent();

    // Verify the guidance popover is not displayed
    expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
  });

  test('dismissing guidance via close button persists the flag and hides popover', async () => {
    renderTestComponent();

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
    const storedValue = localStorage.getItem(`${DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY}_v1`);
    expect(storedValue).toBe('true');
  });

  test('dismissing guidance via "Got it" button persists the flag and hides popover', async () => {
    renderTestComponent();

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
    const storedValue = localStorage.getItem(`${DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY}_v1`);
    expect(storedValue).toBe('true');
  });

  test('prevents re-showing guidance after it has been dismissed', async () => {
    // First render: guidance should show
    const { unmount } = renderTestComponent();

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
    renderTestComponent();

    // Verify the guidance does not show again
    expect(screen.queryByText('Detect Issues in Your Traces')).not.toBeInTheDocument();
  });

  test('button onClick handler is called when button is clicked', async () => {
    // Set guidance as seen to avoid popover
    localStorage.setItem(`${DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY}_v1`, JSON.stringify(true));

    renderTestComponent();

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
});
