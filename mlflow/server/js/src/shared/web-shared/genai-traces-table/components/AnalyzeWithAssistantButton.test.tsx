import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { IntlProvider } from '@databricks/i18n';

import { AnalyzeWithAssistantButton } from './AnalyzeWithAssistantButton';
import {
  DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY,
  DETECT_ISSUES_GUIDANCE_DISMISSED_EVENT,
} from './DetectIssuesButton';
import { shouldEnableIssueDetection } from '../../../../common/utils/FeatureUtils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  shouldEnableIssueDetection: jest.fn(),
}));

const ASSISTANT_GUIDANCE_KEY = 'mlflow.assistant.tracesGuidanceShown_v1';
const DETECT_ISSUES_GUIDANCE_KEY = `${DEFAULT_DETECT_ISSUES_GUIDANCE_STORAGE_KEY}_v1`;
const GUIDANCE_TITLE = 'Chat with Traces in Assistant';

describe('AnalyzeWithAssistantButton', () => {
  const onClickMock = jest.fn();
  const componentId = 'test-analyze-with-assistant-button';

  beforeEach(() => {
    onClickMock.mockClear();
    localStorage.clear();
    jest.mocked(shouldEnableIssueDetection).mockReturnValue(false);
  });

  afterEach(() => {
    jest.restoreAllMocks();
    localStorage.clear();
  });

  const renderTestComponent = () =>
    render(<AnalyzeWithAssistantButton componentId={componentId} onClick={onClickMock} />, {
      wrapper: ({ children }) => <IntlProvider locale="en">{children}</IntlProvider>,
    });

  test('renders the Analyze with Assistant button', () => {
    renderTestComponent();
    expect(screen.getByText('Analyze with Assistant')).toBeInTheDocument();
  });

  test('shows guidance on first visit when issue detection is disabled', async () => {
    jest.mocked(shouldEnableIssueDetection).mockReturnValue(false);
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(GUIDANCE_TITLE)).toBeInTheDocument();
    });
  });

  test('waits for the Detect Issues guidance before showing (sequential)', () => {
    // Issue detection is enabled and its guidance has not been dismissed yet, so ours stays hidden.
    jest.mocked(shouldEnableIssueDetection).mockReturnValue(true);
    renderTestComponent();

    expect(screen.queryByText(GUIDANCE_TITLE)).not.toBeInTheDocument();
  });

  test('shows guidance once the Detect Issues guidance has been dismissed', async () => {
    jest.mocked(shouldEnableIssueDetection).mockReturnValue(true);
    localStorage.setItem(DETECT_ISSUES_GUIDANCE_KEY, JSON.stringify(true));
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(GUIDANCE_TITLE)).toBeInTheDocument();
    });
  });

  test('reveals guidance immediately when the Detect Issues dismissal event fires', async () => {
    jest.mocked(shouldEnableIssueDetection).mockReturnValue(true);
    renderTestComponent();

    // Hidden while the Detect Issues guidance is still pending.
    expect(screen.queryByText(GUIDANCE_TITLE)).not.toBeInTheDocument();

    act(() => {
      window.dispatchEvent(new Event(DETECT_ISSUES_GUIDANCE_DISMISSED_EVENT));
    });

    await waitFor(() => {
      expect(screen.getByText(GUIDANCE_TITLE)).toBeInTheDocument();
    });
  });

  test('does not show guidance when it has already been seen', () => {
    localStorage.setItem(ASSISTANT_GUIDANCE_KEY, JSON.stringify(true));
    renderTestComponent();

    expect(screen.queryByText(GUIDANCE_TITLE)).not.toBeInTheDocument();
  });

  test('dismissing guidance via "Got it" persists the flag and hides the popover', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(GUIDANCE_TITLE)).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Got it'));

    await waitFor(() => {
      expect(screen.queryByText(GUIDANCE_TITLE)).not.toBeInTheDocument();
    });
    expect(localStorage.getItem(ASSISTANT_GUIDANCE_KEY)).toBe('true');
  });

  test('dismissing guidance via the close button persists the flag', async () => {
    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByText(GUIDANCE_TITLE)).toBeInTheDocument();
    });

    await userEvent.click(screen.getByLabelText('Close guidance'));

    await waitFor(() => {
      expect(screen.queryByText(GUIDANCE_TITLE)).not.toBeInTheDocument();
    });
    expect(localStorage.getItem(ASSISTANT_GUIDANCE_KEY)).toBe('true');
  });

  test('onClick handler is called when the button is clicked', async () => {
    // Mark guidance as seen so the popover does not intercept the click.
    localStorage.setItem(ASSISTANT_GUIDANCE_KEY, JSON.stringify(true));
    renderTestComponent();

    await userEvent.click(screen.getByText('Analyze with Assistant'));

    expect(onClickMock).toHaveBeenCalledTimes(1);
  });
});
