import { jest, describe, beforeEach, it, expect } from '@jest/globals';
import React from 'react';
import { renderHook, act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { useRunJudgesOnTracesConfiguration } from './useRunScorerInTracesViewConfiguration';
import type { ScorerEvaluation } from '../useEvaluateTracesAsync';
import type { ModelTraceExplorerRunJudgeConfig } from '@databricks/web-shared/model-trace-explorer';

// Wrapper with required providers
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <DesignSystemProvider>
    <IntlProvider locale="en">{children}</IntlProvider>
  </DesignSystemProvider>
);

describe('useRunJudgesOnTracesConfiguration', () => {
  let mockEvaluateTraces: jest.Mock;

  beforeEach(() => {
    mockEvaluateTraces = jest.fn();
  });

  it('should return showRunJudgesModal and RunJudgesModal', () => {
    const { result } = renderHook(() => useRunJudgesOnTracesConfiguration(mockEvaluateTraces, undefined, undefined), {
      wrapper,
    });

    expect(result.current.showRunJudgesModal).toBeInstanceOf(Function);
    expect(result.current.RunJudgesModal).toBeDefined();
    expect(result.current.JudgesStatusBanner).toBeNull();
  });

  it('should show the modal when showRunJudgesModal is called', () => {
    const { result } = renderHook(() => useRunJudgesOnTracesConfiguration(mockEvaluateTraces, undefined, undefined), {
      wrapper,
    });

    act(() => {
      result.current.showRunJudgesModal(['trace-1', 'trace-2']);
    });

    // The modal should now be visible (RunJudgesModal is a JSX element)
    expect(result.current.RunJudgesModal).toBeDefined();
  });

  it('should return null JudgesStatusBanner when there are no evaluations', () => {
    const { result } = renderHook(() => useRunJudgesOnTracesConfiguration(mockEvaluateTraces, {}, undefined), {
      wrapper,
    });

    expect(result.current.JudgesStatusBanner).toBeNull();
  });

  it('should show loading banner when evaluation is in progress', () => {
    const evaluations: Record<string, ScorerEvaluation> = {
      'eval-1': {
        requestKey: 'eval-1',
        label: 'Test Judge',
        jobIds: ['job-1'],
        jobStatuses: {},
        isLoading: true,
      },
    };

    const { result } = renderHook(() => useRunJudgesOnTracesConfiguration(mockEvaluateTraces, evaluations, undefined), {
      wrapper,
    });

    expect(result.current.JudgesStatusBanner).not.toBeNull();

    // Render the banner and check its content
    const { getByText } = render(<>{result.current.JudgesStatusBanner}</>, { wrapper });
    expect(getByText(/Running judge "Test Judge"/)).toBeDefined();
  });

  it('should show error banner when evaluation fails', () => {
    const evaluations: Record<string, ScorerEvaluation> = {
      'eval-1': {
        requestKey: 'eval-1',
        label: 'Failing Judge',
        jobIds: ['job-1'],
        jobStatuses: {},
        isLoading: false,
        error: new Error('Something went wrong'),
      },
    };

    const { result } = renderHook(() => useRunJudgesOnTracesConfiguration(mockEvaluateTraces, evaluations, undefined), {
      wrapper,
    });

    expect(result.current.JudgesStatusBanner).not.toBeNull();

    const { getByText } = render(<>{result.current.JudgesStatusBanner}</>, { wrapper });
    expect(getByText(/Failing Judge/)).toBeDefined();
    expect(getByText(/Something went wrong/)).toBeDefined();
  });

  it('should show success banner when evaluation completes', () => {
    const evaluations: Record<string, ScorerEvaluation> = {
      'eval-1': {
        requestKey: 'eval-1',
        label: 'Success Judge',
        jobIds: ['job-1'],
        jobStatuses: {},
        isLoading: false,
      },
    };

    const { result } = renderHook(() => useRunJudgesOnTracesConfiguration(mockEvaluateTraces, evaluations, undefined), {
      wrapper,
    });

    expect(result.current.JudgesStatusBanner).not.toBeNull();

    const { getByText } = render(<>{result.current.JudgesStatusBanner}</>, { wrapper });
    expect(getByText(/Success Judge.*completed successfully/)).toBeDefined();
  });

  it('should hide dismissed evaluations from the banner', async () => {
    const evaluations: Record<string, ScorerEvaluation> = {
      'eval-1': {
        requestKey: 'eval-1',
        label: 'Judge A',
        jobIds: ['job-1'],
        jobStatuses: {},
        isLoading: false,
      },
      'eval-2': {
        requestKey: 'eval-2',
        label: 'Judge B',
        jobIds: ['job-2'],
        jobStatuses: {},
        isLoading: false,
      },
    };

    // Use a component that renders the banner directly from the hook,
    // so dismiss state updates propagate correctly
    const BannerTestComponent = () => {
      const { JudgesStatusBanner } = useRunJudgesOnTracesConfiguration(mockEvaluateTraces, evaluations, undefined);
      return <div>{JudgesStatusBanner}</div>;
    };

    const { getByText, queryByText } = render(<BannerTestComponent />, { wrapper });

    // Both should be visible initially
    expect(getByText(/Judge A/)).toBeDefined();
    expect(getByText(/Judge B/)).toBeDefined();

    // Dismiss eval-1 by clicking the close button on the first alert
    const user = userEvent.setup();
    const closeButtons = screen.getAllByRole('button');
    await user.click(closeButtons[0]);

    // Judge A should be dismissed, Judge B should remain
    expect(queryByText(/Judge A/)).toBeNull();
    expect(getByText(/Judge B/)).toBeDefined();
  });

  it('should pass subscribeToScorerFinished through', () => {
    const mockSubscribe = jest.fn();
    const { result } = renderHook(
      () =>
        useRunJudgesOnTracesConfiguration(
          mockEvaluateTraces,
          undefined,
          mockSubscribe as unknown as ModelTraceExplorerRunJudgeConfig['subscribeToScorerFinished'],
        ),
      { wrapper },
    );

    expect(result.current.subscribeToScorerFinished).toBe(mockSubscribe);
  });
});
