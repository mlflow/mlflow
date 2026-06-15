import { describe, jest, it, expect, beforeEach, afterEach } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemEventProvider, DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { FocusedReview } from './FocusedReview';
import type { LabelSchema } from '../../components/label-schemas';
import type { ReviewQueueItem } from './types';
import Utils from '../../../common/utils/Utils';

const FEEDBACK_SUBMITTED_COMPONENT_ID = 'mlflow.experiment-review-queue.focused-review.feedback-submitted';

let mockTraceData: unknown[] = [];
jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  useGetTracesById: () => ({ data: mockTraceData, isLoading: false }),
  ModelTraceExplorer: () => null,
}));

const mockCreateAssessment = jest.fn();
jest.mock('./hooks/useCreateReviewAssessmentMutation', () => ({
  useCreateReviewAssessmentMutation: () => ({
    createReviewAssessmentAsync: mockCreateAssessment,
    isCreatingAssessment: false,
  }),
}));
let mockPriorAnswersResult: { priorAnswers: unknown[]; isLoading: boolean; isFetching: boolean } = {
  priorAnswers: [],
  isLoading: false,
  isFetching: false,
};
jest.mock('./hooks/useTraceAssessmentsQuery', () => ({
  useTraceAssessmentsQuery: () => mockPriorAnswersResult,
}));

const passFailSchema = (schemaId = 's1', name = 'Looks good?', enableComment = false): LabelSchema => ({
  schema_id: schemaId,
  experiment_id: 'exp-1',
  name,
  type: 'FEEDBACK',
  input: { pass_fail: { positive_label: 'Pass', negative_label: 'Fail' } },
  enable_comment: enableComment,
});

const pendingItem: ReviewQueueItem = {
  queue_id: 'rq-1',
  item_type: 'TRACE',
  item_id: 'tr-1',
  status: 'PENDING',
  creation_time_ms: 1_780_000_000_000,
  last_update_time_ms: 1_780_000_000_000,
};

const renderFocused = (
  schemas: LabelSchema[],
  onSetStatus: (status: string) => Promise<void>,
  opts: {
    canReview?: boolean;
    onAssignSelf?: () => void;
    item?: ReviewQueueItem;
    onSelect?: () => void;
    onBack?: () => void;
    eventCallback?: (e: { componentId: string }) => void;
  } = {},
) => {
  const item = opts.item ?? pendingItem;
  return render(
    <DesignSystemEventProvider callback={opts.eventCallback ?? (() => {})}>
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <FocusedReview
            item={item}
            items={[item]}
            schemas={schemas}
            completedBy="tester"
            isSettingStatus={false}
            canReview={opts.canReview ?? true}
            onAssignSelf={opts.onAssignSelf}
            onBack={opts.onBack ?? jest.fn()}
            onSelect={opts.onSelect ?? jest.fn()}
            onSetStatus={onSetStatus}
          />
        </DesignSystemProvider>
      </IntlProvider>
    </DesignSystemEventProvider>,
  );
};

const completeItem: ReviewQueueItem = {
  ...pendingItem,
  status: 'COMPLETE',
  completed_by: 'tester',
  completed_time_ms: 1_780_000_001_000,
};

const declinedItem: ReviewQueueItem = { ...pendingItem, status: 'DECLINED' };

describe('FocusedReview single Pass/Fail auto-submit', () => {
  beforeEach(() => {
    mockCreateAssessment.mockReset();
    mockCreateAssessment.mockImplementation(() => Promise.resolve());
  });

  it('hides Submit and auto-submits the picked answer when the only question is a Pass/Fail', async () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema()], onSetStatus);

    // No explicit Submit button — picking the answer submits.
    expect(screen.queryByText('Submit')).not.toBeInTheDocument();

    fireEvent.click(screen.getByText('Pass'));

    await waitFor(() => expect(onSetStatus).toHaveBeenCalledWith('COMPLETE'));
    expect(mockCreateAssessment).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'Looks good?', value: true, assessmentKind: 'feedback' }),
    );
  });

  it('keeps the Submit button and does not auto-submit with more than one question', () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus);

    expect(screen.getByText('Submit')).toBeInTheDocument();
    fireEvent.click(screen.getAllByText('Pass')[0]);
    expect(onSetStatus).not.toHaveBeenCalled();
    expect(mockCreateAssessment).not.toHaveBeenCalled();
  });

  it('does not auto-submit a single Pass/Fail that also collects a rationale', () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema('s1', 'Looks good?', true)], onSetStatus);

    expect(screen.getByText('Submit')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Pass'));
    expect(onSetStatus).not.toHaveBeenCalled();
  });
});

describe('FocusedReview submit requires at least one answer', () => {
  let errorToastSpy: jest.SpiedFunction<typeof Utils.displayGlobalErrorNotification>;
  beforeEach(() => {
    errorToastSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    mockCreateAssessment.mockReset();
    mockCreateAssessment.mockImplementation(() => Promise.resolve());
  });
  afterEach(() => {
    errorToastSpy.mockRestore();
    mockPriorAnswersResult = { priorAnswers: [], isLoading: false, isFetching: false };
  });

  it('disables Submit until a question is answered, then completes', async () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    // Two questions so the explicit Submit button renders (not the auto-submit case).
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus);

    // Nothing answered yet — completing now would record no assessments.
    expect(screen.getByText('Submit').closest('button')).toBeDisabled();

    fireEvent.click(screen.getAllByText('Pass')[0]);
    expect(screen.getByText('Submit').closest('button')).not.toBeDisabled();

    fireEvent.click(screen.getByText('Submit'));
    await waitFor(() => expect(onSetStatus).toHaveBeenCalledWith('COMPLETE'));
    expect(mockCreateAssessment).toHaveBeenCalledTimes(1);
    expect(mockCreateAssessment).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'Looks good?', value: true, assessmentKind: 'feedback' }),
    );
  });

  it('does not offer a Decline action on a pending trace (decline removed from the UI)', () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus);
    // Decline is gone; a pending trace's only action is Submit (Move to Todo is
    // terminal-only).
    expect(screen.queryByText('Decline')).not.toBeInTheDocument();
    expect(screen.queryByText('Move to Todo')).not.toBeInTheDocument();
    expect(screen.getByText('Submit')).toBeInTheDocument();
  });

  it('logs a feedback-submitted telemetry event once the review is submitted', async () => {
    const eventCallback = jest.fn();
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus, { eventCallback });

    fireEvent.click(screen.getAllByText('Pass')[0]);
    fireEvent.click(screen.getByText('Submit'));
    await waitFor(() =>
      expect(eventCallback).toHaveBeenCalledWith(
        expect.objectContaining({ componentId: FEEDBACK_SUBMITTED_COMPONENT_ID }),
      ),
    );
  });

  it('logs the feedback-submitted event on the single Pass/Fail auto-submit path too', async () => {
    const eventCallback = jest.fn();
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    // Single Pass/Fail -> picking the answer auto-submits (no explicit Submit click).
    renderFocused([passFailSchema()], onSetStatus, { eventCallback });

    fireEvent.click(screen.getByText('Pass'));
    await waitFor(() =>
      expect(eventCallback).toHaveBeenCalledWith(
        expect.objectContaining({ componentId: FEEDBACK_SUBMITTED_COMPONENT_ID }),
      ),
    );
  });

  it('does not log the feedback-submitted event when the assessment write fails', async () => {
    mockCreateAssessment.mockImplementation(() => Promise.reject(new Error('boom')));
    const eventCallback = jest.fn();
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus, { eventCallback });

    fireEvent.click(screen.getAllByText('Pass')[0]);
    fireEvent.click(screen.getByText('Submit'));
    // The error surfaces as a toast; the event must not fire (it sits after the writes succeed).
    await waitFor(() => expect(errorToastSpy).toHaveBeenCalledTimes(1));
    expect(errorToastSpy.mock.calls[0][0]).toMatch(/could not save your review/i);
    expect(eventCallback).not.toHaveBeenCalledWith(
      expect.objectContaining({ componentId: FEEDBACK_SUBMITTED_COMPONENT_ID }),
    );
  });

  it('enables Submit on load when a prior answer prefills a question', () => {
    // A reopened trace whose answer comes from prefill (no edit) must still count
    // toward answeredCount, exercising the `prefilled` branch of the memo.
    mockPriorAnswersResult = {
      priorAnswers: [{ name: 'Looks good?', kind: 'feedback', value: true, valid: true }],
      isLoading: false,
      isFetching: false,
    };
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus);

    expect(screen.getByText('Submit').closest('button')).not.toBeDisabled();
  });
});

describe('FocusedReview view-only when not assigned', () => {
  beforeEach(() => {
    mockCreateAssessment.mockReset();
    mockCreateAssessment.mockImplementation(() => Promise.resolve());
  });

  it('shows a not-assigned notice and disables submitting when canReview is false', () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    // Two questions so the Submit button renders (not the auto-submit case).
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus, { canReview: false });

    expect(screen.getByText(/not assigned to this queue/i)).toBeInTheDocument();
    expect(screen.getByText('Submit').closest('button')).toBeDisabled();
    fireEvent.click(screen.getAllByText('Pass')[0]);
    expect(mockCreateAssessment).not.toHaveBeenCalled();
    expect(onSetStatus).not.toHaveBeenCalled();
  });

  it('offers an Assign myself action when provided', () => {
    const onAssignSelf = jest.fn();
    renderFocused(
      [passFailSchema()],
      jest.fn(() => Promise.resolve()),
      { canReview: false, onAssignSelf },
    );

    fireEvent.click(screen.getByText('Assign myself'));
    expect(onAssignSelf).toHaveBeenCalled();
  });
});

describe('FocusedReview waits for prior answers to settle before submitting', () => {
  beforeEach(() => {
    mockCreateAssessment.mockReset();
    mockCreateAssessment.mockImplementation(() => Promise.resolve());
  });
  afterEach(() => {
    mockPriorAnswersResult = { priorAnswers: [], isLoading: false, isFetching: false };
  });

  it('disables the Submit button while the prior-answers query is refetching', () => {
    // Supersede ids come from this query; submitting against a stale snapshot
    // could leave two live assessments for one question.
    mockPriorAnswersResult = { priorAnswers: [], isLoading: false, isFetching: true };
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus);

    expect(screen.getByText('Submit').closest('button')).toBeDisabled();
  });

  it('does not auto-submit while the prior-answers query is refetching', () => {
    mockPriorAnswersResult = { priorAnswers: [], isLoading: false, isFetching: true };
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    // Single Pass/Fail -> auto-submit on pick; the fetch guard must suppress it.
    renderFocused([passFailSchema()], onSetStatus);

    fireEvent.click(screen.getByText('Pass'));
    expect(mockCreateAssessment).not.toHaveBeenCalled();
    expect(onSetStatus).not.toHaveBeenCalled();
  });
});

describe('FocusedReview trace content rendering', () => {
  const onSetStatus = jest.fn((_status: string) => Promise.resolve());

  afterEach(() => {
    mockTraceData = [];
  });

  it('renders markdown for plain-text response content', () => {
    mockTraceData = [{ info: { trace_id: 'tr-1', request_preview: null, response_preview: 'Hello **world**' } }];
    renderFocused([passFailSchema()], onSetStatus);
    expect(screen.getByText('world')).toBeInTheDocument();
  });

  it('renders JSON content as a preformatted code block, not markdown', () => {
    const json = JSON.stringify({ key: '# not a heading', list: [1, 2] });
    mockTraceData = [{ info: { trace_id: 'tr-1', request_preview: null, response_preview: json } }];
    renderFocused([passFailSchema()], onSetStatus);

    // JSON keys/values with markdown-like characters must NOT be parsed as
    // headings or bold — they should appear as literal text in a <pre>.
    expect(screen.queryByRole('heading')).not.toBeInTheDocument();
    expect(screen.getByText(/# not a heading/)).toBeInTheDocument();
  });

  it('falls back to truncated plain text for content exceeding the size limit', () => {
    const huge = 'x'.repeat(1_100_000);
    mockTraceData = [{ info: { trace_id: 'tr-1', request_preview: null, response_preview: huge } }];
    renderFocused([passFailSchema()], onSetStatus);

    // The rendered text should be capped at 10 000 chars (the slice limit).
    const preElement = screen.getByText((_, el) => el?.tagName === 'PRE' && (el.textContent?.length ?? 0) <= 10_001);
    expect(preElement).toBeInTheDocument();
  });

  it('renders input preview as a right-aligned chat bubble', () => {
    mockTraceData = [{ info: { trace_id: 'tr-1', request_preview: 'user prompt', response_preview: null } }];
    renderFocused([passFailSchema()], onSetStatus);
    expect(screen.getByText('user prompt')).toBeInTheDocument();
  });
});

describe('FocusedReview edit-in-place on a completed trace', () => {
  let toastSpy: jest.SpiedFunction<typeof Utils.displayGlobalInfoNotification>;
  let errorToastSpy: jest.SpiedFunction<typeof Utils.displayGlobalErrorNotification>;
  beforeEach(() => {
    toastSpy = jest.spyOn(Utils, 'displayGlobalInfoNotification').mockImplementation(() => {});
    errorToastSpy = jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
    mockCreateAssessment.mockReset();
    mockCreateAssessment.mockImplementation(() => Promise.resolve());
    // The completed trace prefills its prior answer (with its assessment id, so a
    // re-save supersedes it rather than duplicating), so the form is pre-populated.
    mockPriorAnswersResult = {
      priorAnswers: [{ name: 'Looks good?', kind: 'feedback', value: true, valid: true, assessmentId: 'a-prior' }],
      isLoading: false,
      isFetching: false,
    };
  });
  afterEach(() => {
    toastSpy.mockRestore();
    errorToastSpy.mockRestore();
    mockPriorAnswersResult = { priorAnswers: [], isLoading: false, isFetching: false };
  });

  it('keeps a completed trace editable and re-saves (superseding the prior) without changing status or navigating', async () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    const onSelect = jest.fn();
    const onBack = jest.fn();
    // Two questions so the explicit primary button renders (not the auto-submit case).
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus, {
      item: completeItem,
      onSelect,
      onBack,
    });

    // The primary action re-saves edits rather than completing afresh, and the
    // inputs stay editable (not disabled like a declined trace). With no edit yet
    // it's a no-op, so the button is disabled until the reviewer actually changes
    // something — a passive prefill must not re-supersede identical assessments.
    expect(screen.queryByText('Submit')).not.toBeInTheDocument();
    expect(screen.getByText('Save changes').closest('button')).toBeDisabled();

    // Edit the second question (the editable inputs are what "stays editable" means).
    fireEvent.click(screen.getAllByText('Pass')[1]);
    expect(screen.getByText('Save changes').closest('button')).not.toBeDisabled();
    fireEvent.click(screen.getByText('Save changes'));

    await waitFor(() => expect(mockCreateAssessment).toHaveBeenCalled());
    // The prior answer is superseded via `overrides` (not duplicated), with the
    // right value/source — the core edit-in-place contract.
    expect(mockCreateAssessment).toHaveBeenCalledWith(
      expect.objectContaining({
        traceId: 'tr-1',
        name: 'Looks good?',
        value: true,
        sourceId: 'tester',
        overrides: 'a-prior',
      }),
    );
    // Re-saving rewrites the answers but must NOT flip the status or move the
    // reviewer off the trace — it stays COMPLETE / off the to-do list.
    expect(onSetStatus).not.toHaveBeenCalled();
    expect(onSelect).not.toHaveBeenCalled();
    expect(onBack).not.toHaveBeenCalled();
    // The save is confirmed by a global toast (not an inline alert that would
    // reflow the action buttons).
    await waitFor(() => expect(toastSpy).toHaveBeenCalledWith('Changes saved'));
  });

  it('shows the save error and suppresses the saved confirmation when an in-place re-save fails', async () => {
    mockCreateAssessment.mockImplementation(() => Promise.reject(new Error('boom')));
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus, { item: completeItem });

    fireEvent.click(screen.getAllByText('Pass')[1]);
    fireEvent.click(screen.getByText('Save changes'));

    await waitFor(() => expect(errorToastSpy).toHaveBeenCalledTimes(1));
    expect(errorToastSpy.mock.calls[0][0]).toMatch(/could not save your review/i);
    expect(toastSpy).not.toHaveBeenCalledWith('Changes saved');
    expect(onSetStatus).not.toHaveBeenCalled();
  });

  it('still offers Move to Todo as the explicit way to send a completed trace back to the to-do list', async () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus, { item: completeItem });

    fireEvent.click(screen.getByText('Move to Todo'));
    await waitFor(() => expect(onSetStatus).toHaveBeenCalledWith('PENDING'));
  });

  it('locks a declined trace and offers only Move to Todo (no save/submit)', () => {
    const onSetStatus = jest.fn((_status: string) => Promise.resolve());
    renderFocused([passFailSchema(), passFailSchema('s2', 'Also good?')], onSetStatus, { item: declinedItem });

    expect(screen.getByText('Move to Todo')).toBeInTheDocument();
    expect(screen.queryByText('Save changes')).not.toBeInTheDocument();
    expect(screen.queryByText('Submit')).not.toBeInTheDocument();
    // The lock is the inputs being disabled, not just the missing button.
    expect(screen.getAllByRole('radio', { name: 'Pass' })[0]).toBeDisabled();
  });
});
