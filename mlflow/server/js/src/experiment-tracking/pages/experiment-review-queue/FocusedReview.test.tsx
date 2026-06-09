import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { FocusedReview } from './FocusedReview';
import type { LabelSchema } from '../../components/label-schemas';
import type { ReviewQueueItem } from './types';

jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  useGetTracesById: () => ({ data: [], isLoading: false }),
  ModelTraceExplorer: () => null,
}));

const mockCreateAssessment = jest.fn();
jest.mock('./hooks/useCreateReviewAssessmentMutation', () => ({
  useCreateReviewAssessmentMutation: () => ({
    createReviewAssessmentAsync: mockCreateAssessment,
    isCreatingAssessment: false,
  }),
}));
jest.mock('./hooks/useTraceAssessmentsQuery', () => ({
  useTraceAssessmentsQuery: () => ({ priorAnswers: [], isLoading: false }),
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
  opts: { canReview?: boolean; onAssignSelf?: () => void } = {},
) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <FocusedReview
          item={pendingItem}
          items={[pendingItem]}
          schemas={schemas}
          completedBy="tester"
          isSettingStatus={false}
          canReview={opts.canReview ?? true}
          onAssignSelf={opts.onAssignSelf}
          onBack={jest.fn()}
          onSelect={jest.fn()}
          onSetStatus={onSetStatus}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

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
