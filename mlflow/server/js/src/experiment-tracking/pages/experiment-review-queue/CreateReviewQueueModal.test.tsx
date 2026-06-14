import { describe, jest, it, expect, beforeEach } from '@jest/globals';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { CreateReviewQueueModal } from './CreateReviewQueueModal';
import { MAX_ASSIGNED_USERS } from './ReviewerChecklistCombobox';

// Stub the question picker so a question can be selected (required to submit).
jest.mock('./QuestionChecklistCombobox', () => ({
  QuestionChecklistCombobox: ({ onToggle }: { onToggle: (id: string) => void }) => (
    <button type="button" onClick={() => onToggle('s1')}>
      toggle-question
    </button>
  ),
}));

// Keep the real MAX_ASSIGNED_USERS export, stub the component to capture props and
// expose member toggles (creator name included, to exercise the dedupe).
let reviewerProps: { maxSelected?: number } = {};
jest.mock('./ReviewerChecklistCombobox', () => ({
  ...jest.requireActual<typeof import('./ReviewerChecklistCombobox')>('./ReviewerChecklistCombobox'),
  ReviewerChecklistCombobox: (props: { onToggle: (u: string) => void; maxSelected?: number }) => {
    reviewerProps = props;
    return (
      <div>
        <button type="button" onClick={() => props.onToggle('bob')}>
          toggle-bob
        </button>
        <button type="button" onClick={() => props.onToggle('creator')}>
          toggle-creator
        </button>
      </div>
    );
  },
}));

jest.mock('../../components/label-schemas', () => ({
  useListLabelSchemasQuery: () => ({
    labelSchemas: [{ schema_id: 's1', name: 'Q1', type: 'FEEDBACK', input: { text: {} } }],
    isLoading: false,
  }),
  LabelSchemaInputRenderer: () => null,
  LabelSchemaFormModal: () => null,
}));
jest.mock('../../../account/hooks', () => ({
  useIsAuthAvailable: () => true,
}));
jest.mock('./hooks/useReviewer', () => ({
  useReviewer: () => 'creator',
  useIsReviewerResolved: () => true,
}));
jest.mock('./hooks/useAssignableUsersQuery', () => ({
  useAssignableUsersQuery: () => ({ users: [], isLoading: false }),
}));
const mockCreate = jest.fn<(...args: any[]) => Promise<any>>();
jest.mock('./hooks/useCreateReviewQueueMutation', () => ({
  useCreateReviewQueueMutation: () => ({ createReviewQueueAsync: mockCreate, isCreatingQueue: false, error: null }),
}));

const renderModal = () => {
  const onClose = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <CreateReviewQueueModal experimentId="exp-1" onClose={onClose} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onClose };
};

const fillAndSubmit = () => {
  fireEvent.change(screen.getByPlaceholderText(/hallucination review/i), { target: { value: 'My Queue' } });
  fireEvent.click(screen.getByText('toggle-question'));
  fireEvent.click(screen.getByText('Create'));
};

describe('CreateReviewQueueModal', () => {
  beforeEach(() => {
    reviewerProps = {};
    mockCreate.mockReset();
    mockCreate.mockImplementation(() => Promise.resolve({ review_queue: { queue_id: 'rq-new' } }));
  });

  it('caps reviewers at one below the assigned-user limit (the creator takes a slot)', () => {
    renderModal();
    expect(reviewerProps.maxSelected).toBe(MAX_ASSIGNED_USERS - 1);
  });

  it('always includes the creator in the queue users, deduped if also picked as a reviewer', async () => {
    renderModal();
    fireEvent.change(screen.getByPlaceholderText(/hallucination review/i), { target: { value: 'My Queue' } });
    fireEvent.click(screen.getByText('toggle-question'));
    fireEvent.click(screen.getByText('toggle-bob'));
    fireEvent.click(screen.getByText('toggle-creator'));
    fireEvent.click(screen.getByText('Create'));
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    expect(mockCreate).toHaveBeenCalledWith(
      expect.objectContaining({ created_by: 'creator', users: ['creator', 'bob'], schema_ids: ['s1'] }),
    );
  });

  it('keeps the modal open when creation fails (e.g. a duplicate name) instead of crashing', async () => {
    mockCreate.mockImplementation(() => Promise.reject(new Error('Review queue with name already exists')));
    const { onClose } = renderModal();
    fillAndSubmit();
    await waitFor(() => expect(mockCreate).toHaveBeenCalledTimes(1));
    // The rejection is swallowed (surfaced via the error Alert), so the modal stays open.
    expect(onClose).not.toHaveBeenCalled();
  });
});
