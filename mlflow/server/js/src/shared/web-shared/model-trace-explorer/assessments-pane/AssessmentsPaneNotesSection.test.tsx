import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AssessmentsPaneNotesSection, NOTES_ASSESSMENT_NAME } from './AssessmentsPaneNotesSection';
import { useUserIdentity } from '../../global-settings/getUser';
import type { FeedbackAssessment } from '../ModelTrace.types';

jest.mock('../../global-settings/getUser', () => ({
  getUser: () => 'User',
  useUserIdentity: jest.fn(),
}));

jest.mock('../hooks/useCreateAssessment', () => ({
  useCreateAssessment: () => ({ createAssessmentMutation: jest.fn(), isLoading: false }),
}));

jest.mock('../hooks/useUpdateAssessment', () => ({
  useUpdateAssessment: () => ({ updateAssessmentMutation: jest.fn(), isLoading: false }),
}));

const note = (sourceId: string, value: string) =>
  ({
    assessment_id: `note-${sourceId}`,
    assessment_name: NOTES_ASSESSMENT_NAME,
    trace_id: 'tr-test',
    source: { source_type: 'HUMAN', source_id: sourceId },
    create_time: '2025-04-19T09:04:07.875Z',
    last_update_time: '2025-04-19T09:04:07.875Z',
    feedback: { value },
  }) as FeedbackAssessment;

const renderNotes = (feedbacks: FeedbackAssessment[]) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AssessmentsPaneNotesSection traceId="tr-test" feedbacks={feedbacks} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('AssessmentsPaneNotesSection', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('shows legacy placeholder notes read-only next to the notes of the signed-in user', () => {
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'alice', isResolved: true });

    renderNotes([note('User', 'shared note'), note('alice', 'my note')]);

    expect(screen.getByText('Earlier notes saved without a username')).toBeInTheDocument();
    expect(screen.getByText('shared note')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveValue('my note');
  });

  it('edits placeholder notes directly when auth is unavailable', () => {
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'User', isResolved: true });

    renderNotes([note('User', 'shared note')]);

    expect(screen.queryByText('Earlier notes saved without a username')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveValue('shared note');
  });

  it('disables the notes input until the current user is known', () => {
    jest.mocked(useUserIdentity).mockReturnValue({ user: '', isResolved: false });

    renderNotes([]);

    expect(screen.getByRole('textbox')).toBeDisabled();
  });
});
