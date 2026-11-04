import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AssessmentEditForm } from './AssessmentEditForm';
import { useUserIdentity } from '../../global-settings/getUser';
import type { Assessment } from '../ModelTrace.types';
import { MOCK_ASSESSMENT, MOCK_EXPECTATION } from '../ModelTraceExplorer.test-utils';

jest.mock('../../global-settings/getUser', () => ({
  ...jest.requireActual<typeof import('../../global-settings/getUser')>('../../global-settings/getUser'),
  getUser: () => 'User',
  useUserIdentity: jest.fn(),
}));

const mockUpdateAssessmentMutation = jest.fn();
const mockOverrideAssessmentMutation = jest.fn();

jest.mock('../hooks/useUpdateAssessment', () => ({
  useUpdateAssessment: () => ({ updateAssessmentMutation: mockUpdateAssessmentMutation, isLoading: false }),
}));

jest.mock('../hooks/useOverrideAssessment', () => ({
  useOverrideAssessment: () => ({ overrideAssessmentMutation: mockOverrideAssessmentMutation, isLoading: false }),
}));

const fromAlice = (assessment: Assessment) =>
  ({ ...assessment, source: { source_type: 'HUMAN', source_id: 'alice' } }) as Assessment;

const renderEditForm = (assessment: Assessment) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AssessmentEditForm assessment={assessment} onCancel={jest.fn()} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

const saveButton = () => screen.getByRole('button', { name: 'Save' });

describe('AssessmentEditForm', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('updates an expectation in place for its owner', async () => {
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'alice', isResolved: true });

    renderEditForm(fromAlice(MOCK_EXPECTATION));
    await userEvent.click(saveButton());

    expect(mockUpdateAssessmentMutation).toHaveBeenCalledTimes(1);
  });

  it('does not save an expectation after the signed-in user changes to someone else', async () => {
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'bob', isResolved: true });

    renderEditForm(fromAlice(MOCK_EXPECTATION));

    expect(saveButton()).toBeDisabled();
    await userEvent.click(saveButton());
    expect(mockUpdateAssessmentMutation).not.toHaveBeenCalled();
  });

  it('saves feedback from another user as an override', async () => {
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'bob', isResolved: true });

    renderEditForm(fromAlice(MOCK_ASSESSMENT));
    await userEvent.click(saveButton());

    expect(mockUpdateAssessmentMutation).not.toHaveBeenCalled();
    expect(mockOverrideAssessmentMutation).toHaveBeenCalledTimes(1);
  });

  it('closes the form when the signed-in user changes', () => {
    const onCancel = jest.fn();
    const editForm = () => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <AssessmentEditForm assessment={fromAlice(MOCK_ASSESSMENT)} onCancel={onCancel} />
        </DesignSystemProvider>
      </IntlProvider>
    );
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'alice', isResolved: true });
    const { rerender } = render(editForm());
    expect(onCancel).not.toHaveBeenCalled();

    jest.mocked(useUserIdentity).mockReturnValue({ user: 'bob', isResolved: true });
    rerender(editForm());

    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
