import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { AssessmentActionsOverflowMenu } from './AssessmentActionsOverflowMenu';
import type { Assessment } from '../ModelTrace.types';
import { MOCK_ASSESSMENT, MOCK_EXPECTATION } from '../ModelTraceExplorer.test-utils';

jest.mock('../../global-settings/getUser', () => ({
  getUser: () => 'User',
  useUserIdentity: () => ({ user: 'alice', isResolved: true }),
}));

const humanSource = (sourceId: string) => ({ source: { source_type: 'HUMAN', source_id: sourceId } }) as const;

const openMenu = async (assessment: Assessment) => {
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <AssessmentActionsOverflowMenu
          assessment={assessment}
          setIsEditing={jest.fn()}
          setShowDeleteModal={jest.fn()}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  await userEvent.click(screen.getByRole('button'));
  await screen.findByRole('menuitem', { name: 'Delete' });
};

describe('AssessmentActionsOverflowMenu', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('offers Edit for an assessment from the signed-in user', async () => {
    await openMenu({ ...MOCK_EXPECTATION, ...humanSource('alice') } as Assessment);

    expect(screen.getByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
  });

  it('offers Edit for legacy feedback saved under the placeholder', async () => {
    await openMenu({ ...MOCK_ASSESSMENT, ...humanSource('User') } as Assessment);

    expect(screen.getByRole('menuitem', { name: 'Edit' })).toBeInTheDocument();
  });

  it('does not offer Edit for a placeholder expectation, which would be updated in place', async () => {
    await openMenu({ ...MOCK_EXPECTATION, ...humanSource('User') } as Assessment);

    expect(screen.queryByRole('menuitem', { name: 'Edit' })).not.toBeInTheDocument();
  });

  it('does not offer Edit for human feedback from a different user', async () => {
    await openMenu({ ...MOCK_ASSESSMENT, ...humanSource('bob') } as Assessment);

    expect(screen.queryByRole('menuitem', { name: 'Edit' })).not.toBeInTheDocument();
  });
});
