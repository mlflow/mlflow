import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { DatasetRecordDetailFooter } from './DatasetRecordDetailFooter';
import type { SaveStatus } from './DatasetRecordDetailFooter';

jest.mock('../../../../assistant/useFloatingObstruction', () => ({
  useRegisterFloatingBottomObstruction: jest.fn(),
}));

const renderFooter = (status: SaveStatus, isDirty: boolean) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <DatasetRecordDetailFooter status={status} isDirty={isDirty} onSave={jest.fn()} onDiscard={jest.fn()} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('DatasetRecordDetailFooter', () => {
  test('disables Cancel when the editors match their baseline even if status is dirty', () => {
    // Create mode seeds the editors with defaults, so status is 'dirty' before any edit.
    renderFooter('dirty', false);
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Save' })).toBeEnabled();
  });

  test.each<SaveStatus>(['dirty', 'invalid', 'empty-inputs', 'error'])(
    'enables Cancel when there are edits to revert (status=%s)',
    (status) => {
      renderFooter(status, true);
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeEnabled();
    },
  );

  test('disables Cancel while saving', () => {
    renderFooter('saving', true);
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
  });
});
