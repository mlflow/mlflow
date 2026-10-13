import { describe, beforeEach, afterEach, jest, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import Utils from '../../../common/utils/Utils';
import { LabelSchemaFormModal } from './LabelSchemaFormModal';
import type { LabelSchema } from './types';

const mockUpdateAsync = jest.fn();
const mockCreateAsync = jest.fn();
jest.mock('./hooks/useCreateLabelSchemaMutation', () => ({
  useCreateLabelSchemaMutation: () => ({
    createLabelSchemaAsync: mockCreateAsync,
    isCreating: false,
    reset: jest.fn(),
  }),
}));
jest.mock('./hooks/useUpdateLabelSchemaMutation', () => ({
  useUpdateLabelSchemaMutation: () => ({
    updateLabelSchemaAsync: mockUpdateAsync,
    isUpdating: false,
    reset: jest.fn(),
  }),
}));

// The form renderer, preview, and resizable pane are heavy and unrelated to the
// submit/error path under test; stub them so the modal renders with just its Save
// button. The form value comes from `editingSchema` via react-hook-form defaults.
jest.mock('./LabelSchemaFormRenderer', () => ({ LabelSchemaFormRenderer: () => null }));
jest.mock('./LabelSchemaPreview', () => ({ LabelSchemaPreview: () => null }));
jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  ModelTraceExplorerResizablePane: () => null,
}));

// The form fields are stubbed out (above), so we can't type valid values; bypass
// validation so submitting reaches the mutation in both create and edit mode. The
// real validator is exercised by its own callers, not this submit/error test.
jest.mock('./labelSchemaFormUtils', () => ({
  ...jest.requireActual<typeof import('./labelSchemaFormUtils')>('./labelSchemaFormUtils'),
  validateLabelSchemaForm: () => ({}),
}));

// A valid free-text FEEDBACK schema: it passes `validateLabelSchemaForm` with no
// required sub-fields, so submitting reaches the update mutation.
const editingSchema: LabelSchema = {
  schema_id: 's1',
  experiment_id: 'exp-1',
  name: 'Q1',
  type: 'FEEDBACK',
  input: { text: {} },
};

const mockOnClose = jest.fn();
// `editingSchema = null` opens the modal in create mode; otherwise it edits.
const renderModal = (editing: LabelSchema | null = editingSchema) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <LabelSchemaFormModal experimentId="exp-1" editingSchema={editing} visible onClose={mockOnClose} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('LabelSchemaFormModal save', () => {
  beforeEach(() => {
    mockOnClose.mockReset();
    mockUpdateAsync.mockReset().mockImplementation(() => Promise.resolve());
    mockCreateAsync.mockReset().mockImplementation(() => Promise.resolve());
    jest.spyOn(Utils, 'displayGlobalErrorNotification').mockImplementation(() => {});
  });

  // Restore the Utils spy so a fresh `beforeEach` spy doesn't wrap an already-spied
  // method and stack layers across tests.
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('surfaces a toast and keeps the modal open when saving fails', async () => {
    mockUpdateAsync.mockImplementation(() => Promise.reject(new Error('boom')));
    renderModal();
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(mockUpdateAsync).toHaveBeenCalledTimes(1);
    // Edit mode must route through update, never create.
    expect(mockCreateAsync).not.toHaveBeenCalled();
    await waitFor(() => expect(Utils.displayGlobalErrorNotification).toHaveBeenCalledTimes(1));
    expect(Utils.displayGlobalErrorNotification).toHaveBeenCalledWith(expect.stringContaining('boom'));
    // The modal stays open (its title is still shown, and it isn't closed) so the
    // user can retry.
    expect(screen.getByText('Edit question')).toBeInTheDocument();
    expect(mockOnClose).not.toHaveBeenCalled();
  });

  it('closes the modal without a toast on a successful save', async () => {
    renderModal();
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(mockUpdateAsync).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(mockOnClose).toHaveBeenCalledTimes(1));
    expect(Utils.displayGlobalErrorNotification).not.toHaveBeenCalled();
  });

  it('surfaces a toast and keeps the modal open when creating fails', async () => {
    mockCreateAsync.mockImplementation(() => Promise.reject(new Error('boom')));
    renderModal(null);
    await userEvent.click(screen.getByRole('button', { name: 'Create' }));

    expect(mockCreateAsync).toHaveBeenCalledTimes(1);
    // Create mode must route through create, never update.
    expect(mockUpdateAsync).not.toHaveBeenCalled();
    await waitFor(() => expect(Utils.displayGlobalErrorNotification).toHaveBeenCalledTimes(1));
    expect(Utils.displayGlobalErrorNotification).toHaveBeenCalledWith(expect.stringContaining('boom'));
    // The modal stays open (its create-mode title is still shown) so the user can retry.
    expect(screen.getByText('Add question')).toBeInTheDocument();
    expect(mockOnClose).not.toHaveBeenCalled();
  });
});
