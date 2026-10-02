import { describe, test, expect } from '@jest/globals';
import {
  createOpenColumnSelectorViewStatePatch,
  mapSelectedRunUuids,
  shouldShowRunsBulkActions,
} from './experimentPage.view-state-utils';

describe('experimentPage.view-state-utils', () => {
  test('maps selected run UUIDs to a lookup map', () => {
    expect(mapSelectedRunUuids(['run-1', 'run-2'])).toEqual({
      'run-1': true,
      'run-2': true,
    });
  });

  test('shows bulk actions only when runs are selected and column selector is hidden', () => {
    const selectedRuns = mapSelectedRunUuids(['run-1']);

    expect(shouldShowRunsBulkActions(selectedRuns, false)).toBe(true);
    expect(shouldShowRunsBulkActions(selectedRuns, true)).toBe(false);
    expect(shouldShowRunsBulkActions({}, false)).toBe(false);
  });

  test('creates a patch that opens the column selector', () => {
    expect(createOpenColumnSelectorViewStatePatch()).toEqual({
      columnSelectorVisible: true,
    });
  });
});
