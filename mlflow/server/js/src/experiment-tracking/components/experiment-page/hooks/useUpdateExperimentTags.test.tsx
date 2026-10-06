import { jest, describe, it, expect, beforeEach } from '@jest/globals';
import React from 'react';
import { renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { describeTagSaveOutcome, useUpdateExperimentTags } from './useUpdateExperimentTags';
import { useEditKeyValueTagsModal } from '../../../../common/hooks/useEditKeyValueTagsModal';
import { MlflowService } from '../../../sdk/MlflowService';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';

jest.mock('../../../../common/hooks/useEditKeyValueTagsModal', () => ({
  useEditKeyValueTagsModal: jest.fn(),
}));

const permissionDenied = () =>
  new ErrorWrapper(JSON.stringify({ error_code: 'PERMISSION_DENIED', message: 'Permission denied' }), 403);

describe('describeTagSaveOutcome', () => {
  // A tag save is one request per tag, so it is not atomic. `Promise.all` rejected on the
  // first failure and reported the save as failed while the permitted writes landed: the
  // user saw "Permission denied", reopened the modal, and found half their edits applied.
  // These cases pin that the message now names both halves.

  it('names what was saved as well as what was refused', () => {
    const message = describeTagSaveOutcome([{ key: 'bob', reason: permissionDenied() }], ['carol']);
    expect(message).toContain('Could not save bob');
    expect(message).toContain('Permission denied');
    expect(message).toContain('These changes were saved: carol');
  });

  it('says plainly when nothing landed, so the user is not sent looking for a partial write', () => {
    const message = describeTagSaveOutcome([{ key: 'bob', reason: permissionDenied() }], []);
    expect(message).toContain('No changes were made');
    expect(message).not.toContain('were saved:');
  });

  it('states one shared reason once rather than per key', () => {
    // The common case: a single condition refuses several keys for the same reason.
    const message = describeTagSaveOutcome(
      [
        { key: 'bob', reason: permissionDenied() },
        { key: 'dave', reason: permissionDenied() },
      ],
      ['carol'],
    );
    expect(message).toContain('Could not save bob, dave: Permission denied');
    expect(message.match(/Permission denied/g)).toHaveLength(1);
  });

  it('attributes per key when the reasons differ', () => {
    const message = describeTagSaveOutcome(
      [
        { key: 'bob', reason: permissionDenied() },
        { key: 'dave', reason: new Error('Tag value too long') },
      ],
      [],
    );
    expect(message).toContain('bob: Permission denied');
    expect(message).toContain('dave: Tag value too long');
  });

  it('does not double the period when the server reason is a full sentence', () => {
    // The server's condition denial ends in a period and gets composed into a longer
    // sentence, which read "...not permitted.. These changes were saved: carol."
    const wrapped = new ErrorWrapper(
      JSON.stringify({ error_code: 'PERMISSION_DENIED', message: 'Permission denied by a condition.' }),
      403,
    );
    const message = describeTagSaveOutcome([{ key: 'bob', reason: wrapped }], ['carol']);
    expect(message).not.toContain('..');
    expect(message).toContain('by a condition. These changes were saved: carol.');
  });

  it('falls back to a usable phrase when a rejection carries no message', () => {
    const message = describeTagSaveOutcome([{ key: 'bob', reason: undefined }], []);
    expect(message).toContain('Could not save bob');
    expect(message).toContain('the request was rejected');
  });
});

describe('useUpdateExperimentTags partial-save refresh', () => {
  // A tag save is one request per tag, so a partial save really writes some tags and then
  // rejects. The reject reaches the modal's error path, which keeps the modal open -- and
  // the caller's refresh (`invalidateExperimentList`) hung off the success path only, so
  // the list went on showing the pre-save tags while the server already had the new ones.

  let queryClient: QueryClient;
  let saveTags: ((entity: any, existing: any[], next: any[]) => Promise<any>) | undefined;

  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  const denied = () =>
    new ErrorWrapper(JSON.stringify({ error_code: 'PERMISSION_DENIED', message: 'Permission denied' }), 403);

  beforeEach(() => {
    jest.clearAllMocks();
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    saveTags = undefined;
    // Capture the handler the hook hands the modal, so the save can be driven directly
    // rather than through the modal's form.
    jest.mocked(useEditKeyValueTagsModal).mockImplementation((options: any) => {
      saveTags = options.saveTagsHandler;
      return { EditTagsModal: null, showEditTagsModal: jest.fn(), isLoading: false } as any;
    });
  });

  const renderWithRefresh = () => {
    const onSuccess = jest.fn();
    renderHook(() => useUpdateExperimentTags({ onSuccess }), { wrapper });
    return onSuccess;
  };

  const twoNewTags = [
    { key: 'kept', value: '1' },
    { key: 'refused', value: '2' },
  ];

  it('refreshes the experiment list when only some tags were saved', async () => {
    jest
      .spyOn(MlflowService, 'setExperimentTag')
      .mockResolvedValueOnce({} as any)
      .mockRejectedValueOnce(denied());

    const onSuccess = renderWithRefresh();

    await expect(saveTags!({ experimentId: '1' }, [], twoNewTags)).rejects.toThrow(/Could not save/);
    // Reported as failed -- and the refresh must still run, because `kept` really landed.
    expect(onSuccess).toHaveBeenCalled();
  });

  it('does not refresh when nothing was saved', async () => {
    jest.spyOn(MlflowService, 'setExperimentTag').mockRejectedValue(denied());

    const onSuccess = renderWithRefresh();

    await expect(saveTags!({ experimentId: '1' }, [], twoNewTags)).rejects.toThrow(/Could not save/);
    // Nothing changed server-side, so there is nothing to refresh.
    expect(onSuccess).not.toHaveBeenCalled();
  });

  it('refreshes on a complete success', async () => {
    jest.spyOn(MlflowService, 'setExperimentTag').mockResolvedValue({} as any);

    const onSuccess = renderWithRefresh();

    await expect(saveTags!({ experimentId: '1' }, [], twoNewTags)).resolves.toBeUndefined();
    expect(onSuccess).toHaveBeenCalled();
  });
});
