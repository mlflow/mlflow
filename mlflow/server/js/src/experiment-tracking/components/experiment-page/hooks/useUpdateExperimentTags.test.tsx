import { describe, it, expect } from '@jest/globals';

import { describeTagSaveOutcome } from './useUpdateExperimentTags';
import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';

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
