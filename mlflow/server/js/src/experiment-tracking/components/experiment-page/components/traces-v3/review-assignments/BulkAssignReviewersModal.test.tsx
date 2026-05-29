import { describe, test, expect, jest, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import { renderWithDesignSystem, screen, waitFor } from '../../../../../../common/utils/TestUtils.react18';
import { BulkAssignReviewersModal, parseReviewers } from './BulkAssignReviewersModal';
import { useBulkAssignReviewers } from './useBulkAssignReviewers';
import { useCurrentUserQuery } from '../../../../../../account/hooks';

jest.mock('./useBulkAssignReviewers');
jest.mock('../../../../../../account/hooks');

const mockBulkAssign = jest.fn();

const setUser = (username: string | undefined) => {
  jest.mocked(useCurrentUserQuery).mockReturnValue({
    data: username ? { user: { username } } : undefined,
  } as any);
};

describe('parseReviewers', () => {
  test.each([
    ['a@x.com\nb@x.com', ['a@x.com', 'b@x.com']],
    ['a@x.com, b@x.com', ['a@x.com', 'b@x.com']],
    ['  a@x.com \n\n a@x.com ', ['a@x.com']],
    ['', []],
    ['\n,  ,\n', []],
  ])('parses %p', (input, expected) => {
    expect(parseReviewers(input)).toEqual(expected);
  });
});

describe('BulkAssignReviewersModal', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setUser('kris@example.com');
    jest.mocked(useBulkAssignReviewers).mockReturnValue({
      mutate: mockBulkAssign,
      isLoading: false,
      error: null,
      reset: jest.fn(),
    } as any);
  });

  const renderModal = (props?: Partial<React.ComponentProps<typeof BulkAssignReviewersModal>>) =>
    renderWithDesignSystem(
      <BulkAssignReviewersModal experimentId="1" traceIds={['tr-1', 'tr-2']} onClose={jest.fn()} {...props} />,
    );

  test('disables Assign until reviewers are entered', async () => {
    renderModal();
    const assignButton = screen.getByText('Assign (0)').closest('button');
    expect(assignButton).toBeDisabled();
  });

  test('assigns parsed reviewers across all selected traces', async () => {
    renderModal();
    await userEvent.type(screen.getByLabelText('Reviewers'), 'sme1@example.com, sme2@example.com');

    await userEvent.click(screen.getByText('Assign (2)'));

    await waitFor(() => expect(mockBulkAssign).toHaveBeenCalledTimes(1));
    expect(mockBulkAssign).toHaveBeenCalledWith(
      {
        experimentId: '1',
        traceIds: ['tr-1', 'tr-2'],
        reviewers: ['sme1@example.com', 'sme2@example.com'],
        assigner: 'kris@example.com',
      },
      expect.anything(),
    );
  });

  test('closes on success when nothing failed', async () => {
    mockBulkAssign.mockImplementation((_params: any, { onSuccess }: any) =>
      onSuccess({ created: [{}], existing: [], failed: [] }),
    );
    const onClose = jest.fn();
    renderModal({ onClose });
    await userEvent.type(screen.getByLabelText('Reviewers'), 'sme1@example.com');

    await userEvent.click(screen.getByText('Assign (1)'));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
  });

  test('keeps the modal open and warns when some assignments failed', async () => {
    mockBulkAssign.mockImplementation((_params: any, { onSuccess }: any) =>
      onSuccess({ created: [], existing: [], failed: [{ target_id: 'tr-1', reviewer: 'x', error_message: 'boom' }] }),
    );
    const onClose = jest.fn();
    renderModal({ onClose });
    await userEvent.type(screen.getByLabelText('Reviewers'), 'sme1@example.com');

    await userEvent.click(screen.getByText('Assign (1)'));

    expect(await screen.findByText('1 assignment could not be created.')).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });

  test('clears the partial-failure warning on a subsequent successful retry', async () => {
    mockBulkAssign
      .mockImplementationOnce((_params: any, { onSuccess }: any) =>
        onSuccess({ created: [], existing: [], failed: [{ target_id: 'tr-1', reviewer: 'x', error_message: 'boom' }] }),
      )
      .mockImplementationOnce((_params: any, { onSuccess }: any) =>
        onSuccess({ created: [{}], existing: [], failed: [] }),
      );
    const onClose = jest.fn();
    renderModal({ onClose });
    await userEvent.type(screen.getByLabelText('Reviewers'), 'sme1@example.com');

    await userEvent.click(screen.getByText('Assign (1)'));
    expect(await screen.findByText('1 assignment could not be created.')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Assign (1)'));
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(screen.queryByText('1 assignment could not be created.')).not.toBeInTheDocument();
  });

  test('renders the error banner when the mutation fails', async () => {
    jest.mocked(useBulkAssignReviewers).mockReturnValue({
      mutate: mockBulkAssign,
      isLoading: false,
      error: new Error('Network boom'),
      reset: jest.fn(),
    } as any);
    renderModal();

    expect(screen.getByText('Network boom')).toBeInTheDocument();
  });

  test('disables Assign when no current user is available', async () => {
    setUser(undefined);
    renderModal();
    await userEvent.type(screen.getByLabelText('Reviewers'), 'sme1@example.com');

    expect(screen.getByText('Assign (1)').closest('button')).toBeDisabled();
    expect(mockBulkAssign).not.toHaveBeenCalled();
  });
});
