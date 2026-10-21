import { describe, expect, jest, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import { renderWithDesignSystem, screen, within } from '../../../common/utils/TestUtils.react18';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { ExperimentIssuesDetectionProgress } from './ExperimentIssuesDetectionProgress';

const renderPanel = (onCancelDetection?: () => Promise<void>) =>
  renderWithDesignSystem(
    <MemoryRouter>
      <ExperimentIssuesDetectionProgress experimentId="1" onCancelDetection={onCancelDetection} />
    </MemoryRouter>,
  );

describe('ExperimentIssuesDetectionProgress', () => {
  test('announces the run in progress to assistive technology', () => {
    renderPanel(jest.fn<() => Promise<void>>());

    const status = screen.getByRole('status');
    expect(status).toHaveAttribute('aria-live', 'polite');
    expect(status).toHaveTextContent('Detecting issues...');
    // The run outlives the page, which is the thing a waiting user needs told.
    expect(status).toHaveTextContent('you can leave this page and return later');
  });

  test('keeps a spinner on the cancel button while the run goes', () => {
    renderPanel(jest.fn<() => Promise<void>>());

    // The spinner reports the run, not the cancel request, so the button spins and still works.
    const cancel = screen.getByRole('button', { name: 'Cancel detection' });
    expect(cancel).toBeEnabled();
    expect(cancel.querySelector('[role="img"], svg')).not.toBeNull();
  });

  test('cancels only after the dialog is confirmed', async () => {
    const onCancelDetection = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    renderPanel(onCancelDetection);

    await userEvent.click(screen.getByRole('button', { name: 'Cancel detection' }));
    expect(onCancelDetection).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Keep running' }));
    expect(onCancelDetection).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Cancel detection' }));
    const dialog = screen.getByRole('dialog');
    // The dialog says what cancelling costs; when to cancel is already on the page.
    expect(dialog).toHaveTextContent('Cancelling discards the analysis done so far');
    expect(dialog).not.toHaveTextContent('Cancel only if it seems to have stopped');
    await userEvent.click(within(dialog).getByRole('button', { name: 'Cancel detection' }));
    expect(onCancelDetection).toHaveBeenCalledTimes(1);
  });

  test('offers no cancel while the run is still being created', () => {
    renderPanel(undefined);

    // Nothing exists to cancel yet, so the button is present but inert rather than failing.
    expect(screen.getByRole('button', { name: 'Cancel detection' })).toBeDisabled();
  });

  test('links to the traces tab so the wait is not a dead end', () => {
    renderPanel(jest.fn<() => Promise<void>>());

    expect(screen.getByRole('button', { name: 'Go to Traces' })).toBeEnabled();
  });
});
