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

  test('cancels only after the dialog is confirmed', async () => {
    const onCancelDetection = jest.fn<() => Promise<void>>().mockResolvedValue(undefined);
    renderPanel(onCancelDetection);

    await userEvent.click(screen.getByRole('button', { name: 'Cancel detection' }));
    expect(onCancelDetection).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Keep running' }));
    expect(onCancelDetection).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Cancel detection' }));
    await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Cancel detection' }));
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
