import { describe, expect, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { MemoryRouter, useLocation } from '../../../common/utils/RoutingUtils';
import { ExperimentIssuesDetectionProgress } from './ExperimentIssuesDetectionProgress';

const Location = () => {
  const location = useLocation();
  return <output aria-label="Current URL">{location.pathname + location.search}</output>;
};

const renderPanel = () =>
  renderWithDesignSystem(
    <MemoryRouter initialEntries={['/experiments/1/issues?startTimeLabel=LAST_7_DAYS']}>
      <ExperimentIssuesDetectionProgress experimentId="1" />
      <Location />
    </MemoryRouter>,
  );

describe('ExperimentIssuesDetectionProgress', () => {
  test('announces the run in progress to assistive technology', () => {
    renderPanel();

    const status = screen.getByRole('status');
    expect(status).toHaveAttribute('aria-live', 'polite');
    expect(status).toHaveTextContent('Detecting issues...');
    // The run outlives the page, which is the thing a waiting user needs told.
    expect(status).toHaveTextContent('you can leave this page and return later');
  });

  test('links to the traces tab and retains its time range', async () => {
    renderPanel();

    expect(screen.queryByRole('button', { name: 'Cancel detection' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Go to Traces' }));
    expect(screen.getByLabelText('Current URL')).toHaveTextContent('/experiments/1/traces?startTimeLabel=LAST_7_DAYS');
  });
});
