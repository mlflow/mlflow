import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TEST_SPAN_FILTER_STATE } from './TimelineTree.test-utils';
import { TimelineTreeHeader } from './TimelineTreeHeader';

const TestWrapper = () => {
  const [showTimelineInfo, setShowTimelineInfo] = useState<boolean>(false);
  const [spanFilterState, setSpanFilterState] = useState(TEST_SPAN_FILTER_STATE);

  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TimelineTreeHeader
          showTimelineInfo={showTimelineInfo}
          setShowTimelineInfo={setShowTimelineInfo}
          spanFilterState={spanFilterState}
          setSpanFilterState={setSpanFilterState}
        />
        <span>{String(showTimelineInfo)}</span>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('TimelineTreeHeader', () => {
  it('should switch the timeline tree view state', async () => {
    render(<TestWrapper />);

    expect(screen.getByText('false')).toBeInTheDocument();

    const showTimelineButton = screen.getByTestId('show-timeline-info-button');
    await userEvent.click(showTimelineButton);
    expect(await screen.findByText('true')).toBeInTheDocument();

    const hideTimelineButton = screen.getByTestId('hide-timeline-info-button');
    await userEvent.click(hideTimelineButton);
    expect(await screen.findByText('false')).toBeInTheDocument();
  });
});
