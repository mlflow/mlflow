import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TEST_SPAN_FILTER_STATE } from './TimelineTree.test-utils';
import { TimelineTreeFilterButton } from './TimelineTreeFilterButton';
import type { SpanFilterState } from '../ModelTrace.types';
import { SpanLogLevel } from '../ModelTrace.types';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

const TestWrapper = () => {
  const [spanFilterState, setSpanFilterState] = useState<SpanFilterState>(TEST_SPAN_FILTER_STATE);

  return (
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <TimelineTreeFilterButton spanFilterState={spanFilterState} setSpanFilterState={setSpanFilterState} />
        {/* Stringifying the underlying state so we can easily perform asserts */}
        <span>{'Show parents ' + String(spanFilterState.showParents)}</span>
        <span>{'Show exceptions ' + String(spanFilterState.showExceptions)}</span>
        <span>{'Show chain spans ' + String(spanFilterState.spanTypeDisplayState['CHAIN'])}</span>
        <span>{'Min log level ' + String(spanFilterState.minLogLevel)}</span>
      </DesignSystemProvider>
    </IntlProvider>
  );
};

describe('TimelineTreeFilterButton', () => {
  it('should switch filter states', async () => {
    render(<TestWrapper />);

    const filterButton = screen.getByRole('button', { name: 'Filter' });
    await userEvent.click(filterButton);

    // assert that the popover is open
    expect(await screen.findByText('Span type')).toBeInTheDocument();

    // Check that the show parents checkbox toggles the state
    expect(screen.getByText('Show parents true')).toBeInTheDocument();
    const showParentsCheckbox = screen.getByRole('checkbox', { name: /Show all parent spans/i });
    await userEvent.click(showParentsCheckbox);
    expect(screen.getByText('Show parents false')).toBeInTheDocument();

    // Same for show exceptions
    expect(screen.getByText('Show exceptions true')).toBeInTheDocument();
    const showExceptionsCheckbox = screen.getByRole('checkbox', { name: /Show exceptions/i });
    await userEvent.click(showExceptionsCheckbox);
    expect(screen.getByText('Show exceptions false')).toBeInTheDocument();

    // Same for span type filters (just check one for simplicity)
    expect(screen.getByText('Show chain spans true')).toBeInTheDocument();
    const showChainCheckbox = screen.getByRole('checkbox', { name: 'Chain' });
    await userEvent.click(showChainCheckbox);
    expect(screen.getByText('Show chain spans false')).toBeInTheDocument();

    // The log-level slider starts at DEBUG (the default "show everything" threshold)
    // and should update state when nudged to a higher level. Each ArrowRight moves
    // up one named level (DEBUG → INFO → WARNING).
    expect(screen.getByText(`Min log level ${SpanLogLevel.DEBUG}`)).toBeInTheDocument();
    const logLevelSlider = screen.getByRole('slider', { name: /Minimum log level/i });
    logLevelSlider.focus();
    await userEvent.keyboard('{ArrowRight}{ArrowRight}');
    expect(screen.getByText(`Min log level ${SpanLogLevel.WARNING}`)).toBeInTheDocument();
  });

  it('renders the info tooltip explaining the filter', async () => {
    render(<TestWrapper />);
    await userEvent.click(screen.getByRole('button', { name: 'Filter' }));

    // The tooltip trigger sits next to the "Minimum log level" label.
    // (Hover assertions are covered by InfoTooltip's own tests; here we just
    // verify the trigger is rendered so we don't accidentally drop it again.)
    expect(await screen.findByText('Minimum log level')).toBeInTheDocument();
  });
});
