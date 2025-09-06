import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TEST_SPAN_FILTER_STATE } from './TimelineTree.test-utils';
import { TimelineTreeFilterButton } from './TimelineTreeFilterButton';
import type { SpanFilterState } from '../ModelTrace.types';

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
  });
});
