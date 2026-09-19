import { jest, describe, it, expect } from '@jest/globals';
import { fireEvent, screen, within } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
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

    await userEvent.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await userEvent.hover(screen.getByRole('menuitem', { name: 'Filter by span type' }));

    // assert that the filter submenu is open
    expect(await screen.findByRole('menuitemcheckbox', { name: 'Chain' })).toBeInTheDocument();

    // Check that the show parents checkbox toggles the state
    expect(screen.getByText('Show parents true')).toBeInTheDocument();
    const showParentsCheckbox = screen.getByRole('menuitemcheckbox', { name: 'Show all parent spans' });
    await userEvent.click(showParentsCheckbox);
    expect(screen.getByText('Show parents false')).toBeInTheDocument();

    // Same for show exceptions
    expect(screen.getByText('Show exceptions true')).toBeInTheDocument();
    const showExceptionsCheckbox = screen.getByRole('menuitemcheckbox', { name: 'Show exceptions' });
    await userEvent.click(showExceptionsCheckbox);
    expect(screen.getByText('Show exceptions false')).toBeInTheDocument();

    // Same for span type filters (just check one for simplicity)
    expect(screen.getByText('Show chain spans true')).toBeInTheDocument();
    const showChainCheckbox = screen.getByRole('menuitemcheckbox', { name: 'Chain' });
    await userEvent.click(showChainCheckbox);
    expect(screen.getByText('Show chain spans false')).toBeInTheDocument();
  });

  it('explains why parent spans can remain visible', async () => {
    const user = userEvent.setup();
    render(<TestWrapper />);

    await user.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await user.hover(screen.getByRole('menuitem', { name: 'Filter by span type' }));
    const showParentsCheckbox = await screen.findByRole('menuitemcheckbox', { name: 'Show all parent spans' });
    fireEvent.pointerMove(
      within(showParentsCheckbox).getByRole('img', { name: 'More information about showing parent spans' }),
      { pointerType: 'mouse' },
    );

    expect(
      await screen.findByRole('tooltip', {
        name: 'Always show parents of matched spans, regardless of filter conditions',
      }),
    ).toBeInTheDocument();
  });

  it('explains why exception spans can remain visible', async () => {
    const user = userEvent.setup();
    render(<TestWrapper />);

    await user.click(screen.getByRole('button', { name: 'Trace display settings' }));
    await user.hover(screen.getByRole('menuitem', { name: 'Filter by span type' }));
    const showExceptionsCheckbox = await screen.findByRole('menuitemcheckbox', { name: 'Show exceptions' });
    fireEvent.pointerMove(
      within(showExceptionsCheckbox).getByRole('img', { name: 'More information about showing exception spans' }),
      { pointerType: 'mouse' },
    );

    expect(
      await screen.findByRole('tooltip', {
        name: 'Always show spans with exceptions, regardless of filter conditions',
      }),
    ).toBeInTheDocument();
  });
});
