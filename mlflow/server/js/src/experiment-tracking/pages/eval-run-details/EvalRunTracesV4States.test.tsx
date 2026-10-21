import { describe, expect, jest, test } from '@jest/globals';
import { type ReactNode } from 'react';
import userEvent from '@testing-library/user-event';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { EvalRunSelectedTraceErrorAlert, EvalRunTracesEmptyState } from './EvalRunTracesV4States';

const renderState = (component: ReactNode) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{component}</DesignSystemProvider>
    </IntlProvider>,
  );

describe('EvalRunTracesV4States', () => {
  test('describes an empty evaluation run without experiment-wide tracing guidance', () => {
    renderState(<EvalRunTracesEmptyState />);

    expect(screen.getByText('No traces in this evaluation run')).toBeInTheDocument();
    expect(screen.getByText('This evaluation run did not produce any trace results.')).toBeInTheDocument();
    expect(screen.queryByText('No traces recorded')).not.toBeInTheDocument();
  });

  test('shows a translated selected-trace failure and retries it', async () => {
    const onRetry = jest.fn();
    renderState(<EvalRunSelectedTraceErrorAlert onRetry={onRetry} />);

    expect(screen.getByText("Couldn't open the selected trace")).toBeInTheDocument();
    expect(screen.getByText('The trace could not be verified as part of this evaluation run.')).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
