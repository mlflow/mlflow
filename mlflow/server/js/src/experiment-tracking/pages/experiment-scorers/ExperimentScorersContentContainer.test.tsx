import React from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { jest, describe, it, expect } from '@jest/globals';
import ExperimentScorersContentContainer from './ExperimentScorersContentContainer';
import { useGetScheduledScorers } from './hooks/useGetScheduledScorers';
import { LLM_TEMPLATE } from './types';

jest.mock('./hooks/useGetScheduledScorers');
jest.mock('./ScorerCardContainer', () => ({
  __esModule: true,
  default: () => <div>Existing judge</div>,
}));
jest.mock('./ScorerModalRenderer', () => ({
  __esModule: true,
  default: ({ visible, initialTemplate }: { visible: boolean; initialTemplate: LLM_TEMPLATE }) =>
    visible ? <div data-testid="create-modal">{initialTemplate}</div> : null,
}));

describe('ExperimentScorersContentContainer', () => {
  it.each([false, true])('opens the judge dropdown before the modal when existing scorers = %s', async (hasScorers) => {
    jest.mocked(useGetScheduledScorers).mockReturnValue({
      data: {
        scheduledScorers: hasScorers ? [{ name: 'Existing judge', type: 'llm' }] : [],
      },
      isLoading: false,
      isError: false,
      error: null,
    } as unknown as ReturnType<typeof useGetScheduledScorers>);

    render(
      <DesignSystemProvider>
        <IntlProvider locale="en">
          <ExperimentScorersContentContainer experimentId="experiment-1" />
        </IntlProvider>
      </DesignSystemProvider>,
    );

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'New LLM judge' }));

    expect(screen.queryByTestId('create-modal')).not.toBeInTheDocument();
    expect(screen.getByRole('menuitem', { name: 'Completeness' })).toBeInTheDocument();

    await user.click(screen.getByRole('menuitem', { name: 'Completeness' }));
    expect(screen.getByTestId('create-modal')).toHaveTextContent(LLM_TEMPLATE.COMPLETENESS);
  });
});
