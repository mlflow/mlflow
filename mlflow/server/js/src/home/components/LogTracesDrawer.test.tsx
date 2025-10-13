import React from 'react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { LogTracesDrawer } from './LogTracesDrawer';
import { HomePageViewStateProvider, useHomePageViewState } from '../HomePageViewStateContext';

jest.mock('@mlflow/mlflow/src/experiment-tracking/components/traces/quickstart/TraceTableGenericQuickstart', () => ({
  TraceTableGenericQuickstart: ({ flavorName, baseComponentId }: { flavorName: string; baseComponentId: string }) => (
    <div data-testid="quickstart" data-flavor={flavorName} data-base={baseComponentId} />
  ),
}));

const OpenOnMount = () => {
  const { openLogTracesDrawer } = useHomePageViewState();
  React.useEffect(() => {
    openLogTracesDrawer();
  }, [openLogTracesDrawer]);
  return null;
};

describe('LogTracesDrawer', () => {
  it('renders the drawer with default framework selected', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <HomePageViewStateProvider>
          <OpenOnMount />
          <LogTracesDrawer />
        </HomePageViewStateProvider>
      </MemoryRouter>,
    );

    expect(
      screen.getByRole('dialog', {
        name: 'Log traces',
      }),
    ).toBeInTheDocument();

    const openAiButton = screen.getByRole('button', { name: 'OpenAI' });
    expect(openAiButton).toHaveAttribute('aria-pressed', 'true');

    const quickstart = screen.getByTestId('quickstart');
    expect(quickstart).toHaveAttribute('data-flavor', 'openai');
    expect(quickstart).toHaveAttribute('data-base', 'mlflow.home.log_traces.drawer.openai');
  });

  it('updates quickstart content when selecting a different framework', async () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <HomePageViewStateProvider>
          <OpenOnMount />
          <LogTracesDrawer />
        </HomePageViewStateProvider>
      </MemoryRouter>,
    );

    const langChainButton = screen.getByRole('button', { name: 'LangChain' });
    await userEvent.click(langChainButton);

    expect(langChainButton).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'OpenAI' })).toHaveAttribute('aria-pressed', 'false');

    const quickstart = screen.getByTestId('quickstart');
    expect(quickstart).toHaveAttribute('data-flavor', 'langchain');
    expect(quickstart).toHaveAttribute('data-base', 'mlflow.home.log_traces.drawer.langchain');
  });
});
