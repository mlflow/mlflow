import { describe, expect, it, jest } from '@jest/globals';
import React from 'react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider, DesignSystemThemeProvider } from '@databricks/design-system';
import { renderWithDesignSystem, renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { LogTracesDrawer } from './LogTracesDrawer';
import { useHomePageViewState } from '../HomePageViewStateContext';
import OpenAiLogo from '../../common/static/logos/openai.svg';
import OpenAiLogoDark from '../../common/static/logos/openai-dark.svg';
import LangChainLogo from '../../common/static/logos/langchain.svg';
import LangChainLogoDark from '../../common/static/logos/langchain-dark.png';
import AnthropicLogo from '../../common/static/logos/anthropic.svg';
import AnthropicLogoDark from '../../common/static/logos/anthropic-dark.png';
import GeminiLogo from '../../common/static/logos/gemini.png';

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
  it.each([false, true])('uses contrasting framework icons with isDarkMode=%s', async (isDarkMode) => {
    renderWithIntl(
      // eslint-disable-next-line react/forbid-elements
      <DesignSystemThemeProvider isDarkMode={isDarkMode}>
        <DesignSystemProvider>
          <MemoryRouter>
            <OpenOnMount />
            <LogTracesDrawer />
          </MemoryRouter>
        </DesignSystemProvider>
      </DesignSystemThemeProvider>,
    );

    const frameworks = [
      { name: 'OpenAI', logo: OpenAiLogo, lightLogo: OpenAiLogoDark },
      { name: 'LangChain', logo: LangChainLogo, lightLogo: LangChainLogoDark },
      { name: 'Anthropic', logo: AnthropicLogo, lightLogo: AnthropicLogoDark },
    ];
    const getIcon = (name: string) => screen.getByRole('button', { name }).querySelector('img');

    await userEvent.click(screen.getByRole('button', { name: 'Gemini' }));
    for (const { name, logo, lightLogo } of frameworks) {
      expect(getIcon(name)).toHaveAttribute('src', isDarkMode ? lightLogo : logo);
    }
    expect(getIcon('LangGraph')).toHaveStyle({ filter: isDarkMode ? 'invert(1)' : '' });
    expect(getIcon('Gemini')).toHaveAttribute('src', GeminiLogo);
    expect(getIcon('Gemini')).toHaveStyle({ filter: '' });

    for (const { name, lightLogo } of frameworks) {
      await userEvent.click(screen.getByRole('button', { name }));
      expect(getIcon(name)).toHaveAttribute('src', lightLogo);
    }

    await userEvent.click(screen.getByRole('button', { name: 'LangGraph' }));
    expect(getIcon('LangGraph')).toHaveStyle({ filter: 'invert(1)' });
    expect(getIcon('Gemini')).toHaveAttribute('src', GeminiLogo);
    expect(getIcon('Gemini')).toHaveStyle({ filter: '' });
  });

  it('renders the drawer with default framework selected', () => {
    renderWithDesignSystem(
      <MemoryRouter>
        <OpenOnMount />
        <LogTracesDrawer />
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
        <OpenOnMount />
        <LogTracesDrawer />
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
