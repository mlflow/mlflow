import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

import { RunViewEvaluationAnalyzeButton } from './RunViewEvaluationAnalyzeButton';

const mockOpenPanel = jest.fn();
const mockPrefillPrompt = jest.fn();
let mockCanUseAssistant = true;

jest.mock('@mlflow/mlflow/src/assistant', () => ({
  __esModule: true,
  useAssistant: () => ({
    openPanel: mockOpenPanel,
    prefillPrompt: mockPrefillPrompt,
    canUseAssistant: mockCanUseAssistant,
  }),
  AssistantSparkleIcon: ({ iconSize }: { iconSize?: number }) => (
    <span data-testid="assistant-sparkle-icon" data-icon-size={iconSize} />
  ),
}));

const renderButton = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <RunViewEvaluationAnalyzeButton runUuid="run-123" />
    </DesignSystemProvider>,
  );

beforeEach(() => {
  mockOpenPanel.mockClear();
  mockPrefillPrompt.mockClear();
  mockCanUseAssistant = true;
});

describe('RunViewEvaluationAnalyzeButton', () => {
  it('renders the Analyze button with the assistant icon', () => {
    renderButton();

    expect(screen.getByRole('button', { name: /Analyze/ })).toBeInTheDocument();
    expect(screen.getByTestId('assistant-sparkle-icon')).toBeInTheDocument();
  });

  it('hides the button when Assistant is unavailable', () => {
    mockCanUseAssistant = false;

    renderButton();

    expect(screen.queryByRole('button', { name: /Analyze/ })).not.toBeInTheDocument();
  });

  it('opens Assistant and prefills the evaluation analysis prompt', async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderButton();

    await user.click(screen.getByRole('button', { name: /Analyze/ }));

    expect(mockOpenPanel).toHaveBeenCalledTimes(1);
    expect(mockPrefillPrompt).toHaveBeenCalledTimes(1);
    expect(mockPrefillPrompt).toHaveBeenCalledWith(
      'Analyze evaluation run run-123 and provide insights on the results',
    );
  });
});
