import { describe, test, expect } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { ToolCallCard, type ToolCallPart } from './ToolCallCard';

const renderCard = (part: ToolCallPart) =>
  renderWithIntl(
    <DesignSystemProvider>
      <ToolCallCard part={part} />
    </DesignSystemProvider>,
  );

const toolCall = (overrides: Partial<ToolCallPart> = {}): ToolCallPart => ({
  type: 'toolCall',
  toolUseId: 't1',
  name: 'Bash',
  input: { command: 'mlflow traces search' },
  ...overrides,
});

describe('ToolCallCard', () => {
  test('renders the tool name and a one-line input summary', () => {
    renderCard(toolCall());
    expect(screen.getByText('Bash')).toBeInTheDocument();
    expect(screen.getByText('mlflow traces search')).toBeInTheDocument();
  });

  test('summarizes trace_id with its jq_filter', () => {
    renderCard(toolCall({ name: 'trace_analyse', input: { trace_id: 'tr-1', jq_filter: '.data.spans' } }));
    expect(screen.getByText('tr-1 · .data.spans')).toBeInTheDocument();
  });

  test.each([
    ['done', 'tool-call-status-done'],
    ['error', 'tool-call-status-error'],
    ['running', 'tool-call-status-running'],
  ] as const)('renders the %s status badge', (status, testId) => {
    renderCard(toolCall({ status }));
    expect(screen.getByTestId(testId)).toBeInTheDocument();
  });

  test('treats a missing status as running', () => {
    renderCard(toolCall({ status: undefined }));
    expect(screen.getByTestId('tool-call-status-running')).toBeInTheDocument();
  });

  test('is collapsed by default and expands on click to show input and output', async () => {
    const user = userEvent.setup();
    renderCard(toolCall({ status: 'done', result: 'search results here' }));

    expect(screen.queryByText('Input')).not.toBeInTheDocument();
    expect(screen.queryByText('Output')).not.toBeInTheDocument();

    await user.click(screen.getByText('Bash'));

    expect(screen.getByText('Input')).toBeInTheDocument();
    expect(screen.getByText('Output')).toBeInTheDocument();
    await waitFor(() => expect(document.body.textContent).toContain('search results here'));
  });

  test('does not render an Output section when there is no result', async () => {
    const user = userEvent.setup();
    renderCard(toolCall({ status: 'running' }));
    await user.click(screen.getByText('Bash'));
    expect(screen.getByText('Input')).toBeInTheDocument();
    expect(screen.queryByText('Output')).not.toBeInTheDocument();
  });

  test('truncates long output with a marker', async () => {
    const user = userEvent.setup();
    const longResult = 'x'.repeat(5000);
    renderCard(toolCall({ status: 'done', result: longResult }));

    await user.click(screen.getByText('Bash'));
    await waitFor(() => expect(document.body.textContent).toContain('truncated, 1000 more chars'));
  });
});
