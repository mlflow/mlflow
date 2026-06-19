import { describe, test, expect } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { ToolCallCard, ToolCallGroup, groupStatus, toolNameSummary, type ToolCallPart } from './ToolCallCard';

const renderCard = (part: ToolCallPart) =>
  renderWithIntl(
    <DesignSystemProvider>
      <ToolCallCard part={part} />
    </DesignSystemProvider>,
  );

const renderGroup = (parts: ToolCallPart[]) =>
  renderWithIntl(
    <DesignSystemProvider>
      <ToolCallGroup parts={parts} />
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

describe('groupStatus', () => {
  test('is running while any call is unresolved', () => {
    expect(groupStatus([toolCall({ status: 'done' }), toolCall({ status: 'running' })])).toBe('running');
    expect(groupStatus([toolCall({ status: undefined })])).toBe('running');
  });

  test('is error when the run ends on a failure', () => {
    expect(groupStatus([toolCall({ status: 'done' }), toolCall({ status: 'error' })])).toBe('error');
  });

  test('is done when a later call recovers from an earlier error', () => {
    expect(groupStatus([toolCall({ status: 'error' }), toolCall({ status: 'done' })])).toBe('done');
  });

  test('is done when every call resolved successfully', () => {
    expect(groupStatus([toolCall({ status: 'done' }), toolCall({ status: 'done' })])).toBe('done');
  });
});

describe('toolNameSummary', () => {
  test('dedupes repeated names with a count, preserving first-seen order', () => {
    const parts = [toolCall({ name: 'load_skill' }), toolCall({ name: 'Bash' }), toolCall({ name: 'Bash' })];
    expect(toolNameSummary(parts)).toBe('load_skill, Bash ×2');
  });

  test('omits the count for a single occurrence', () => {
    expect(toolNameSummary([toolCall({ name: 'Bash' })])).toBe('Bash');
  });
});

describe('ToolCallGroup', () => {
  const calls = [
    toolCall({ toolUseId: 't1', name: 'load_skill', status: 'done', result: 'a' }),
    toolCall({ toolUseId: 't2', name: 'Bash', status: 'done', result: 'b' }),
    toolCall({ toolUseId: 't3', name: 'Bash', status: 'done', result: 'c' }),
  ];

  test('renders the count, name summary, and a status label', () => {
    renderGroup(calls);
    expect(screen.getByText('3 tool calls')).toBeInTheDocument();
    expect(screen.getByText('load_skill, Bash ×2')).toBeInTheDocument();
    expect(screen.getByTestId('tool-group-status-done')).toBeInTheDocument();
  });

  test('treats a missing status as running', () => {
    renderGroup([toolCall({ status: undefined })]);
    expect(screen.getByTestId('tool-group-status-running')).toBeInTheDocument();
  });

  test('surfaces a failed call in the header status', () => {
    renderGroup([toolCall({ status: 'done' }), toolCall({ toolUseId: 't2', status: 'error' })]);
    expect(screen.getByTestId('tool-group-status-error')).toBeInTheDocument();
  });

  test('is collapsed by default and expands to reveal the inner cards', async () => {
    const user = userEvent.setup();
    renderGroup(calls);

    expect(screen.queryByLabelText('Tool call: load_skill')).not.toBeInTheDocument();

    await user.click(screen.getByText('3 tool calls'));

    expect(screen.getByLabelText('Tool call: load_skill')).toBeInTheDocument();
    expect(screen.getAllByLabelText('Tool call: Bash')).toHaveLength(2);
  });
});
