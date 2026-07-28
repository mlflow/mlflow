import { describe, test, expect, jest } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DesignSystemProvider } from '@databricks/design-system';
import { ToolPermissionPrompt } from './ToolPermissionPrompt';

const renderPrompt = (toolInput: Record<string, any>, onRespond = jest.fn()) => {
  renderWithIntl(
    <DesignSystemProvider>
      <ToolPermissionPrompt
        request={{ sessionId: 'sess-1', requestId: 'req-1', toolName: 'Bash', toolInput }}
        onRespond={onRespond}
      />
    </DesignSystemProvider>,
  );
  return onRespond;
};

describe('ToolPermissionPrompt', () => {
  test('shows the tool name and a command preview', () => {
    renderPrompt({ command: 'ls -la' });
    expect(screen.getByText('Allow the assistant to run Bash?')).toBeInTheDocument();
    expect(screen.getByText('ls -la')).toBeInTheDocument();
  });

  test('Allow calls onRespond(true)', async () => {
    const user = userEvent.setup();
    const onRespond = renderPrompt({ command: 'ls' });
    await user.click(screen.getByRole('button', { name: 'Allow' }));
    expect(onRespond).toHaveBeenCalledWith(true);
  });

  test('Deny calls onRespond(false)', async () => {
    const user = userEvent.setup();
    const onRespond = renderPrompt({ command: 'ls' });
    await user.click(screen.getByRole('button', { name: 'Deny' }));
    expect(onRespond).toHaveBeenCalledWith(false);
  });
});
