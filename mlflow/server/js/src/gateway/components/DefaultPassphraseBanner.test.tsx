import { describe, expect, it, beforeEach } from '@jest/globals';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DefaultPassphraseBanner } from './DefaultPassphraseBanner';

describe('DefaultPassphraseBanner', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('renders security notice banner', () => {
    renderWithDesignSystem(<DefaultPassphraseBanner />);

    expect(screen.getByText('Security Notice: Default Passphrase in Use')).toBeInTheDocument();
  });

  it('dismisses banner on close and persists in localStorage', async () => {
    renderWithDesignSystem(<DefaultPassphraseBanner />);

    expect(screen.getByText('Security Notice: Default Passphrase in Use')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /close/i }));

    expect(screen.queryByText('Security Notice: Default Passphrase in Use')).not.toBeInTheDocument();
    expect(localStorage.getItem('mlflow.gateway.default-passphrase-warning.dismissed_v1')).toBe('true');
  });

  it('does not render when previously dismissed', () => {
    localStorage.setItem('mlflow.gateway.default-passphrase-warning.dismissed_v1', 'true');

    renderWithDesignSystem(<DefaultPassphraseBanner />);

    expect(screen.queryByText('Security Notice: Default Passphrase in Use')).not.toBeInTheDocument();
  });
});
