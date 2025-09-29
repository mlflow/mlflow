import React from 'react';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { DarkThemeSwitch } from './DarkThemeSwitch';
import userEvent from '@testing-library/user-event';

describe('DarkThemeSwitch', () => {
  const mockSetIsDarkTheme = jest.fn();

  beforeEach(() => {
    mockSetIsDarkTheme.mockClear();
  });

  test('should render toggle button with sun icon when light theme is active', () => {
    renderWithIntl(<DarkThemeSwitch isDarkTheme={false} setIsDarkTheme={mockSetIsDarkTheme} />);

    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
    expect(button).toHaveAttribute('aria-label', 'Switch to dark theme');
  });

  test('should render toggle button with moon icon when dark theme is active', () => {
    renderWithIntl(<DarkThemeSwitch isDarkTheme setIsDarkTheme={mockSetIsDarkTheme} />);

    const button = screen.getByRole('button');
    expect(button).toBeInTheDocument();
    expect(button).toHaveAttribute('aria-label', 'Switch to light theme');
  });

  test('should call setIsDarkTheme with opposite value when clicked', async () => {
    renderWithIntl(<DarkThemeSwitch isDarkTheme={false} setIsDarkTheme={mockSetIsDarkTheme} />);

    const button = screen.getByRole('button');
    await userEvent.click(button);

    expect(mockSetIsDarkTheme).toHaveBeenCalledWith(true);
  });

  test('should call setIsDarkTheme with opposite value when clicked in dark mode', async () => {
    renderWithIntl(<DarkThemeSwitch isDarkTheme setIsDarkTheme={mockSetIsDarkTheme} />);

    const button = screen.getByRole('button');
    await userEvent.click(button);

    expect(mockSetIsDarkTheme).toHaveBeenCalledWith(false);
  });

  test('should be keyboard accessible', async () => {
    renderWithIntl(<DarkThemeSwitch isDarkTheme={false} setIsDarkTheme={mockSetIsDarkTheme} />);

    const button = screen.getByRole('button');
    button.focus();

    await userEvent.keyboard('{Enter}');
    expect(mockSetIsDarkTheme).toHaveBeenCalledWith(true);
  });
});
