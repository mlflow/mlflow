import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { MCPServerVersionDiffSelectorButton } from './MCPServerVersionDiffSelectorButton';

const renderButton = (props: Partial<React.ComponentProps<typeof MCPServerVersionDiffSelectorButton>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <MCPServerVersionDiffSelectorButton isSelectedBaseline={false} isSelectedCompared={false} {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('MCPServerVersionDiffSelectorButton', () => {
  it('renders two radio buttons', () => {
    renderButton();
    const radios = screen.getAllByRole('radio');
    expect(radios).toHaveLength(2);
  });

  it('marks baseline as checked when selected', () => {
    renderButton({ isSelectedBaseline: true });
    const radios = screen.getAllByRole('radio');
    expect(radios[0]).toHaveAttribute('aria-checked', 'true');
    expect(radios[1]).toHaveAttribute('aria-checked', 'false');
  });

  it('marks compared as checked when selected', () => {
    renderButton({ isSelectedCompared: true });
    const radios = screen.getAllByRole('radio');
    expect(radios[0]).toHaveAttribute('aria-checked', 'false');
    expect(radios[1]).toHaveAttribute('aria-checked', 'true');
  });

  it('calls onSelectBaseline when baseline button is clicked', async () => {
    const onSelectBaseline = jest.fn();
    renderButton({ onSelectBaseline });
    const radios = screen.getAllByRole('radio');
    await userEvent.click(radios[0]);
    expect(onSelectBaseline).toHaveBeenCalledTimes(1);
  });

  it('calls onSelectCompared when compared button is clicked', async () => {
    const onSelectCompared = jest.fn();
    renderButton({ onSelectCompared });
    const radios = screen.getAllByRole('radio');
    await userEvent.click(radios[1]);
    expect(onSelectCompared).toHaveBeenCalledTimes(1);
  });
});
