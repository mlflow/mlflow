import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { SaveToRegistryButton } from './SaveToRegistryButton';

const renderButton = (props: Partial<React.ComponentProps<typeof SaveToRegistryButton>> = {}) => {
  const onOpen = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <SaveToRegistryButton onOpen={onOpen} {...props} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onOpen };
};

describe('SaveToRegistryButton', () => {
  it('renders the localized label', () => {
    renderButton();
    expect(screen.getByRole('button', { name: /save prompt/i })).toBeInTheDocument();
  });

  it('calls onOpen when clicked', async () => {
    const { onOpen } = renderButton();
    await userEvent.click(screen.getByRole('button', { name: /save prompt/i }));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it('does not call onOpen when disabled', async () => {
    const { onOpen } = renderButton({ disabled: true });
    expect(screen.getByRole('button', { name: /save prompt/i })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: /save prompt/i }));
    expect(onOpen).not.toHaveBeenCalled();
  });
});
