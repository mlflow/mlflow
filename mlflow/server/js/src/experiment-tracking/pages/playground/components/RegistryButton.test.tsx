import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { RegistryButton } from './RegistryButton';

describe('RegistryButton', () => {
  it('renders the localized label', () => {
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <RegistryButton onOpen={jest.fn()} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    expect(screen.getByRole('button', { name: /load prompt from registry/i })).toBeInTheDocument();
  });

  it('calls onOpen when clicked', async () => {
    const onOpen = jest.fn();
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <RegistryButton onOpen={onOpen} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    await userEvent.click(screen.getByRole('button', { name: /load prompt from registry/i }));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });
});
