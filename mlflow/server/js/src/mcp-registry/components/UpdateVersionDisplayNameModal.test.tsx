import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { UpdateVersionDisplayNameModal } from './UpdateVersionDisplayNameModal';

const renderModal = (props: Partial<React.ComponentProps<typeof UpdateVersionDisplayNameModal>> = {}) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <UpdateVersionDisplayNameModal
          visible
          currentDisplayName="Original Name"
          onUpdate={jest.fn()}
          onCancel={jest.fn()}
          {...props}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('UpdateVersionDisplayNameModal', () => {
  it('renders with current display name', () => {
    renderModal();
    expect(screen.getByDisplayValue('Original Name')).toBeInTheDocument();
  });

  it('does not render when not visible', () => {
    renderModal({ visible: false });
    expect(screen.queryByDisplayValue('Original Name')).not.toBeInTheDocument();
  });

  it('calls onUpdate with trimmed value on save', async () => {
    const onUpdate = jest.fn();
    renderModal({ onUpdate });
    const input = screen.getByDisplayValue('Original Name');
    await userEvent.clear(input);
    await userEvent.type(input, '  New Name  ');
    await userEvent.click(screen.getByText('Save'));
    expect(onUpdate).toHaveBeenCalledWith('New Name');
  });

  it('shows error alert when error is provided', () => {
    renderModal({ error: new Error('Failed to save') });
    expect(screen.getByText('Failed to save')).toBeInTheDocument();
  });

  it('resets draft when visibility changes', () => {
    const { rerender } = render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <UpdateVersionDisplayNameModal
            visible={false}
            currentDisplayName="Original"
            onUpdate={jest.fn()}
            onCancel={jest.fn()}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    rerender(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <UpdateVersionDisplayNameModal
            visible
            currentDisplayName="Updated"
            onUpdate={jest.fn()}
            onCancel={jest.fn()}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );
    expect(screen.getByDisplayValue('Updated')).toBeInTheDocument();
  });
});
