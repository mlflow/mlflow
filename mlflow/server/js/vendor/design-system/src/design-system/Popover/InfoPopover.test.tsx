import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { InfoPopover } from './InfoPopover';
import { DesignSystemProvider } from '../DesignSystemProvider';
import { Modal } from '../Modal';

const TestComponent = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const handleModalClose = jest.fn(() => setIsModalOpen(false));

  return (
    <DesignSystemProvider>
      <button onClick={() => setIsModalOpen(true)}>Open Modal</button>
      {isModalOpen && (
        <Modal componentId="test-modal" onCancel={handleModalClose} title="Test Modal" visible={isModalOpen}>
          <InfoPopover ariaLabel="Test InfoPopover">Popover content</InfoPopover>
        </Modal>
      )}
    </DesignSystemProvider>
  );
};

describe('InfoPopover', () => {
  it('closes InfoPopover with ESC key without closing Modal', async () => {
    render(<TestComponent />);

    // Open the Modal
    await userEvent.click(screen.getByText('Open Modal'));
    expect(screen.getByText('Test Modal')).toBeInTheDocument();

    // Open the InfoPopover
    await userEvent.click(screen.getByLabelText('Test InfoPopover'));
    expect(screen.getByText('Popover content')).toBeInTheDocument();

    // Close the InfoPopover with ESC key
    await userEvent.keyboard('{Escape}');

    // Verify that the InfoPopover is closed
    // Verify that the Modal is still open
    expect(screen.queryByText('Popover content')).not.toBeInTheDocument();
    expect(screen.getByText('Test Modal')).toBeInTheDocument();
  });
});
