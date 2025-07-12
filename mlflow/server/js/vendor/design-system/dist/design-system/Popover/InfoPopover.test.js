import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { InfoPopover } from './InfoPopover';
import { DesignSystemProvider } from '../DesignSystemProvider';
import { Modal } from '../Modal';
const TestComponent = () => {
    const [isModalOpen, setIsModalOpen] = useState(false);
    const handleModalClose = jest.fn(() => setIsModalOpen(false));
    return (_jsxs(DesignSystemProvider, { children: [_jsx("button", { onClick: () => setIsModalOpen(true), children: "Open Modal" }), isModalOpen && (_jsx(Modal, { componentId: "test-modal", onCancel: handleModalClose, title: "Test Modal", visible: isModalOpen, children: _jsx(InfoPopover, { ariaLabel: "Test InfoPopover", children: "Popover content" }) }))] }));
};
describe('InfoPopover', () => {
    it('closes InfoPopover with ESC key without closing Modal', async () => {
        render(_jsx(TestComponent, {}));
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
//# sourceMappingURL=InfoPopover.test.js.map