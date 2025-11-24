import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HoverCard } from '.';
import { Button } from '../Button';
import { DesignSystemProvider } from '../DesignSystemProvider';
describe('HoverCard', function () {
    function renderComponent() {
        return render(_jsx(DesignSystemProvider, { children: _jsx(HoverCard, { trigger: _jsx(Button, { componentId: "codegen_design-system_src_design-system_hovercard_hovercard.test.tsx_14", "data-testid": "test-trigger", children: "Hover to see content" }), content: _jsx("div", { children: "HoverCard content" }), align: "start" }) }));
    }
    it('renders HoverCard on hover and hides on mouse leave', async () => {
        renderComponent();
        // Trigger hover event
        await userEvent.hover(screen.getByTestId('test-trigger'));
        // Wait for content to appear
        await waitFor(() => {
            expect(screen.getByText('HoverCard content')).toBeInTheDocument();
        });
        // Trigger unhover event
        await userEvent.unhover(screen.getByTestId('test-trigger'));
        // Wait for content to disappear
        await waitFor(() => {
            expect(screen.queryByText('HoverCard content')).toBeNull();
        });
    });
});
//# sourceMappingURL=HoverCard.test.js.map