import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { expect, describe, it } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import { openDropdownMenu } from '.';
import { Button, DesignSystemProvider, DropdownMenu } from '../../design-system';
describe('openDropdownMenu', () => {
    it('opens dropdown menu', async () => {
        render(_jsx(DesignSystemProvider, { children: _jsxs(DropdownMenu.Root, { children: [_jsx(DropdownMenu.Trigger, { asChild: true, children: _jsx(Button, { componentId: "codegen_design-system_src_test-utils_rtl_index.test.tsx_14", children: "Open menu" }) }), _jsx(DropdownMenu.Content, { children: _jsx(DropdownMenu.Item, { componentId: "codegen_design-system_src_test-utils_rtl_index.test.tsx_17", children: "Option 1" }) })] }) }));
        await openDropdownMenu(screen.getByText('Open menu'));
        const dropdownMenu = await screen.findByRole('menu');
        expect(within(dropdownMenu).getByText('Option 1')).toBeInTheDocument();
    });
});
//# sourceMappingURL=index.test.js.map