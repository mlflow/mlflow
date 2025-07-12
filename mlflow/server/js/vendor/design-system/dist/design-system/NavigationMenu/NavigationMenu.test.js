import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect } from '@jest/globals';
import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { NavigationMenu } from '.';
describe('NavigationMenu', () => {
    const Component = () => {
        const [active, setActive] = useState(0);
        return (_jsxs(NavigationMenu.Root, { children: [_jsx(NavigationMenu.List, { children: [...Array(3)].map((_o, idx) => (_jsx(NavigationMenu.Item, { active: idx === active, children: _jsx("a", { href: `#${idx}`, onClick: () => setActive(idx), children: `Link ${idx}` }) }, idx))) }), `Link ${active} Content`] }));
    };
    it('renders a set of links', async () => {
        render(_jsx(Component, {}));
        const list = screen.getByRole('list');
        const links = within(list).getAllByRole('link');
        expect(links).toHaveLength(3);
        expect(links[0]).toHaveTextContent('Link 0');
        expect(links[1]).toHaveTextContent('Link 1');
        expect(links[2]).toHaveTextContent('Link 2');
        expect(links[0]).toHaveAttribute('aria-current', 'page');
        expect(screen.getByText('Link 0 Content')).toBeInTheDocument();
        expect(screen.queryByText('Link 1 Content')).not.toBeInTheDocument();
        await userEvent.click(links[1]);
        expect(links[1]).toHaveAttribute('aria-current', 'page');
        expect(screen.getByText('Link 1 Content')).toBeInTheDocument();
        expect(screen.queryByText('Link 0 Content')).not.toBeInTheDocument();
    });
    it('links are keyboard navigable', async () => {
        render(_jsx(Component, {}));
        const list = screen.getByRole('list');
        const links = within(list).getAllByRole('link');
        act(() => {
            links[0].focus();
        });
        await userEvent.keyboard('{ArrowRight}');
        expect(links[1]).toHaveFocus();
        await userEvent.keyboard('{Enter}');
        expect(links[1]).toHaveAttribute('aria-current', 'page');
        expect(screen.getByText('Link 1 Content')).toBeInTheDocument();
        await userEvent.keyboard('{ArrowLeft}');
        expect(links[0]).toHaveFocus();
        await userEvent.keyboard('{Enter}');
        expect(links[0]).toHaveAttribute('aria-current', 'page');
        expect(screen.getByText('Link 0 Content')).toBeInTheDocument();
        await userEvent.keyboard('{Tab}');
        expect(links[1]).toHaveFocus();
        await userEvent.keyboard('{Enter}');
        expect(links[1]).toHaveAttribute('aria-current', 'page');
        expect(screen.getByText('Link 1 Content')).toBeInTheDocument();
        await userEvent.keyboard('{Shift>}{Tab}');
        expect(links[0]).toHaveFocus();
        await userEvent.keyboard('{Enter}');
        expect(links[0]).toHaveAttribute('aria-current', 'page');
        expect(screen.getByText('Link 0 Content')).toBeInTheDocument();
    });
});
//# sourceMappingURL=NavigationMenu.test.js.map