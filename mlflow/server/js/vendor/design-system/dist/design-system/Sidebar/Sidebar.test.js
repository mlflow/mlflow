import { jsx as _jsx, jsxs as _jsxs } from "@emotion/react/jsx-runtime";
import { describe, it, expect, jest, beforeAll } from '@jest/globals';
import { render, screen, waitFor, waitForElementToBeRemoved, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { Sidebar } from './Sidebar';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
const Example = ({ initialPanelId, destroyInactivePanels, forceRenderPanelId, panelHeaderComponentId = 'example.sidebar.panel_header', }) => {
    const [openPanelId, setOpenPanelId] = useState(initialPanelId);
    return (_jsxs(Sidebar, { children: [_jsxs(Sidebar.Nav, { children: [_jsx(Sidebar.NavButton, { componentId: "example.sidebar.navbutton.panel0", active: openPanelId === 0, icon: _jsx("span", { children: "Panel0" }), onClick: () => {
                            setOpenPanelId(0);
                        } }), _jsx(Sidebar.NavButton, { componentId: "example.sidebar.navbutton.panel1", active: openPanelId === 1, icon: _jsx("span", { children: "Panel1" }), onClick: () => {
                            setOpenPanelId(1);
                        } })] }), _jsxs(Sidebar.Content, { componentId: "codegen_design-system_src_design-system_sidebar_sidebar.test.tsx_44", openPanelId: openPanelId, closable: true, onClose: () => {
                    setOpenPanelId(undefined);
                }, destroyInactivePanels: destroyInactivePanels, children: [_jsxs(Sidebar.Panel, { panelId: 0, forceRender: forceRenderPanelId === 0, children: [_jsx(Sidebar.PanelHeader, { componentId: panelHeaderComponentId, children: "Panel Title 0" }), _jsx(Sidebar.PanelBody, { children: "Panel Content 0" })] }), _jsxs(Sidebar.Panel, { panelId: 1, forceRender: forceRenderPanelId === 1, children: [_jsx(Sidebar.PanelHeader, { componentId: "codegen_design-system_src_design-system_sidebar_sidebar.test.tsx_57", children: "Panel Title 1" }), _jsx(Sidebar.PanelBody, { children: "Panel Content 1" })] })] })] }));
};
describe('Sidebar', () => {
    it('does not render panels until they are open', async () => {
        render(_jsx(Example, {}));
        expect(screen.queryByText('Panel Content 0')).not.toBeInTheDocument();
        expect(screen.queryByText('Panel Content 1')).not.toBeInTheDocument();
        const navigation = screen.getByRole('navigation');
        userEvent.click(within(navigation).getByRole('button', { name: /Panel1/ }));
        expect(await screen.findByText('Panel Content 1')).toBeVisible();
        expect(screen.queryByText('Panel Content 0')).not.toBeInTheDocument();
        // Panel 1 remains in the DOM after switching to Panel 0
        userEvent.click(within(navigation).getByRole('button', { name: /Panel0/ }));
        expect(await screen.findByText('Panel Content 0')).toBeVisible();
        expect(screen.getByText('Panel Content 1')).not.toBeVisible();
        // all panels remain in the DOM after closing
        userEvent.click(screen.getByRole('button', { name: /Close/ }));
        await waitFor(() => expect(screen.getByText('Panel Content 0')).not.toBeVisible());
        expect(screen.getByText('Panel Content 1')).not.toBeVisible();
    });
    it('removes all panels from the DOM after closing with destroyInactivePanels', async () => {
        render(_jsx(Example, { destroyInactivePanels: true }));
        expect(screen.queryByText('Panel Content 0')).not.toBeInTheDocument();
        expect(screen.queryByText('Panel Content 1')).not.toBeInTheDocument();
        const navigation = screen.getByRole('navigation');
        userEvent.click(within(navigation).getByRole('button', { name: /Panel1/ }));
        expect(await screen.findByText('Panel Content 1')).toBeVisible();
        expect(screen.queryByText('Panel Content 0')).not.toBeInTheDocument();
        // Panel 1 is removed from DOM after switching to Panel 0
        userEvent.click(within(navigation).getByRole('button', { name: /Panel0/ }));
        expect(await screen.findByText('Panel Content 0')).toBeVisible();
        expect(screen.queryByText('Panel Content 1')).not.toBeInTheDocument();
        // all panels are removed from DOM after closing
        userEvent.click(screen.getByRole('button', { name: /Close/ }));
        await waitForElementToBeRemoved(() => screen.getByText('Panel Content 0'));
        expect(screen.queryByText('Panel Content 1')).not.toBeInTheDocument();
    });
    it('controls panels that are kept in the DOM with forceRender', async () => {
        const { rerender } = render(_jsx(Example, { destroyInactivePanels: true, forceRenderPanelId: 1 }));
        expect(screen.getByText('Panel Content 1')).not.toBeVisible();
        expect(screen.queryByText('Panel Content 0')).not.toBeInTheDocument();
        const navigation = screen.getByRole('navigation');
        userEvent.click(within(navigation).getByRole('button', { name: /Panel0/ }));
        expect(await screen.findByText('Panel Content 0')).toBeVisible();
        expect(screen.getByText('Panel Content 1')).not.toBeVisible();
        // Panel 1 is removed from the DOM after removing forceRender
        rerender(_jsx(Example, { destroyInactivePanels: true }));
        expect(screen.queryByText('Panel Content 1')).not.toBeInTheDocument();
        expect(screen.getByText('Panel Content 0')).toBeVisible();
    });
    it('emits onClick event when panel header close button is clicked', async () => {
        const eventCallback = jest.fn();
        const panelHeaderComponentId = 'example.sidebar.panel_header.close';
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Example, { panelHeaderComponentId: panelHeaderComponentId }) }));
        expect(eventCallback).not.toHaveBeenCalled();
        const navigation = screen.getByRole('navigation');
        await userEvent.click(within(navigation).getByRole('button', { name: /Panel0/ }));
        expect(await screen.findByText('Panel Content 0')).toBeVisible();
        expect(eventCallback).toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'example.sidebar.navbutton.panel0',
            componentType: 'button',
            shouldStartInteraction: true,
            isInteractionSubject: true,
            value: undefined,
            event: expect.anything(),
        });
        await userEvent.click(screen.getByRole('button', { name: /Close/ }));
        await waitFor(() => expect(screen.getByText('Panel Content 0')).not.toBeVisible());
        expect(eventCallback).toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: `${panelHeaderComponentId}.close`,
            componentType: 'button',
            shouldStartInteraction: true,
            isInteractionSubject: true,
            value: undefined,
            event: expect.anything(),
        });
    });
    describe('compact mode', () => {
        const contentComponentId = 'example.sidebar.content';
        const renderNonClosablePanel = (eventCallback = () => { }) => {
            const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
            return render(_jsxs("div", { css: { height: 500, display: 'flex' }, children: [_jsx(DesignSystemEventProviderForTest, { children: _jsx(Sidebar, { children: _jsx(Sidebar.Content, { componentId: contentComponentId, openPanelId: 0, closable: false, disableResize: true, enableCompact: true, children: _jsxs(Sidebar.Panel, { panelId: 0, children: [_jsx(Sidebar.PanelHeader, { componentId: "codegen_design-system_src_design-system_sidebar_sidebar.test.tsx_177", children: "Panel Title 0" }), _jsx(Sidebar.PanelBody, { children: "Panel Content 0" })] }) }) }) }), _jsx("div", { css: { marginLeft: 10 }, children: "Main Panel" })] }));
        };
        it('should not show toggle button by default', () => {
            renderNonClosablePanel();
            expect(screen.queryByRole('button')).not.toBeInTheDocument();
        });
        describe('with compact viewport', () => {
            beforeAll(() => {
                Object.defineProperty(global, 'matchMedia', {
                    writable: true,
                    configurable: true,
                    value: jest.fn((query) => ({
                        get matches() {
                            // assumes it is compact screen
                            return true;
                        },
                        media: query,
                        addEventListener: () => { },
                        removeEventListener: () => { },
                    })),
                });
            });
            it('should render toggle button for compact screen', () => {
                renderNonClosablePanel();
                expect(screen.getByRole('button', { name: 'hide sidebar' })).toBeInTheDocument();
            });
            it('hide / expand panel when toggle button is clicked', async () => {
                renderNonClosablePanel();
                await userEvent.click(screen.getByRole('button', { name: 'hide sidebar' }));
                expect(screen.getByText('Panel Title 0')).not.toBeVisible();
                await userEvent.click(screen.getByRole('button', { name: 'expand sidebar' }));
                expect(screen.getByText('Panel Title 0')).toBeVisible();
            });
            it('emits onClick event when toggle button is clicked', async () => {
                const eventCallback = jest.fn();
                renderNonClosablePanel(eventCallback);
                expect(eventCallback).not.toHaveBeenCalled();
                await userEvent.click(screen.getByRole('button', { name: 'hide sidebar' }));
                expect(screen.getByText('Panel Title 0')).not.toBeVisible();
                expect(eventCallback).toHaveBeenCalled();
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onClick',
                    componentId: `${contentComponentId}.toggle`,
                    componentType: 'button',
                    shouldStartInteraction: true,
                    isInteractionSubject: true,
                    value: undefined,
                    event: expect.anything(),
                });
                await userEvent.click(screen.getByRole('button', { name: 'expand sidebar' }));
                expect(screen.getByText('Panel Title 0')).toBeVisible();
                expect(eventCallback).toHaveBeenCalled();
                expect(eventCallback).toHaveBeenCalledWith({
                    eventType: 'onClick',
                    componentId: `${contentComponentId}.toggle`,
                    componentType: 'button',
                    shouldStartInteraction: true,
                    isInteractionSubject: true,
                    value: undefined,
                    event: expect.anything(),
                });
            });
        });
    });
});
//# sourceMappingURL=Sidebar.test.js.map