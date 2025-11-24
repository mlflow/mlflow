import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, expect, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Tag } from './index';
import { DesignSystemEventProviderAnalyticsEventTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
describe('Tag', () => {
    it('renders a tag with the correct text', async () => {
        render(_jsx(Tag, { componentId: "codegen_design-system_src_design-system_tag_tag.test.tsx_9", children: "Tag" }));
        expect(screen.getByText('Tag')).toBeInTheDocument();
    });
    it('closes the tag when the close button is clicked', async () => {
        const closeCallback = jest.fn();
        render(_jsx(Tag, { componentId: "codegen_design-system_src_design-system_tag_tag.test.tsx_16", onClose: closeCallback, closable: true, children: "Tag" }));
        expect(screen.getByText('Tag')).toBeInTheDocument();
        const closeButton = screen.getByRole('button');
        await userEvent.click(closeButton);
        expect(closeCallback).toHaveBeenCalled();
    });
    it('emits click event when the tag is clicked', async () => {
        const clickCallback = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Tag, { componentId: "tag_test", onClick: clickCallback, analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick], children: "Tag" }) }));
        const tagElement = screen.getByText('Tag');
        expect(tagElement).toBeInTheDocument();
        await userEvent.click(tagElement);
        expect(clickCallback).toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'tag_test',
            componentType: 'tag',
            shouldStartInteraction: true,
            value: undefined,
            event: expect.anything(),
            interactionSubject: undefined,
        });
    });
    it('emits view event when the tag is viewed', async () => {
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Tag, { componentId: "tag_test", analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnView], children: "Tag" }) }));
        expect(screen.getByText('Tag')).toBeVisible();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'tag_test',
            componentType: 'tag',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
    it('emits click event when the tag is closed', async () => {
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Tag, { closable: true, componentId: "tag_test", children: "Tag" }) }));
        // no onView event should be emitted
        const closeButton = screen.getByRole('button');
        await userEvent.click(closeButton);
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: 'tag_test.close',
            componentType: 'button',
            shouldStartInteraction: true,
            value: undefined,
            event: expect.any(Object),
            isInteractionSubject: undefined,
        });
    });
});
//# sourceMappingURL=Tag.test.js.map