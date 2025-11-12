import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Card } from './Card';
import { DesignSystemEventProviderAnalyticsEventTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
describe('<Card/>', () => {
    const cardComponentId = 'test_card_id';
    const cardText = 'A';
    const handleClickSpy = jest.fn();
    const eventCallback = jest.fn();
    const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
    const renderComponent = (href, analyticsEvents = [DesignSystemEventProviderAnalyticsEventTypes.OnClick]) => {
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Card, { componentId: cardComponentId, onClick: handleClickSpy, href: href, analyticsEvents: analyticsEvents, children: "A" }) }));
    };
    it('emits onClick event on mouse click', async () => {
        renderComponent();
        expect(handleClickSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        await userEvent.click(screen.getByText(cardText));
        expect(handleClickSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: cardComponentId,
            componentType: 'card',
            shouldStartInteraction: true,
            value: undefined,
            isInteractionSubject: undefined,
            event: expect.any(Object),
        });
    });
    it('emits onClick event on keyboard navigable select Tab+Enter', async () => {
        renderComponent('databricks.com');
        expect(handleClickSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        // handleSelection() is only called when the inner content is focused
        // Tab to the <a> tag, Tab to the inner card content, Enter to trigger handleSelection()
        await userEvent.keyboard('{Tab}{Tab}{Enter}');
        expect(handleClickSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: cardComponentId,
            componentType: 'card',
            shouldStartInteraction: true,
            value: undefined,
            isInteractionSubject: undefined,
            event: expect.any(Object),
        });
    });
    it('emits onClick event on keyboard navigable select Tab+Space', async () => {
        renderComponent('databricks.com');
        expect(handleClickSpy).not.toHaveBeenCalled();
        expect(eventCallback).not.toHaveBeenCalled();
        // handleSelection() is only called when the inner content is focused
        // Tab to the <a> tag, Tab to the inner card content, Space to trigger handleSelection()
        await userEvent.keyboard('{Tab}{Tab} ');
        expect(handleClickSpy).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onClick',
            componentId: cardComponentId,
            componentType: 'card',
            shouldStartInteraction: true,
            value: undefined,
            isInteractionSubject: undefined,
            event: expect.any(Object),
        });
    });
    it('emits onView on first render', async () => {
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
        renderComponent('databricks.com', [DesignSystemEventProviderAnalyticsEventTypes.OnView]);
        expect(screen.getByText('A')).toBeVisible();
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: cardComponentId,
            componentType: 'card',
            shouldStartInteraction: false,
            value: undefined,
        });
    });
});
//# sourceMappingURL=Card.test.js.map