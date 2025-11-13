import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, it, jest, expect, beforeEach } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Link } from './Link';
import { setupDesignSystemEventProviderForTesting } from '../DesignSystemEventProvider';
import { setupSafexTesting } from '../utils/safex';
describe('Link', () => {
    const { setSafex } = setupSafexTesting();
    beforeEach(() => {
        setSafex({
            'databricks.fe.observability.defaultComponentView.typographyLink': true,
        });
        // Disable IntersectionObserver for useNotifyOnFirstView hook to trigger
        window.IntersectionObserver = undefined;
    });
    it('handles clicks', async () => {
        // Arrange
        const handleClick = jest.fn();
        render(_jsx(Link, { componentId: "TEST_LINK", onClick: handleClick, children: "LINK HERE" }));
        expect(handleClick).not.toHaveBeenCalled();
        // Act
        await userEvent.click(screen.getByText('LINK HERE'));
        // Assert
        expect(handleClick).toHaveBeenCalledTimes(1);
    });
    it('handles clicks with DesignSystemEventProvider', async () => {
        // Arrange
        const handleClick = jest.fn();
        const eventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(eventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Link, { componentId: "TEST_LINK", onClick: handleClick, children: "LINK HERE" }) }));
        await waitFor(() => {
            expect(screen.getByText('LINK HERE')).toBeVisible();
        });
        expect(handleClick).not.toHaveBeenCalled();
        expect(eventCallback).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: 'TEST_LINK',
            componentType: 'typography_link',
            shouldStartInteraction: false,
        });
        // Act
        await userEvent.click(screen.getByText('LINK HERE'));
        // Assert
        expect(handleClick).toHaveBeenCalledTimes(1);
        expect(eventCallback).toHaveBeenCalledTimes(2);
        expect(eventCallback).toHaveBeenNthCalledWith(2, {
            eventType: 'onClick',
            componentId: 'TEST_LINK',
            componentType: 'typography_link',
            shouldStartInteraction: false,
            value: undefined,
            event: expect.any(Object),
            isInteractionSubject: undefined,
        });
    });
});
//# sourceMappingURL=Link.test.js.map