import { jsx as _jsx } from "@emotion/react/jsx-runtime";
import { describe, jest, it, expect } from '@jest/globals';
import { render } from '@testing-library/react';
import { Alert } from '.';
import { DesignSystemEventProviderComponentSubTypes, setupDesignSystemEventProviderForTesting, } from '../DesignSystemEventProvider';
describe('Alert', () => {
    window.HTMLElement.prototype.hasPointerCapture = jest.fn();
    it.each([
        { type: 'error', componentSubType: DesignSystemEventProviderComponentSubTypes.Error },
        { type: 'info', componentSubType: DesignSystemEventProviderComponentSubTypes.Info },
        { type: 'success', componentSubType: DesignSystemEventProviderComponentSubTypes.Success },
        { type: 'warning', componentSubType: DesignSystemEventProviderComponentSubTypes.Warning },
    ])('callback for %s Alerts is being recorded and sub type is being passed through', async ({ type, componentSubType }) => {
        window.IntersectionObserver = undefined;
        const mockUseOnEventCallback = jest.fn();
        const { DesignSystemEventProviderForTest } = setupDesignSystemEventProviderForTesting(mockUseOnEventCallback);
        render(_jsx(DesignSystemEventProviderForTest, { children: _jsx(Alert, { type: type, message: type, componentId: `test.internal-design-system-event-provider.${type}` }) }));
        expect(mockUseOnEventCallback).toHaveBeenCalledWith({
            eventType: 'onView',
            componentId: `test.internal-design-system-event-provider.${type}`,
            componentType: 'alert',
            componentSubType,
            shouldStartInteraction: false,
            value: undefined,
        });
    });
});
//# sourceMappingURL=Alert.test.js.map