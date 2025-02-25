import { render } from '@testing-library/react';

import { Alert, type AlertType } from '.';
import { DesignSystemEventProvider, DesignSystemEventProviderComponentSubTypes } from '../DesignSystemEventProvider';

describe('Alert', () => {
  window.HTMLElement.prototype.hasPointerCapture = jest.fn();

  it.each([
    { type: 'error', componentSubType: DesignSystemEventProviderComponentSubTypes.Error },
    { type: 'info', componentSubType: DesignSystemEventProviderComponentSubTypes.Info },
    { type: 'success', componentSubType: DesignSystemEventProviderComponentSubTypes.Success },
    { type: 'warning', componentSubType: DesignSystemEventProviderComponentSubTypes.Warning },
  ])(
    'callback for %s Alerts is being recorded and sub type is being passed through',
    async ({ type, componentSubType }) => {
      (window as any).IntersectionObserver = undefined;
      const mockUseOnEventCallback = jest.fn();

      render(
        <DesignSystemEventProvider callback={mockUseOnEventCallback}>
          <Alert
            type={type as AlertType}
            message={type}
            componentId={`test.internal-design-system-event-provider.${type}`}
          />
        </DesignSystemEventProvider>,
      );

      expect(mockUseOnEventCallback).toHaveBeenCalledWith({
        eventType: 'onView',
        componentId: `test.internal-design-system-event-provider.${type}`,
        componentType: 'alert',
        componentSubType,
        shouldStartInteraction: false,
        value: undefined,
      });
    },
  );
});
