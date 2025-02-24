import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import {
  DesignSystemEventProvider,
  DesignSystemEventProviderAnalyticsEventTypes,
  DesignSystemEventProviderComponentTypes,
  useDesignSystemEventComponentCallbacks,
} from './DesignSystemEventProvider';

const MockChildComponent: React.FC<{
  analyticsEvents?: ReadonlyArray<DesignSystemEventProviderAnalyticsEventTypes>;
}> = (props) => {
  const button1Context = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: 'button1',
    analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  });
  const button2Context = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Button,
    componentId: 'button2',
    analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
  });
  const input1Context = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Input,
    componentId: 'input1',
    analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
  });
  const formContext = useDesignSystemEventComponentCallbacks({
    componentType: DesignSystemEventProviderComponentTypes.Form,
    componentId: 'form1',
    analyticsEvents: props.analyticsEvents ?? [DesignSystemEventProviderAnalyticsEventTypes.OnSubmit],
  });

  button1Context.onView();
  button2Context.onView();
  input1Context.onView();

  return (
    <div>
      <button onClick={button1Context.onClick}>Click me</button>
      <button onClick={button2Context.onClick}>View me</button>
      <input onChange={() => input1Context.onValueChange()} />
      <form
        onSubmit={(e) =>
          formContext.onSubmit(e, { type: DesignSystemEventProviderComponentTypes.Button, id: 'submit-button' })
        }
      >
        <button type="submit">Submit</button>
      </form>
    </div>
  );
};

describe('DesignSystemEventProvider', () => {
  it('provides onClick callback', async () => {
    const mockUseOnEventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={mockUseOnEventCallback}>
        <MockChildComponent />
      </DesignSystemEventProvider>,
    );

    expect(screen.getByText('Click me')).toBeInTheDocument();
    expect(screen.getByText('View me')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Click me'));
    await userEvent.click(screen.getByText('View me'));

    expect(mockUseOnEventCallback).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'button1',
      componentType: 'button',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
    expect(mockUseOnEventCallback).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'button2',
      componentType: 'button',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
  });

  it('provides onView callback', () => {
    const mockUseOnEventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={mockUseOnEventCallback}>
        <MockChildComponent />
      </DesignSystemEventProvider>,
    );

    expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: 'button1',
      componentType: 'button',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
    expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: 'button2',
      componentType: 'button',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
    expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: 'input1',
      componentType: 'input',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
  });

  it('provides onValueChange callback', async () => {
    const mockUseOnEventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={mockUseOnEventCallback}>
        <MockChildComponent />
      </DesignSystemEventProvider>,
    );

    expect(screen.getByRole('textbox')).toBeInTheDocument();

    await userEvent.type(screen.getByRole('textbox'), 'test');

    expect(mockUseOnEventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'input1',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('should use the nearest context', async () => {
    const mockUseOnEventCallbackFar = jest.fn();
    const mockUseOnEventCallbackNear = jest.fn();
    render(
      <DesignSystemEventProvider callback={mockUseOnEventCallbackFar}>
        <DesignSystemEventProvider callback={mockUseOnEventCallbackNear}>
          <MockChildComponent />
        </DesignSystemEventProvider>
      </DesignSystemEventProvider>,
    );

    expect(screen.getByText('Click me')).toBeInTheDocument();
    expect(screen.getByText('View me')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Click me'));
    await userEvent.click(screen.getByText('View me'));

    expect(mockUseOnEventCallbackFar).not.toHaveBeenCalled();
    expect(mockUseOnEventCallbackNear).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'button1',
      componentType: 'button',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
    expect(mockUseOnEventCallbackNear).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'button2',
      componentType: 'button',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
  });

  it('handles absence of callbacks', async () => {
    render(<MockChildComponent />);

    expect(screen.getByText('Click me')).toBeInTheDocument();
    expect(screen.getByText('View me')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Click me'));
    await userEvent.type(screen.getByRole('textbox'), 'test');

    expect(screen.getByText('Click me')).toBeInTheDocument();
    expect(screen.getByText('View me')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toHaveValue('test');
  });

  it('Overriding analyticsEvents behaves as expected', async () => {
    const mockUseOnEventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={mockUseOnEventCallback}>
        <MockChildComponent analyticsEvents={[DesignSystemEventProviderAnalyticsEventTypes.OnView]} />
      </DesignSystemEventProvider>,
    );

    await userEvent.click(screen.getByText('Click me'));

    expect(mockUseOnEventCallback).not.toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'button1',
      componentType: 'button',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
    });
    expect(mockUseOnEventCallback).toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: 'button1',
      componentType: 'button',
      shouldStartInteraction: false,
      value: undefined,
    });
    expect(mockUseOnEventCallback).toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: 'button2',
      componentType: 'button',
      shouldStartInteraction: false,
      value: undefined,
    });
    expect(mockUseOnEventCallback).toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: 'input1',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('provides onSubmit callback', async () => {
    const mockUseOnEventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={mockUseOnEventCallback}>
        <MockChildComponent />
      </DesignSystemEventProvider>,
    );

    expect(screen.getByText('Submit')).toBeInTheDocument();

    await userEvent.click(screen.getByText('Submit'));

    expect(mockUseOnEventCallback).toHaveBeenCalledWith({
      eventType: 'onSubmit',
      componentId: 'form1',
      componentType: 'form',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.anything(),
      referrerComponent: { type: 'button', id: 'submit-button' },
    });
  });
});
