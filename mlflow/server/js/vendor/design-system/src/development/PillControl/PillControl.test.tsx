import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { PillControl } from '.';
import { DesignSystemEventProvider } from '../../design-system/DesignSystemEventProvider/DesignSystemEventProvider';

describe('PillControl', () => {
  it('renders pills as a radio-group', async () => {
    const onValueChangeSpy = jest.fn();

    render(
      <PillControl.Root componentId="test" onValueChange={onValueChangeSpy}>
        <PillControl.Item value="a">A</PillControl.Item>
        <PillControl.Item value="b">B</PillControl.Item>
        <PillControl.Item value="c">C</PillControl.Item>
        <PillControl.Item value="d" disabled>
          D
        </PillControl.Item>
      </PillControl.Root>,
    );

    const radioGroup = screen.getByRole('radiogroup');
    const pills = within(radioGroup).getAllByRole('radio');

    // Ensure pills are rendered
    expect(pills).toHaveLength(4);

    // Ensure pills are interactive
    await userEvent.click(pills[1]);
    expect(onValueChangeSpy).toHaveBeenCalledWith('b');

    // Ensure disabled items can not have interaction
    expect(pills[3]).toBeDisabled();
  });

  it('emits value change events without value', async () => {
    const onValueChangeSpy = jest.fn();
    const eventCallback = jest.fn();

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <PillControl.Root componentId="test" onValueChange={onValueChangeSpy}>
          <PillControl.Item value="a">A</PillControl.Item>
          <PillControl.Item value="b">B</PillControl.Item>
          <PillControl.Item value="c">C</PillControl.Item>
          <PillControl.Item value="d" disabled>
            D
          </PillControl.Item>
        </PillControl.Root>
      </DesignSystemEventProvider>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const radioGroup = screen.getByRole('radiogroup');
    const pills = within(radioGroup).getAllByRole('radio');

    // Ensure pills are rendered
    expect(pills).toHaveLength(4);

    // Ensure pills are interactive
    await userEvent.click(pills[1]);
    expect(onValueChangeSpy).toHaveBeenCalledWith('b');
    expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'test',
      componentType: 'pill_control',
      shouldStartInteraction: false,
      value: undefined,
    });

    // Ensure disabled items can not have interaction
    expect(pills[3]).toBeDisabled();
  });

  it('emits value change events with value', async () => {
    const onValueChangeSpy = jest.fn();
    const eventCallback = jest.fn();

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <PillControl.Root componentId="test" valueHasNoPii onValueChange={onValueChangeSpy}>
          <PillControl.Item value="a">A</PillControl.Item>
          <PillControl.Item value="b">B</PillControl.Item>
          <PillControl.Item value="c">C</PillControl.Item>
          <PillControl.Item value="d" disabled>
            D
          </PillControl.Item>
        </PillControl.Root>
      </DesignSystemEventProvider>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const radioGroup = screen.getByRole('radiogroup');
    const pills = within(radioGroup).getAllByRole('radio');

    // Ensure pills are rendered
    expect(pills).toHaveLength(4);

    // Ensure pills are interactive
    await userEvent.click(pills[1]);
    expect(onValueChangeSpy).toHaveBeenCalledWith('b');
    expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'test',
      componentType: 'pill_control',
      shouldStartInteraction: false,
      value: 'b',
    });

    // Ensure disabled items can not have interaction
    expect(pills[3]).toBeDisabled();
  });
});
