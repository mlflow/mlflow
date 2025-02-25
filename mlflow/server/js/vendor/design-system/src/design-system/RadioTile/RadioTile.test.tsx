import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { RadioTile } from '.';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { Radio } from '../Radio';

describe('RadioTile', () => {
  it('emits value change events without value', async () => {
    const onValueChangeSpy = jest.fn();
    const eventCallback = jest.fn();

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Radio.Group name="test" componentId="test" onChange={onValueChangeSpy}>
          <RadioTile value="a">A</RadioTile>
          <RadioTile value="b">B</RadioTile>
          <RadioTile value="c">C</RadioTile>
          <RadioTile value="d" disabled>
            D
          </RadioTile>
        </Radio.Group>
      </DesignSystemEventProvider>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const radioTile = screen.getByRole('radio', { name: 'B' });

    await userEvent.click(radioTile);
    expect(onValueChangeSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({
          value: 'b',
        }),
      }),
    );
    expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'test',
      componentType: 'radio_group',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('emits value change events with value', async () => {
    const onValueChangeSpy = jest.fn();
    const eventCallback = jest.fn();

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Radio.Group name="test" componentId="test" onChange={onValueChangeSpy} valueHasNoPii>
          <RadioTile value="a">A</RadioTile>
          <RadioTile value="b">B</RadioTile>
          <RadioTile value="c">C</RadioTile>
          <RadioTile value="d" disabled>
            D
          </RadioTile>
        </Radio.Group>
      </DesignSystemEventProvider>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const radioTile = screen.getByRole('radio', { name: 'B' });

    await userEvent.click(radioTile);
    expect(onValueChangeSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        target: expect.objectContaining({
          value: 'b',
        }),
      }),
    );
    expect(onValueChangeSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'test',
      componentType: 'radio_group',
      shouldStartInteraction: false,
      value: 'b',
    });
  });
});
