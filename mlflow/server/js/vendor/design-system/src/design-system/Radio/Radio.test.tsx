import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Radio } from '.';
import { Form, useFormContext } from '../../development/Form/Form';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider/DesignSystemEventProvider';

describe('Radio', () => {
  it('emits value change events without value', async () => {
    const onValueChangeSpy = jest.fn();
    const eventCallback = jest.fn();

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Radio.Group name="test" componentId="test" onChange={onValueChangeSpy}>
          <Radio value="a">A</Radio>
          <Radio value="b">B</Radio>
          <Radio value="c">C</Radio>
          <Radio value="d" disabled>
            D
          </Radio>
        </Radio.Group>
      </DesignSystemEventProvider>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const radio = screen.getByText('B');

    // Ensure pills are interactive
    await userEvent.click(radio);
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
          <Radio value="a">A</Radio>
          <Radio value="b">B</Radio>
          <Radio value="c">C</Radio>
          <Radio value="d" disabled>
            D
          </Radio>
        </Radio.Group>
      </DesignSystemEventProvider>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const radio = screen.getByText('B');

    // Ensure pills are interactive
    await userEvent.click(radio);
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

  it('works with form submission', async () => {
    // Arrange
    const handleSubmit = jest.fn().mockResolvedValue(undefined);
    const eventCallback = jest.fn();

    const TestComponent = () => {
      const formContext = useFormContext();
      return (
        <Radio.Group
          componentId="test-radio"
          onChange={(e) => {
            formContext.formRef?.current?.requestSubmit();
          }}
          name="test-radio"
        >
          <Radio value={1}>1</Radio>
          <Radio value={2}>2</Radio>
        </Radio.Group>
      );
    };

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Form componentId="eventForm" onSubmit={handleSubmit}>
          <TestComponent />
        </Form>
      </DesignSystemEventProvider>,
    );

    // Act
    const radio = screen.getByText('2');
    await userEvent.click(radio);

    // Assert
    expect(handleSubmit).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'test-radio',
      componentType: 'radio_group',
      shouldStartInteraction: false,
    });
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onSubmit',
      componentId: 'eventForm',
      componentType: 'form',
      shouldStartInteraction: true,
      event: expect.anything(),
      referrerComponent: {
        id: 'test-radio',
        type: 'radio_group',
      },
    });
  });

  it('emits value change events for standalone radio without RadioGroup', async () => {
    const onChangeSpy = jest.fn();
    const eventCallback = jest.fn();

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Radio componentId="standalone-radio" value="test" onChange={onChangeSpy}>
          Test Radio
        </Radio>
      </DesignSystemEventProvider>,
    );

    const radio = screen.getByText('Test Radio');
    await userEvent.click(radio);

    expect(onChangeSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'standalone-radio',
      componentType: 'radio',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('does not emit value change events for radio within RadioGroup', async () => {
    const onChangeSpy = jest.fn();
    const groupOnChangeSpy = jest.fn();
    const eventCallback = jest.fn();

    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Radio.Group name="test-group" componentId="test-group" onChange={groupOnChangeSpy}>
          <Radio componentId="standalone-radio" value="test" onChange={onChangeSpy}>
            Test Radio
          </Radio>
        </Radio.Group>
      </DesignSystemEventProvider>,
    );

    const radio = screen.getByText('Test Radio');
    await userEvent.click(radio);

    expect(onChangeSpy).toHaveBeenCalledTimes(1);
    expect(groupOnChangeSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'test-group',
      componentType: 'radio_group',
      shouldStartInteraction: false,
      value: undefined,
    });
  });
});
