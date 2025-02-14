import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { expect } from '@databricks/config-jest';

import { Input } from './Input';
import { Form } from '../../development/Form';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Input', () => {
  it('calls onChange when input changes', async () => {
    const onChange = jest.fn();
    render(<Input componentId="MY_TRACKING_ID" onChange={onChange} />);
    const input = 'abc';
    await userEvent.type(screen.getByRole('textbox'), input);
    expect(onChange).toHaveBeenCalledTimes(input.length);
  });

  it('calls onChange when input is cleared even if onClear is defined', async () => {
    const onChange = jest.fn();
    const onClear = jest.fn();
    render(<Input componentId="MY_TRACKING_ID" allowClear onChange={onChange} onClear={onClear} />);
    const input = 'abc';
    await userEvent.type(screen.getByRole('textbox'), input);
    await userEvent.clear(screen.getByRole('textbox'));
    expect(onChange).toHaveBeenCalledTimes(input.length + 1);
    expect(onClear).not.toHaveBeenCalled();
  });

  it('calls onClear when clear button is clicked', async () => {
    const onClear = jest.fn();
    const onChange = jest.fn();
    render(<Input componentId="MY_TRACKING_ID" allowClear onChange={onChange} onClear={onClear} />);
    await userEvent.click(screen.getByLabelText('close-circle'));
    expect(onClear).toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('calls onChange when clear button is clicked if onClear is not defined', async () => {
    const onChange = jest.fn();
    render(<Input componentId="MY_TRACKING_ID" allowClear onChange={onChange} />);
    await userEvent.click(screen.getByLabelText('close-circle'));
    expect(onChange).toHaveBeenCalled();
  });

  it('calls onChange with DesignSystemEventProvider', async () => {
    // Arrange
    const eventCallback = jest.fn();
    const onChange = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Input allowClear onChange={onChange} componentId="bestInputEver" />
      </DesignSystemEventProvider>,
    );
    expect(eventCallback).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();

    // input three letters and check onValueChange called
    await userEvent.type(screen.getByRole('textbox'), 'abc');
    await waitFor(() => expect(eventCallback).toBeCalledTimes(1), { timeout: 5000 });
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'bestInputEver',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });

    // input three more letters immediately.
    // only one call should be made until focus event fired
    await userEvent.type(screen.getByRole('textbox'), 'def');
    expect(eventCallback).toHaveBeenCalledTimes(1);

    // focusout and focus again to allow onValueChange to be called again
    fireEvent.focusOut(screen.getByRole('textbox'));
    fireEvent.focus(screen.getByRole('textbox'));

    // called onValueChange for inputing 'hij
    await userEvent.type(screen.getByRole('textbox'), 'hij');
    await waitFor(() => expect(eventCallback).toBeCalledTimes(2), { timeout: 5000 });
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onValueChange',
      componentId: 'bestInputEver',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });

    // focusout and focus again to allow onValueChange to be called again for clear button
    fireEvent.focusOut(screen.getByRole('textbox'));
    fireEvent.focus(screen.getByRole('textbox'));

    // click clear button
    await userEvent.click(screen.getByLabelText('close-circle'));
    await waitFor(() => expect(eventCallback).toBeCalledTimes(3), { timeout: 5000 });

    // called onValueChange for clicking clear button
    expect(eventCallback).toHaveBeenCalledTimes(3);
    expect(eventCallback).toHaveBeenNthCalledWith(3, {
      eventType: 'onValueChange',
      componentId: 'bestInputEver',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('calls onChange and onFocus with DesignSystemEventProvider', async () => {
    // Arrange
    const eventCallback = jest.fn();
    const onChange = jest.fn();
    const onFocus = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Input allowClear onChange={onChange} onFocus={onFocus} componentId="bestInputEver" />
      </DesignSystemEventProvider>,
    );
    expect(eventCallback).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
    expect(onFocus).not.toHaveBeenCalled();

    // input three letters and check onValueChange called
    await userEvent.type(screen.getByRole('textbox'), 'abc');
    await waitFor(() => expect(eventCallback).toBeCalledTimes(1), { timeout: 5000 });
    expect(onFocus).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'bestInputEver',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });

    // input three more letters immediately.
    // only one call should be made until focus event fired
    await userEvent.type(screen.getByRole('textbox'), 'def');
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(onFocus).toHaveBeenCalledTimes(1);

    // focusout and focus again to allow onValueChange to be called again
    fireEvent.focusOut(screen.getByRole('textbox'));
    fireEvent.focus(screen.getByRole('textbox'));

    // called onValueChange for inputing 'hij
    await userEvent.type(screen.getByRole('textbox'), 'hij');
    await waitFor(() => expect(eventCallback).toBeCalledTimes(2), { timeout: 5000 });
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onValueChange',
      componentId: 'bestInputEver',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });
    expect(onFocus).toHaveBeenCalledTimes(2);

    // focusout and focus again to allow onValueChange to be called again for clear button
    fireEvent.focusOut(screen.getByRole('textbox'));
    fireEvent.focus(screen.getByRole('textbox'));

    // click clear button
    await userEvent.click(screen.getByLabelText('close-circle'));
    await waitFor(() => expect(eventCallback).toBeCalledTimes(3), { timeout: 5000 });

    // called onValueChange for clicking clear button
    expect(eventCallback).toHaveBeenCalledTimes(3);
    expect(eventCallback).toHaveBeenNthCalledWith(3, {
      eventType: 'onValueChange',
      componentId: 'bestInputEver',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });
    expect(onFocus).toHaveBeenCalledTimes(3);
  });

  describe('Form submission', () => {
    it('submits form on input enter', async () => {
      const eventCallback = jest.fn();
      const onFormSubmit = jest.fn();
      render(
        <DesignSystemEventProvider callback={eventCallback}>
          <Form onSubmit={onFormSubmit} componentId="test-form">
            <Input componentId="bestInputEver" />
          </Form>
        </DesignSystemEventProvider>,
      );

      expect(onFormSubmit).not.toHaveBeenCalled();
      expect(eventCallback).not.toHaveBeenCalled();

      await userEvent.type(screen.getByRole('textbox'), '{Enter}');
      expect(onFormSubmit).toHaveBeenCalledTimes(1);
      expect(eventCallback).toHaveBeenCalledWith({
        eventType: 'onSubmit',
        componentId: 'test-form',
        componentType: 'form',
        referrerComponent: {
          id: 'bestInputEver',
          type: 'input',
        },
        shouldStartInteraction: true,
        value: undefined,
        event: expect.any(Object),
      });
    });

    it('submits form on input ctrl+enter', async () => {
      const eventCallback = jest.fn();
      const onFormSubmit = jest.fn();
      render(
        <DesignSystemEventProvider callback={eventCallback}>
          <Form onSubmit={onFormSubmit} componentId="test-form">
            <Input componentId="bestInputEver" />
          </Form>
        </DesignSystemEventProvider>,
      );

      expect(onFormSubmit).not.toHaveBeenCalled();
      expect(eventCallback).not.toHaveBeenCalled();

      const textbox = screen.getByRole('textbox');

      await userEvent.type(textbox, 'abc');
      await userEvent.type(textbox, '{Ctrl}');

      expect(onFormSubmit).not.toHaveBeenCalled();

      await userEvent.type(textbox, '{Ctrl>}{Enter}');
      expect(onFormSubmit).toBeCalledTimes(1);
      expect(eventCallback).toHaveBeenCalledWith({
        eventType: 'onSubmit',
        componentId: 'test-form',
        componentType: 'form',
        referrerComponent: {
          id: 'bestInputEver',
          type: 'input',
        },
        shouldStartInteraction: true,
        value: undefined,
        event: expect.any(Object),
      });
    });
  });
});
