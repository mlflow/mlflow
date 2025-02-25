import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { expect } from '@databricks/config-jest';

import { TextArea } from './TextArea';
import { Form } from '../../development';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('TextArea', () => {
  it('calls onChange when input changes', async () => {
    const onChange = jest.fn();
    render(<TextArea componentId="MY_TRACKING_ID" onChange={onChange} />);
    const input = 'abc';
    await userEvent.type(screen.getByRole('textbox'), input);
    expect(onChange).toHaveBeenCalledTimes(input.length);
  });

  it('calls onChange with DesignSystemEventProvider', async () => {
    // Arrange
    const eventCallback = jest.fn();
    const onChange = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <TextArea onChange={onChange} componentId="MY_TRACKING_ID" />
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
      componentId: 'MY_TRACKING_ID',
      componentType: 'text_area',
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
      componentId: 'MY_TRACKING_ID',
      componentType: 'text_area',
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
        <TextArea allowClear onChange={onChange} onFocus={onFocus} componentId="MY_TRACKING_ID" />
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
      componentId: 'MY_TRACKING_ID',
      componentType: 'text_area',
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
      componentId: 'MY_TRACKING_ID',
      componentType: 'text_area',
      shouldStartInteraction: false,
      value: undefined,
    });
    expect(onFocus).toHaveBeenCalledTimes(2);
  });

  describe('Form submission', () => {
    it('does not submit form when allowFormSubmitOnEnter is false', async () => {
      // Arrange
      const onSubmit = jest.fn();
      render(
        <Form onSubmit={onSubmit} componentId="test-form">
          <TextArea allowClear componentId="MY_TRACKING_ID" />
        </Form>,
      );

      // Act
      await userEvent.type(screen.getByRole('textbox'), '{Enter}');

      // Assert
      expect(onSubmit).not.toHaveBeenCalled();
    });

    it('trigger submit with enter when allowFormSubmitOnEnter is true', async () => {
      // Arrange
      const onSubmit = jest.fn();
      render(
        <Form onSubmit={onSubmit} componentId="test-form">
          <TextArea allowClear allowFormSubmitOnEnter componentId="MY_TRACKING_ID" />
        </Form>,
      );

      // Act
      await userEvent.type(screen.getByRole('textbox'), '{Enter}');

      // Assert
      expect(onSubmit).toBeCalledTimes(1);
    });

    it('triggers submit with platform enter', async () => {
      // Arrange
      const eventCallback = jest.fn();
      const onSubmit = jest.fn();
      render(
        <DesignSystemEventProvider callback={eventCallback}>
          <Form onSubmit={onSubmit} componentId="test-form">
            <TextArea allowClear componentId="MY_TRACKING_ID" />
          </Form>
        </DesignSystemEventProvider>,
      );
      expect(eventCallback).not.toHaveBeenCalled();
      expect(onSubmit).not.toHaveBeenCalled();

      // input three letters and check onValueChange called
      const textbox = screen.getByRole('textbox');
      await userEvent.type(textbox, 'abc');
      await waitFor(() => expect(eventCallback).toBeCalledTimes(1), { timeout: 5000 });

      // Control by itself is not enough
      await userEvent.type(textbox, '{Control}');
      expect(onSubmit).not.toHaveBeenCalled();

      // But Control+Enter submits
      await userEvent.type(textbox, '{Control>}{Enter}');
      expect(onSubmit).toBeCalledTimes(1);
    });
  });
});
