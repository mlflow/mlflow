import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Checkbox } from './Checkbox';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Checkbox', () => {
  it('isChecked updates correctly', async () => {
    let isChecked = false;
    const changeHandlerFn = jest.fn();
    const changeHandler = (checked: boolean) => {
      isChecked = checked;
      changeHandlerFn();
    };
    render(
      <Checkbox
        componentId="codegen_design-system_src_design-system_checkbox_checkbox.test.tsx_16"
        isChecked={isChecked}
        onChange={changeHandler}
      >
        Basic checkbox
      </Checkbox>,
    );

    await userEvent.click(screen.getByRole('checkbox'));

    expect(changeHandlerFn).toHaveBeenCalledTimes(1);
    expect(isChecked).toBe(true);
  });

  it("isChecked doesn't update without onChange", async () => {
    // eslint-disable-next-line prefer-const
    let isChecked = false;
    render(
      <Checkbox
        componentId="codegen_design-system_src_design-system_checkbox_checkbox.test.tsx_30"
        isChecked={isChecked}
      >
        Basic checkbox
      </Checkbox>,
    );

    await userEvent.click(screen.getByRole('checkbox'));

    expect(isChecked).toBe(false);
  });

  it('handles changes with DesignSystemEventProvider', async () => {
    // Arrange
    const handleOnChange = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Checkbox componentId="bestCheckboxEver" onChange={handleOnChange} />
      </DesignSystemEventProvider>,
    );
    expect(handleOnChange).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    // Act
    await userEvent.click(screen.getByRole('checkbox'));

    // Assert
    expect(handleOnChange).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'bestCheckboxEver',
      componentType: 'checkbox',
      shouldStartInteraction: false,
      value: true,
    });
  });
});
