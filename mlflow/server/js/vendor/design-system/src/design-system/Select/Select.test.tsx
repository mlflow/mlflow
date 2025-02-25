import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { Select } from './Select';
import { SelectContent } from './SelectContent';
import { SelectOption } from './SelectOption';
import { SelectTrigger } from './SelectTrigger';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Select', () => {
  const eventCallback = jest.fn();

  const Component = ({ valueHasNoPii, defaultValue }: { valueHasNoPii?: boolean; defaultValue?: string }) => {
    const [value, setValue] = useState<string>(defaultValue ?? '');
    const options = ['one', 'two', 'three'];

    return (
      <DesignSystemEventProvider callback={eventCallback}>
        <Select label="example filter" componentId="select_test" valueHasNoPii={valueHasNoPii} value={value}>
          <SelectTrigger />
          <SelectContent>
            {options.map((option) => (
              <SelectOption key={option} value={option} onChange={() => setValue(option)}>
                {option}
              </SelectOption>
            ))}
          </SelectContent>
        </Select>
      </DesignSystemEventProvider>
    );
  };

  it('emits value change event with value through underlying dialog combobox', async () => {
    render(<Component valueHasNoPii />);
    expect(eventCallback).not.toHaveBeenCalled();

    const select = screen.getByRole('combobox', { name: /example filter/ });
    await userEvent.click(select);
    expect(eventCallback).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('option', { name: 'one' }));
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'select_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: '["one"]',
    });

    await userEvent.click(select);
    await userEvent.click(screen.getByRole('option', { name: 'two' }));
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'select_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: '["two"]',
    });
  });

  it('emits value change event without value when valueHasNoPii is not set', async () => {
    render(<Component />);
    expect(eventCallback).not.toHaveBeenCalled();

    const select = screen.getByRole('combobox', { name: /example filter/ });
    await userEvent.click(select);
    expect(eventCallback).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('option', { name: 'one' }));
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'select_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('does not emit a value change event for the default value', async () => {
    render(<Component defaultValue="one" />);
    expect(eventCallback).not.toHaveBeenCalled();
  });
});
