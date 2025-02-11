import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ReactNode } from 'react';

import { ContextMenu } from './ContextMenu';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('ContextMenu', () => {
  const eventCallback = jest.fn();

  const CommonComponent = ({ children }: { children: ReactNode }) => (
    <DesignSystemEventProvider callback={eventCallback}>
      <ContextMenu.Root>
        <ContextMenu.Trigger>Trigger</ContextMenu.Trigger>
        <ContextMenu.Content>{children}</ContextMenu.Content>
      </ContextMenu.Root>
    </DesignSystemEventProvider>
  );

  it('emits click event for item', async () => {
    const onClickSpy = jest.fn();
    render(
      <CommonComponent>
        <ContextMenu.Item componentId="context_menu_item_test" onClick={onClickSpy}>
          Item
        </ContextMenu.Item>
      </CommonComponent>,
    );
    expect(onClickSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    fireEvent.contextMenu(screen.getByText('Trigger'));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Item' }));
    expect(onClickSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'context_menu_item_test',
      componentType: 'context_menu_item',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.any(Object),
      isInteractionSubject: undefined,
    });
  });

  it('does not emit click event for item when asChild is set', async () => {
    const onClickSpy = jest.fn();
    render(
      <CommonComponent>
        <ContextMenu.Item componentId="context_menu_item_test" onClick={onClickSpy} asChild>
          <button>Item</button>
        </ContextMenu.Item>
      </CommonComponent>,
    );
    expect(onClickSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    fireEvent.contextMenu(screen.getByText('Trigger'));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Item' }));
    expect(onClickSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).not.toHaveBeenCalled();
  });

  it('emits value change event for checkbox item', async () => {
    const onCheckedChangeSpy = jest.fn();
    render(
      <CommonComponent>
        <ContextMenu.CheckboxItem componentId="context_menu_checkbox_item_test" onCheckedChange={onCheckedChangeSpy}>
          Checkbox Item
        </ContextMenu.CheckboxItem>
      </CommonComponent>,
    );
    expect(onCheckedChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    fireEvent.contextMenu(screen.getByText('Trigger'));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Checkbox Item' }));
    expect(onCheckedChangeSpy).toHaveBeenCalledWith(true);
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'context_menu_checkbox_item_test',
      componentType: 'context_menu_checkbox_item',
      shouldStartInteraction: false,
      value: true,
    });
  });

  it('emits value change for radio group', async () => {
    const onValueChangeSpy = jest.fn();
    render(
      <CommonComponent>
        <ContextMenu.RadioGroup
          componentId="context_menu_radio_group_test"
          valueHasNoPii
          onValueChange={onValueChangeSpy}
        >
          <ContextMenu.RadioItem value="one">Radio Item 1</ContextMenu.RadioItem>
          <ContextMenu.RadioItem value="two">Radio Item 2</ContextMenu.RadioItem>
        </ContextMenu.RadioGroup>
      </CommonComponent>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    fireEvent.contextMenu(screen.getByText('Trigger'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 1' }));
    expect(onValueChangeSpy).toHaveBeenCalledWith('one');
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'context_menu_radio_group_test',
      componentType: 'context_menu_radio_group',
      shouldStartInteraction: false,
      value: 'one',
    });

    fireEvent.contextMenu(screen.getByText('Trigger'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 2' }));
    expect(onValueChangeSpy).toHaveBeenCalledWith('two');
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'context_menu_radio_group_test',
      componentType: 'context_menu_radio_group',
      shouldStartInteraction: false,
      value: 'two',
    });
  });

  it('emits value change event without value for radio group when valueHasNoPii is not set', async () => {
    const onValueChangeSpy = jest.fn();
    render(
      <CommonComponent>
        <ContextMenu.RadioGroup componentId="context_menu_radio_group_test" onValueChange={onValueChangeSpy}>
          <ContextMenu.RadioItem value="one">Radio Item 1</ContextMenu.RadioItem>
          <ContextMenu.RadioItem value="two">Radio Item 2</ContextMenu.RadioItem>
        </ContextMenu.RadioGroup>
      </CommonComponent>,
    );
    expect(onValueChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    fireEvent.contextMenu(screen.getByText('Trigger'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Radio Item 1' }));
    expect(onValueChangeSpy).toHaveBeenCalledWith('one');
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'context_menu_radio_group_test',
      componentType: 'context_menu_radio_group',
      shouldStartInteraction: false,
      value: undefined,
    });
  });
});
