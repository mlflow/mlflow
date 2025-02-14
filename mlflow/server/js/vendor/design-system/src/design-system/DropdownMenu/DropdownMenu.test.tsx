import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DropdownMenu } from '.';
import { openDropdownMenu } from '../../test-utils/rtl';
import { Button } from '../Button';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider/DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider';

describe('DropdownMenu', function () {
  function renderComponent() {
    return render(
      <DesignSystemProvider>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <Button
              componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15"
              data-testid="test-menubutton"
            >
              Default
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="start">
            <DropdownMenu.Sub>
              <DropdownMenu.SubTrigger>Option 1</DropdownMenu.SubTrigger>
              <DropdownMenu.SubContent>
                <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_27">
                  Option 1a
                </DropdownMenu.Item>
                <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_28">
                  Option 1b
                </DropdownMenu.Item>
                <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_29">
                  Option 1c
                </DropdownMenu.Item>
              </DropdownMenu.SubContent>
            </DropdownMenu.Sub>
            <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_32">
              Option 2
            </DropdownMenu.Item>
            <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_33">
              Option 3
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </DesignSystemProvider>,
    );
  }

  function renderDisabledComponentWithTooltip(onClick: () => void) {
    return render(
      <DesignSystemProvider>
        <DropdownMenu.Root>
          <DropdownMenu.Trigger asChild>
            <Button
              componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_39"
              data-testid="test-menubutton"
            >
              Default
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="start">
            <DropdownMenu.Sub>
              <DropdownMenu.SubTrigger>Option 1</DropdownMenu.SubTrigger>
              <DropdownMenu.SubContent>
                <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_56">
                  Option 1a
                </DropdownMenu.Item>
                <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_57">
                  Option 1b
                </DropdownMenu.Item>
                <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_58">
                  Option 1c
                </DropdownMenu.Item>
              </DropdownMenu.SubContent>
            </DropdownMenu.Sub>
            <DropdownMenu.Item
              componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_61"
              data-testid="test-disableditem"
              disabled
              disabledReason="Option disabled reason"
              onClick={onClick}
            >
              Option 2
            </DropdownMenu.Item>
            <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_69">
              Option 3
            </DropdownMenu.Item>
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      </DesignSystemProvider>,
    );
  }

  // This is a trivial re-test of Radix's tests, but is provided as an
  // example of how to test the DropdownMenu component.
  it('renders proper number of menuitem(s) with proper text', async () => {
    renderComponent();

    await userEvent.click(screen.getByTestId('test-menubutton'));

    expect(screen.queryAllByRole('menuitem')).toHaveLength(3);
    expect(screen.queryByText('Option 1')).not.toBeNull();
    expect(screen.queryByText('Option 2')).not.toBeNull();
    expect(screen.queryByText('Option 3')).not.toBeNull();

    // This is known to not work correctly in `@testing-library/user-event <=13.5.0.
    await userEvent.click(screen.getByText('Option 1'));

    expect(screen.queryAllByRole('menuitem')).toHaveLength(6);

    await userEvent.keyboard('{Escape}');

    expect(screen.queryAllByRole('menuitem')).toHaveLength(0);
  });

  it("doesn't trigger click on tooltip event when disabled", async () => {
    const onClick = jest.fn();
    renderDisabledComponentWithTooltip(onClick);

    openDropdownMenu(screen.getByTestId('test-menubutton'));

    await waitFor(() => {
      expect(screen.getByTestId('test-disableditem')).toBeVisible();
    });

    userEvent.click(screen.getByTestId('test-disableditem').querySelector('span')!);

    expect(onClick).not.toHaveBeenCalled();
  });

  it('emits analytics events for item', async () => {
    const handleClick = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <DesignSystemProvider>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15"
                data-testid="test-menubutton"
              >
                Default
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="start">
              <DropdownMenu.Item onClick={handleClick} componentId="OPTION_A_TEST">
                Option A
              </DropdownMenu.Item>
              <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_132">
                Option B
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </DesignSystemProvider>
      </DesignSystemEventProvider>,
    );
    expect(handleClick).not.toBeCalled();
    expect(eventCallback).not.toBeCalled();

    await userEvent.click(screen.getByTestId('test-menubutton'));
    expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
    await userEvent.click(screen.getByText('Option A'));

    expect(handleClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onClick',
      componentId: 'OPTION_A_TEST',
      componentType: 'dropdown_menu_item',
      shouldStartInteraction: true,
      value: undefined,
      event: expect.any(Object),
      isInteractionSubject: undefined,
    });
  });

  it('does not emit analytics events for menu item with asChild set', async () => {
    const handleClick = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <DesignSystemProvider>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15"
                data-testid="test-menubutton"
              >
                Default
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="start">
              <DropdownMenu.Item onClick={handleClick} componentId="OPTION_A_TEST" asChild>
                <Button componentId="OPTION_A_TEST_CHILD">Option A</Button>
              </DropdownMenu.Item>
              <DropdownMenu.Item componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_174">
                Option B
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </DesignSystemProvider>
      </DesignSystemEventProvider>,
    );
    expect(handleClick).not.toBeCalled();
    expect(eventCallback).not.toBeCalled();

    await userEvent.click(screen.getByTestId('test-menubutton'));
    expect(screen.queryAllByRole('menuitem')).toHaveLength(2);
    await userEvent.click(screen.getByText('Option A'));

    expect(handleClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onClick',
      componentId: 'OPTION_A_TEST_CHILD',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });
  });

  it('emits analytics events for checkbox', async () => {
    const handleClick = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <DesignSystemProvider>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15"
                data-testid="test-menubutton"
              >
                Default
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="start">
              <DropdownMenu.CheckboxItem onCheckedChange={handleClick} componentId="OPTION_A_TEST">
                Option A
              </DropdownMenu.CheckboxItem>
              <DropdownMenu.CheckboxItem componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_218">
                Option B
              </DropdownMenu.CheckboxItem>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </DesignSystemProvider>
      </DesignSystemEventProvider>,
    );
    expect(handleClick).not.toBeCalled();
    expect(eventCallback).not.toBeCalled();

    await userEvent.click(screen.getByTestId('test-menubutton'));
    await userEvent.click(screen.getByText('Option A'));

    expect(handleClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'OPTION_A_TEST',
      componentType: 'dropdown_menu_checkbox_item',
      shouldStartInteraction: false,
      value: true,
    });
  });

  it('emits analytics events for radio group', async () => {
    const handleClick = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <DesignSystemProvider>
          <DropdownMenu.Root>
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="codegen_design-system_src_design-system_dropdownmenu_dropdownmenu.test.tsx_15"
                data-testid="test-menubutton"
              >
                Default
              </Button>
            </DropdownMenu.Trigger>
            <DropdownMenu.Content align="start">
              <DropdownMenu.RadioGroup componentId="OPTION_RADIO_GROUP">
                <DropdownMenu.RadioItem value="A" onClick={handleClick}>
                  Option A
                </DropdownMenu.RadioItem>
                <DropdownMenu.RadioItem value="B">Option B</DropdownMenu.RadioItem>
              </DropdownMenu.RadioGroup>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </DesignSystemProvider>
      </DesignSystemEventProvider>,
    );
    expect(handleClick).not.toBeCalled();
    expect(eventCallback).not.toBeCalled();

    await userEvent.click(screen.getByTestId('test-menubutton'));
    await userEvent.click(screen.getByText('Option A'));

    expect(handleClick).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onValueChange',
      componentId: 'OPTION_RADIO_GROUP',
      componentType: 'dropdown_menu_radio_group',
      shouldStartInteraction: false,
      value: undefined,
    });
  });
});
