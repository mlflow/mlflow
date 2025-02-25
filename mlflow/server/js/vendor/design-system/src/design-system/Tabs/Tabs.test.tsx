import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { Tabs } from '.';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider/DesignSystemEventProvider';

describe('Tabs', () => {
  const eventCallback = jest.fn();

  const UncontrolledTabs = ({ valueHasNoPii }: { valueHasNoPii?: boolean }) => (
    <Tabs.Root componentId="TABS_TEST" defaultValue="tab1" valueHasNoPii={valueHasNoPii}>
      <Tabs.List>
        <Tabs.Trigger value="tab1">Tab 1</Tabs.Trigger>
        <Tabs.Trigger value="tab2">Tab 2</Tabs.Trigger>
        <Tabs.Trigger value="tab3">Tab 3</Tabs.Trigger>
      </Tabs.List>
      <Tabs.Content value="tab1">Tab 1 Content</Tabs.Content>
      <Tabs.Content value="tab2">Tab 2 Content</Tabs.Content>
      <Tabs.Content value="tab3">Tab 3 Content</Tabs.Content>
    </Tabs.Root>
  );

  const ControlledTabs = ({ valueHasNoPii }: { valueHasNoPii?: boolean }) => {
    const [tabs, setTabs] = useState([
      { value: 'tab1', title: 'Tab 1', content: 'Tab 1 Content' },
      { value: 'tab2', title: 'Tab 2', content: 'Tab 2 Content' },
    ]);
    const [activeTab, setActiveTab] = useState('tab1');
    const [nextTabNumber, setNextTabNumber] = useState(3);

    return (
      <Tabs.Root componentId="TABS_TEST" value={activeTab} onValueChange={setActiveTab} valueHasNoPii={valueHasNoPii}>
        <Tabs.List
          addButtonProps={{
            onClick: () => {
              const newTab = {
                value: `tab${nextTabNumber}`,
                title: `Tab ${nextTabNumber}`,
                content: `Tab ${nextTabNumber} Content`,
              };
              setTabs(tabs.concat(newTab));
              setActiveTab(newTab.value);
              setNextTabNumber(nextTabNumber + 1);
            },
          }}
        >
          {tabs.map((tab) => (
            <Tabs.Trigger
              key={tab.value}
              value={tab.value}
              onClose={(value) => {
                const newTabs = tabs.filter((tab) => tab.value !== value);
                setTabs(newTabs);
                if (activeTab === value && newTabs.length > 0) {
                  setActiveTab(newTabs[0].value);
                }
              }}
            >
              {tab.title}
            </Tabs.Trigger>
          ))}
        </Tabs.List>
        {tabs.map((tab) => (
          <Tabs.Content key={tab.value} value={tab.value}>
            {tab.content}
          </Tabs.Content>
        ))}
      </Tabs.Root>
    );
  };

  it('renders a set of tabs', async () => {
    render(<UncontrolledTabs />);

    const tabList = screen.getByRole('tablist');
    const tabs = within(tabList).getAllByRole('tab');

    expect(tabs).toHaveLength(3);
    expect(tabs[0]).toHaveTextContent('Tab 1');
    expect(tabs[1]).toHaveTextContent('Tab 2');
    expect(tabs[2]).toHaveTextContent('Tab 3');

    expect(screen.getByText('Tab 1 Content')).toBeInTheDocument();
    expect(screen.queryByText('Tab 2 Content')).not.toBeInTheDocument();

    await userEvent.click(tabs[1]);
    expect(screen.getByText('Tab 2 Content')).toBeInTheDocument();
    expect(screen.queryByText('Tab 1 Content')).not.toBeInTheDocument();
  });

  it('disabled tabs cannot be selected', async () => {
    render(
      <Tabs.Root componentId="TABS_TEST" defaultValue="tab1">
        <Tabs.List>
          <Tabs.Trigger value="tab1">Tab 1</Tabs.Trigger>
          <Tabs.Trigger value="tab2" disabled>
            Tab 2
          </Tabs.Trigger>
        </Tabs.List>
        <Tabs.Content value="tab1">Tab 1 Content</Tabs.Content>
        <Tabs.Content value="tab2">Tab 2 Content</Tabs.Content>
      </Tabs.Root>,
    );

    const tabList = screen.getByRole('tablist');
    const tabs = within(tabList).getAllByRole('tab');

    expect(tabs[1]).toBeDisabled();
  });

  it('tabs can be added and closed', async () => {
    render(<ControlledTabs />);

    expect(screen.queryByText('Tab 1 Content')).toBeInTheDocument();

    const addButton = screen.getByRole('button', { name: 'Add tab' });
    expect(addButton).toBeInTheDocument();

    await userEvent.click(addButton);
    let tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
    expect(tabs).toHaveLength(3);
    expect(screen.getByText('Tab 3 Content')).toBeInTheDocument();
    expect(screen.queryByText('Tab 1 Content')).not.toBeInTheDocument();

    act(() => {
      tabs[2].focus();
    });
    await userEvent.keyboard('{Delete}');
    tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
    expect(tabs).toHaveLength(2);
    expect(screen.getByText('Tab 1 Content')).toBeInTheDocument();
    expect(screen.queryByText('Tab 3 Content')).not.toBeInTheDocument();

    await userEvent.pointer({ target: tabs[0], keys: '[MouseMiddle]' });
    tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
    expect(tabs).toHaveLength(1);
    expect(screen.getByText('Tab 2 Content')).toBeInTheDocument();
    expect(screen.queryByText('Tab 1 Content')).not.toBeInTheDocument();
  });

  describe('Analytics Events', () => {
    it('uncontrolled tabs emit value change events without value', async () => {
      render(
        <DesignSystemEventProvider callback={eventCallback}>
          <UncontrolledTabs />
        </DesignSystemEventProvider>,
      );
      expect(eventCallback).not.toHaveBeenCalled();

      const tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
      await userEvent.click(tabs[1]);
      expect(eventCallback).toHaveBeenCalledWith({
        eventType: 'onValueChange',
        componentId: 'TABS_TEST',
        componentType: 'tabs',
        shouldStartInteraction: true,
        value: undefined,
      });
    });

    it('uncontrolled tabs emit value change events with value', async () => {
      render(
        <DesignSystemEventProvider callback={eventCallback}>
          <UncontrolledTabs valueHasNoPii />
        </DesignSystemEventProvider>,
      );
      expect(eventCallback).not.toHaveBeenCalled();

      const tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
      await userEvent.click(tabs[1]);
      expect(eventCallback).toHaveBeenCalledWith({
        eventType: 'onValueChange',
        componentId: 'TABS_TEST',
        componentType: 'tabs',
        shouldStartInteraction: true,
        value: 'tab2',
      });
    });

    it('controlled tabs emit value change and on click events', async () => {
      render(
        <DesignSystemEventProvider callback={eventCallback}>
          <ControlledTabs valueHasNoPii />
        </DesignSystemEventProvider>,
      );
      expect(eventCallback).not.toHaveBeenCalled();

      const addButton = screen.getByRole('button', { name: 'Add tab' });
      await userEvent.click(addButton);
      expect(eventCallback).toHaveBeenCalledTimes(1);
      expect(eventCallback).toHaveBeenNthCalledWith(1, {
        eventType: 'onClick',
        componentId: 'TABS_TEST.add_tab',
        componentType: 'button',
        shouldStartInteraction: true,
        isInteractionSubject: true,
        value: undefined,
        event: expect.any(Object),
      });

      let tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
      await userEvent.click(tabs[0]);
      expect(eventCallback).toHaveBeenCalledTimes(2);
      expect(eventCallback).toHaveBeenNthCalledWith(2, {
        eventType: 'onValueChange',
        componentId: 'TABS_TEST',
        componentType: 'tabs',
        shouldStartInteraction: true,
        value: 'tab1',
      });

      const closeIcon = within(tabs[0]).getByLabelText('Press delete to close the tab');
      await userEvent.click(closeIcon);
      expect(eventCallback).toHaveBeenCalledTimes(3);
      expect(eventCallback).toHaveBeenNthCalledWith(3, {
        eventType: 'onClick',
        componentId: 'TABS_TEST.close_tab',
        componentType: 'button',
        shouldStartInteraction: true,
        value: undefined,
        event: expect.any(Object),
        isInteractionSubject: undefined,
      });

      tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
      act(() => {
        tabs[0].focus();
      });
      await userEvent.keyboard('{Delete}');
      expect(eventCallback).toHaveBeenCalledTimes(4);
      expect(eventCallback).toHaveBeenNthCalledWith(4, {
        eventType: 'onClick',
        componentId: 'TABS_TEST.close_tab',
        componentType: 'button',
        shouldStartInteraction: true,
        value: undefined,
        event: expect.any(Object),
        isInteractionSubject: undefined,
      });

      tabs = within(screen.getByRole('tablist')).getAllByRole('tab');
      await userEvent.pointer({ target: tabs[0], keys: '[MouseMiddle]' });
      expect(eventCallback).toHaveBeenCalledTimes(5);
      expect(eventCallback).toHaveBeenNthCalledWith(5, {
        eventType: 'onClick',
        componentId: 'TABS_TEST.close_tab',
        componentType: 'button',
        shouldStartInteraction: true,
        value: undefined,
        event: expect.any(Object),
        isInteractionSubject: undefined,
      });
    });
  });
});
