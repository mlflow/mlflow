import { screen, render } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Accordion } from './index';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Accordion onChange event emits correct values', () => {
  const eventCallback = jest.fn();
  const onChangeSpy = jest.fn();

  const TestAccordion = ({
    displayMode,
    valueHasNoPii,
  }: {
    displayMode?: 'single' | 'multiple';
    valueHasNoPii?: boolean;
  }) => {
    return (
      <Accordion
        displayMode={displayMode}
        componentId="accordion_test"
        valueHasNoPii={valueHasNoPii}
        onChange={onChangeSpy}
      >
        <Accordion.Panel header="Section 1" key="1">
          foo
        </Accordion.Panel>
        <Accordion.Panel header="Section 2" key="2">
          bar
        </Accordion.Panel>
      </Accordion>
    );
  };

  it('emits accordion event with empty value when valueHasNoPii is false', async () => {
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <TestAccordion displayMode="single" />
      </DesignSystemEventProvider>,
    );

    expect(eventCallback).not.toHaveBeenCalled();
    expect(onChangeSpy).not.toHaveBeenCalled();

    const sections = screen.getAllByRole('tab');
    const section1 = sections[0];

    await userEvent.click(section1);

    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: undefined,
    });

    expect(onChangeSpy).toHaveBeenCalledWith('1');
  });

  it('emits accordion events where the accordion only allows one panel open at a time', async () => {
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <TestAccordion displayMode="single" valueHasNoPii />
      </DesignSystemEventProvider>,
    );

    expect(eventCallback).not.toHaveBeenCalled();
    expect(onChangeSpy).not.toHaveBeenCalled();

    // Retrieve the sections to click on and verify that the sections are not expanded.
    const sections = screen.getAllByRole('tab');
    const section1 = sections[0];
    const section2 = sections[1];

    expect(section1).toBeInTheDocument();
    expect(section2).toBeInTheDocument();
    expect(section1).toHaveAttribute('aria-expanded', 'false');
    expect(section2).toHaveAttribute('aria-expanded', 'false');

    // Click on the first section and verify that it is expanded.
    await userEvent.click(section1);

    expect(section1).toHaveAttribute('aria-expanded', 'true');
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '1',
    });

    expect(onChangeSpy).toHaveBeenCalledWith('1');

    // Close the first section and verify that it is no longer expanded.
    await userEvent.click(section1);

    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: undefined,
    });

    expect(section1).toHaveAttribute('aria-expanded', 'false');
    expect(onChangeSpy).toHaveBeenCalledWith(undefined);

    // Click on the second section and verify that it is expanded.
    await userEvent.click(section2);

    expect(eventCallback).toHaveBeenCalledTimes(3);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '2',
    });

    expect(onChangeSpy).toHaveBeenCalledWith('2');
    expect(section2).toHaveAttribute('aria-expanded', 'true');

    // Click on the first section and verify that the second section is closed and the first section is open.
    await userEvent.click(section1);

    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '1',
    });

    expect(eventCallback).toHaveBeenCalledTimes(4);
    expect(onChangeSpy).toHaveBeenCalledWith('1');
    expect(section1).toHaveAttribute('aria-expanded', 'true');
    expect(section2).toHaveAttribute('aria-expanded', 'false');
  });

  it('emits accordion events where the accordion allows multiple sections open at a time', async () => {
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <TestAccordion displayMode="multiple" valueHasNoPii />
      </DesignSystemEventProvider>,
    );

    expect(eventCallback).not.toHaveBeenCalled();
    expect(onChangeSpy).not.toHaveBeenCalled();

    // Retrieve the sections to click on and verify that the sections are not expanded.
    const sections = screen.getAllByRole('button');
    const section1 = sections[0];
    const section2 = sections[1];

    expect(section1).toBeInTheDocument();
    expect(section2).toBeInTheDocument();
    expect(section1).toHaveAttribute('aria-expanded', 'false');
    expect(section2).toHaveAttribute('aria-expanded', 'false');

    // Click on the first section and verify that it is expanded.
    await userEvent.click(section1);

    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '["1"]',
    });

    expect(onChangeSpy).toHaveBeenCalledWith(['1']);

    // Close the first section and verify that it is not expanded.
    await userEvent.click(section1);

    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '[]',
    });

    expect(section1).toHaveAttribute('aria-expanded', 'false');

    // Click on the second section and verify that it is expanded.
    await userEvent.click(section2);

    expect(eventCallback).toHaveBeenCalledTimes(3);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '["2"]',
    });

    expect(onChangeSpy).toHaveBeenCalledWith(['2']);

    // Click on the first section and verify that both sections are expanded.
    await userEvent.click(section1);

    expect(eventCallback).toHaveBeenCalledTimes(4);
    expect(section1).toHaveAttribute('aria-expanded', 'true');
    expect(section2).toHaveAttribute('aria-expanded', 'true');
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '["2","1"]',
    });

    // Close both sections and verify that both sections are no longer expanded.
    await userEvent.click(section1);
    await userEvent.click(section2);

    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'accordion_test',
      componentType: 'accordion',
      shouldStartInteraction: false,
      value: '[]',
    });

    expect(onChangeSpy).toHaveBeenCalledWith([]);
    expect(eventCallback).toHaveBeenCalledTimes(6);
    expect(section1).toHaveAttribute('aria-expanded', 'false');
    expect(section2).toHaveAttribute('aria-expanded', 'false');
  });
});
