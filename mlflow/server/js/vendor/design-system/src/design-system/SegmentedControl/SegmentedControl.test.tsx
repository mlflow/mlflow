import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { SegmentedControlButton, SegmentedControlGroup } from './SegmentedControl';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('SegmentedControl', () => {
  const onChangeSpy = jest.fn();
  const eventCallback = jest.fn();

  const Component = ({ valueHasNoPii }: { valueHasNoPii?: boolean }) => (
    <DesignSystemEventProvider callback={eventCallback}>
      <SegmentedControlGroup
        name="test"
        componentId="segmented_control_group_test"
        onChange={onChangeSpy}
        valueHasNoPii={valueHasNoPii}
      >
        <SegmentedControlButton value="a">A</SegmentedControlButton>
        <SegmentedControlButton value="b">B</SegmentedControlButton>
      </SegmentedControlGroup>
    </DesignSystemEventProvider>
  );

  it('emits value change events without value', async () => {
    render(<Component />);
    expect(onChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const button = screen.getByText('B');
    await userEvent.click(button);
    expect(onChangeSpy).toHaveBeenCalledWith(
      expect.objectContaining({ target: expect.objectContaining({ value: 'b' }) }),
    );
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'segmented_control_group_test',
      componentType: 'segmented_control_group',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('emits value change events with value', async () => {
    render(<Component valueHasNoPii />);
    expect(onChangeSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    const button = screen.getByText('B');
    await userEvent.click(button);
    expect(onChangeSpy).toHaveBeenCalledWith(
      expect.objectContaining({ target: expect.objectContaining({ value: 'b' }) }),
    );
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'segmented_control_group_test',
      componentType: 'segmented_control_group',
      shouldStartInteraction: false,
      value: 'b',
    });
  });
});
