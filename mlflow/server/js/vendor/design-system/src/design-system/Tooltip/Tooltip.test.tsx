import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { Tooltip } from './Tooltip';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';
import { DesignSystemProvider } from '../DesignSystemProvider';

describe('Tooltip', () => {
  const eventCallback = jest.fn();
  const tooltipComponentId = 'tooltip-component-id';
  const renderComponent = () => {
    return render(
      <DesignSystemEventProvider callback={eventCallback}>
        <DesignSystemProvider>
          <Tooltip content="Hello world" componentId={tooltipComponentId}>
            <button>Target</button>
          </Tooltip>
        </DesignSystemProvider>
      </DesignSystemEventProvider>,
    );
  };
  const getButton = () => screen.getByRole('button', { name: 'Target' });
  const getTooltip = () => screen.getByRole('tooltip', { name: 'Hello world' });
  const queryTooltip = () => screen.queryByRole('tooltip', { name: 'Hello world' });

  it('renders tooltip on hover', async () => {
    render(
      <DesignSystemProvider>
        <Tooltip
          componentId="codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_29"
          content="Hello world"
        >
          <button>Target</button>
        </Tooltip>
      </DesignSystemProvider>,
    );

    await userEvent.hover(screen.getByRole('button', { name: 'Target' }));
    await waitFor(() => expect(screen.getByRole('tooltip', { name: 'Hello world' })).toBeInTheDocument());
  });

  it('renders tooltip on focus', async () => {
    render(
      <DesignSystemProvider>
        <Tooltip
          componentId="codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_42"
          content="Hello world"
        >
          <button>Target</button>
        </Tooltip>
      </DesignSystemProvider>,
    );

    const trigger = screen.getByRole('button', { name: 'Target' });

    act(() => {
      trigger.focus();
    });

    await waitFor(() => expect(screen.getByRole('tooltip', { name: 'Hello world' })).toBeInTheDocument());
  });

  it('does not render tooltip with null or undefined content', async () => {
    render(
      <DesignSystemProvider>
        <Tooltip componentId="codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_60" content={null}>
          <button>null button</button>
        </Tooltip>
        <Tooltip componentId="codegen_design-system_src_design-system_tooltip_tooltip.test.tsx_63" content={undefined}>
          <button>undefined button</button>
        </Tooltip>
      </DesignSystemProvider>,
    );

    await userEvent.hover(screen.getByRole('button', { name: 'null button' }));
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
    await userEvent.hover(screen.getByRole('button', { name: 'undefined button' }));
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('emit onView event when tooltip is hovered', async () => {
    const { container } = renderComponent();
    expect(eventCallback).not.toHaveBeenCalled();
    await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());

    // first hover: emits onView event
    await userEvent.hover(getButton());
    await waitFor(() => {
      expect(getTooltip()).toBeVisible();
    });
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: tooltipComponentId,
      componentType: 'tooltip',
      shouldStartInteraction: false,
      value: undefined,
    });

    // JSDOM limitation: must unhover, then click away to hide tooltip and allow future hovers to trigger tooltip
    await userEvent.unhover(screen.getByRole('button', { name: 'Target' }));
    await userEvent.click(container);
    await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());
    expect(eventCallback).toHaveBeenCalledTimes(1);

    // second hover: does not emit new onView event
    await userEvent.hover(getButton());
    await waitFor(() => {
      expect(getTooltip()).toBeVisible();
    });
    expect(eventCallback).toHaveBeenCalledTimes(1);
  });

  it('emit onView event when tooltip is focused', async () => {
    renderComponent();
    expect(eventCallback).not.toHaveBeenCalled();
    const trigger = getButton();
    await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());

    // first focus: emits onView event
    act(() => {
      trigger.focus();
    });
    await waitFor(() => {
      expect(getTooltip()).toBeVisible();
    });
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onView',
      componentId: tooltipComponentId,
      componentType: 'tooltip',
      shouldStartInteraction: false,
      value: undefined,
    });

    act(() => {
      trigger.blur();
    });
    await waitFor(() => expect(queryTooltip()).not.toBeInTheDocument());
    expect(eventCallback).toHaveBeenCalledTimes(1);

    // second focus: does not emit new onView event
    act(() => {
      trigger.focus();
    });
    await waitFor(() => {
      expect(getTooltip()).toBeVisible();
    });
    expect(eventCallback).toHaveBeenCalledTimes(1);
  });
});
