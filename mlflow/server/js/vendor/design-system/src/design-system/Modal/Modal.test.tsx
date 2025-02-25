import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { expect } from '@databricks/config-jest';

import { Modal } from './Modal';
import { DesignSystemEventProvider } from '../DesignSystemEventProvider';

describe('Modal', () => {
  it("doesn't render with ant-scrolling-effect class on body", async () => {
    const { baseElement } = render(
      <Modal
        componentId="codegen_design-system_src_design-system_modal_modal.test.tsx_11"
        visible={true}
        title="Test modal"
      />,
    );
    expect(baseElement).not.toHaveClass('ant-scrolling-effect');
    expect(baseElement).toHaveClass('scroll-lock-effect');
  });

  it('handles views & clicks with DesignSystemEventProvider', async () => {
    // Disable IntersectionObserver for useIsInViewport hook to trigger
    (window as any).IntersectionObserver = undefined;

    // Arrange
    const handleOk = jest.fn();
    const handleCancel = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Modal
          componentId="bestModalEver"
          visible={true}
          title="Test modal"
          cancelText="Cancel"
          onCancel={handleCancel}
          okText="Ok"
          onOk={handleOk}
          shouldStartInteraction
        />
      </DesignSystemEventProvider>,
    );
    expect(handleCancel).not.toHaveBeenCalled();
    expect(handleOk).not.toHaveBeenCalled();

    // Assert for view
    expect(screen.getByText('Test modal')).toBeVisible();
    expect(eventCallback).toBeCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onView',
      componentId: 'bestModalEver',
      componentType: 'modal',
      shouldStartInteraction: true,
      value: undefined,
    });

    // Assert for cancel
    await userEvent.click(screen.getByText('Cancel'));
    expect(handleCancel).toHaveBeenCalledTimes(1);
    expect(eventCallback).toBeCalledTimes(2);
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onClick',
      componentId: 'bestModalEver.footer.cancel',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });

    // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
    await userEvent.click(screen.getByText('Ok'));
    expect(handleOk).toHaveBeenCalledTimes(1);
    expect(eventCallback).toBeCalledTimes(3);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'bestModalEver.footer.ok',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });
  });

  it('handles views & clicks with DesignSystemEventProvider but with shouldStartInteraction not set', async () => {
    // Disable IntersectionObserver for useIsInViewport hook to trigger
    (window as any).IntersectionObserver = undefined;

    // Arrange
    const handleOk = jest.fn();
    const handleCancel = jest.fn();
    const eventCallback = jest.fn();
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <Modal
          componentId="bestModalEver"
          visible={true}
          title="Test modal"
          cancelText="Cancel"
          onCancel={handleCancel}
          okText="Ok"
          onOk={handleOk}
        />
      </DesignSystemEventProvider>,
    );
    expect(handleCancel).not.toHaveBeenCalled();
    expect(handleOk).not.toHaveBeenCalled();

    // Assert for view
    expect(screen.getByText('Test modal')).toBeVisible();
    expect(eventCallback).toBeCalledTimes(1);
    expect(eventCallback).toHaveBeenNthCalledWith(1, {
      eventType: 'onView',
      componentId: 'bestModalEver',
      componentType: 'modal',
      shouldStartInteraction: false,
      value: undefined,
    });

    // Assert for cancel
    await userEvent.click(screen.getByText('Cancel'));
    expect(handleCancel).toHaveBeenCalledTimes(1);
    expect(eventCallback).toBeCalledTimes(2);
    expect(eventCallback).toHaveBeenNthCalledWith(2, {
      eventType: 'onClick',
      componentId: 'bestModalEver.footer.cancel',
      componentType: 'button',
      shouldStartInteraction: false,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });

    // Assert for close ( This isn't a du bois button yet, so it's not being tracked )
    await userEvent.click(screen.getByText('Ok'));
    expect(handleOk).toHaveBeenCalledTimes(1);
    expect(eventCallback).toBeCalledTimes(3);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'bestModalEver.footer.ok',
      componentType: 'button',
      shouldStartInteraction: false,
      isInteractionSubject: true,
      value: undefined,
      event: expect.anything(),
    });
  });
});
