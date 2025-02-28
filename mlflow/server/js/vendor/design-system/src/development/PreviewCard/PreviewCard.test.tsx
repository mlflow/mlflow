import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import type { PreviewCardProps } from './PreviewCard';
import { PreviewCard } from './PreviewCard';
import { DesignSystemEventProvider, DesignSystemEventProviderAnalyticsEventTypes } from '../../design-system';

describe('<PreviewCard/>', () => {
  const eventCallback = jest.fn();

  const renderComponent = (props: PreviewCardProps) => {
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <PreviewCard {...props}>Target</PreviewCard>
      </DesignSystemEventProvider>,
    );
  };

  it('renders the card as a button when onClick is provided', async () => {
    const onClickSpy = jest.fn();
    renderComponent({
      onClick: onClickSpy,
    });

    await userEvent.click(screen.getByRole('button'));
    expect(onClickSpy).toHaveBeenCalled();
  });

  it('does not render card as a button if onClick is not provided', () => {
    renderComponent({});

    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  it('emits an onClick event when clicked', async () => {
    const onClickSpy = jest.fn();
    const testComponentId = 'test_component_id';
    const testId = 'test_id';
    renderComponent({
      onClick: onClickSpy,
      componentId: testComponentId,
      analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick],
      id: testId,
    });
    expect(onClickSpy).not.toHaveBeenCalled();
    expect(eventCallback).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button'));
    expect(onClickSpy).toHaveBeenCalledWith(
      expect.objectContaining({ target: expect.objectContaining({ id: testId }) }),
    );
    expect(onClickSpy).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentType: 'preview_card',
      componentId: testComponentId,
      value: undefined,
      shouldStartInteraction: true,
      event: expect.any(Object),
      isInteractionSubject: undefined,
    });
  });

  it('emits an onClick event when interacted with via keyboard', async () => {
    const onClickSpy = jest.fn();
    renderComponent({
      onClick: onClickSpy,
    });

    await fireEvent.keyDown(screen.getByRole('button'), { key: 'Enter' });
    await fireEvent.keyDown(screen.getByRole('button'), { key: ' ' });

    expect(onClickSpy).toHaveBeenCalledTimes(2);
  });

  it('does not emit an onClick event when clicked if onClick is not provided', async () => {
    renderComponent({ analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnClick] });
    expect(eventCallback).not.toHaveBeenCalled();
    await userEvent.click(screen.getByText('Target'));
    expect(eventCallback).not.toHaveBeenCalled();
  });

  describe('Disabled', () => {
    it('applies disabled a11y properties', () => {
      renderComponent({
        onClick: jest.fn(),
        disabled: true,
      });

      expect(screen.getByRole('button')).toHaveAttribute('aria-disabled', 'true');
    });

    it('does not emit an onClick event when interacted via keyboard and disabled', async () => {
      const onClickSpy = jest.fn();
      renderComponent({
        onClick: onClickSpy,
        disabled: true,
      });

      await fireEvent.keyDown(screen.getByRole('button'), { key: 'Enter' });
      await fireEvent.keyDown(screen.getByRole('button'), { key: ' ' });

      expect(onClickSpy).not.toHaveBeenCalled();
    });
  });

  describe('Selected', () => {
    it('applies selected a11y properties', () => {
      renderComponent({
        onClick: jest.fn(),
        selected: true,
      });

      expect(screen.getByRole('button')).toHaveAttribute('aria-pressed', 'true');
    });
  });
});
