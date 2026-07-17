import { jest, describe, it, expect } from '@jest/globals';
import { fireEvent, screen, renderWithDesignSystem } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { SharedViewBanner } from './SharedViewBanner';

describe('SharedViewBanner', () => {
  it('renders the provided message', () => {
    renderWithDesignSystem(
      <SharedViewBanner
        componentId="test.shared_view"
        message="You are viewing a shared view."
        onDiscard={jest.fn()}
      />,
    );
    expect(screen.getByText('You are viewing a shared view.')).toBeInTheDocument();
  });

  it('invokes onDiscard (and only that) when the discard button is clicked', () => {
    const onOverride = jest.fn();
    const onDiscard = jest.fn();
    renderWithDesignSystem(
      <SharedViewBanner
        componentId="test.shared_view"
        message="msg"
        overrideLabel="Override saved view"
        onOverride={onOverride}
        onDiscard={onDiscard}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /^discard$/i }));

    expect(onDiscard).toHaveBeenCalledTimes(1);
    expect(onOverride).not.toHaveBeenCalled();
  });

  it('invokes onOverride (and only that) when the override button is clicked', () => {
    const onOverride = jest.fn();
    const onDiscard = jest.fn();
    renderWithDesignSystem(
      <SharedViewBanner
        componentId="test.shared_view"
        message="msg"
        overrideLabel="Override saved view"
        onOverride={onOverride}
        onDiscard={onDiscard}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /override saved view/i }));

    expect(onOverride).toHaveBeenCalledTimes(1);
    expect(onDiscard).not.toHaveBeenCalled();
  });

  it('does not render an override button when onOverride is omitted (traces variant)', () => {
    renderWithDesignSystem(<SharedViewBanner componentId="test.shared_view" message="msg" onDiscard={jest.fn()} />);

    expect(screen.queryByRole('button', { name: /override/i })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^discard$/i })).toBeInTheDocument();
  });
});
