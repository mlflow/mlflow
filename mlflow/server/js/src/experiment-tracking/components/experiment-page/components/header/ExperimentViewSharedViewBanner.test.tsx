import { jest, describe, it, expect } from '@jest/globals';
import { fireEvent, screen, renderWithDesignSystem } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { ExperimentViewSharedViewBanner } from './ExperimentViewSharedViewBanner';

describe('ExperimentViewSharedViewBanner', () => {
  it('invokes onOverride (and only that) when the override button is clicked', () => {
    const onOverride = jest.fn();
    const onDiscard = jest.fn();
    renderWithDesignSystem(<ExperimentViewSharedViewBanner onOverride={onOverride} onDiscard={onDiscard} />);

    fireEvent.click(screen.getByRole('button', { name: /override saved view/i }));

    expect(onOverride).toHaveBeenCalledTimes(1);
    expect(onDiscard).not.toHaveBeenCalled();
  });

  it('invokes onDiscard (and only that) when the discard button is clicked', () => {
    const onOverride = jest.fn();
    const onDiscard = jest.fn();
    renderWithDesignSystem(<ExperimentViewSharedViewBanner onOverride={onOverride} onDiscard={onDiscard} />);

    fireEvent.click(screen.getByRole('button', { name: /^discard$/i }));

    expect(onDiscard).toHaveBeenCalledTimes(1);
    expect(onOverride).not.toHaveBeenCalled();
  });
});
