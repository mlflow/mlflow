import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import userEvent from '@testing-library/user-event';

import {
  SharedViewActionsBridgeProvider,
  usePublishSharedViewActions,
  useSharedViewActionsBridge,
} from './useSharedViewActionsBridge';

// A publisher (stands in for ExperimentView) and a consumer (stands in for the header Views button),
// mounted as siblings under one provider — mirroring the real ExperimentPageTabs tree.
const Publisher = ({ active, override, discard }: { active: boolean; override: () => void; discard: () => void }) => {
  usePublishSharedViewActions({ active, override, discard });
  return null;
};

const Consumer = () => {
  const actions = useSharedViewActionsBridge();
  if (!actions) {
    return <div data-testid="no-actions" />;
  }
  return (
    <div>
      <button data-testid="override" onClick={actions.override}>
        override
      </button>
      <button data-testid="discard" onClick={actions.discard}>
        discard
      </button>
    </div>
  );
};

describe('useSharedViewActionsBridge', () => {
  it('exposes no actions to the consumer when no shared view is active', () => {
    render(
      <SharedViewActionsBridgeProvider>
        <Publisher active={false} override={jest.fn()} discard={jest.fn()} />
        <Consumer />
      </SharedViewActionsBridgeProvider>,
    );
    expect(screen.getByTestId('no-actions')).toBeInTheDocument();
    expect(screen.queryByTestId('override')).not.toBeInTheDocument();
  });

  it('bridges override/discard from the publisher to the consumer when active', async () => {
    const override = jest.fn();
    const discard = jest.fn();
    render(
      <SharedViewActionsBridgeProvider>
        <Publisher active override={override} discard={discard} />
        <Consumer />
      </SharedViewActionsBridgeProvider>,
    );

    await userEvent.click(screen.getByTestId('override'));
    expect(override).toHaveBeenCalledTimes(1);

    await userEvent.click(screen.getByTestId('discard'));
    expect(discard).toHaveBeenCalledTimes(1);
  });

  it('invokes the LATEST published handlers (ref-based), not stale closures', async () => {
    const first = jest.fn();
    const second = jest.fn();
    const { rerender } = render(
      <SharedViewActionsBridgeProvider>
        <Publisher active override={first} discard={jest.fn()} />
        <Consumer />
      </SharedViewActionsBridgeProvider>,
    );

    // Republish with a new override closure (as ExperimentView would when its state changes).
    rerender(
      <SharedViewActionsBridgeProvider>
        <Publisher active override={second} discard={jest.fn()} />
        <Consumer />
      </SharedViewActionsBridgeProvider>,
    );

    await userEvent.click(screen.getByTestId('override'));
    expect(second).toHaveBeenCalledTimes(1);
    expect(first).not.toHaveBeenCalled();
  });

  it('is a no-op (no throw) when the publisher is rendered outside a provider', () => {
    expect(() => render(<Publisher active override={jest.fn()} discard={jest.fn()} />)).not.toThrow();
  });
});
