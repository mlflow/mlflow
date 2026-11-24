import { beforeAll, beforeEach, describe, expect, it } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { setLocalStorageItem } from '../../../shared/web-shared/hooks';

import {
  ISSUE_LIST_MIN_WIDTH,
  ISSUE_LIST_RATIO_STORAGE_KEY,
  ISSUE_LIST_RATIO_STORAGE_VERSION,
  IssueSplitPane,
} from './IssueSplitPane';
import { renderWithDesignSystem } from '../../../common/utils/TestUtils.react18';

const WIDE_CONTAINER = 900;
// Too narrow for both minimums (280 + 480), the width at which the old grid also hit its 280px floor.
const NARROW_CONTAINER = 600;

// Reassigned per test, so the fake observer must read it at observe() time rather than capture it.
let containerWidth = WIDE_CONTAINER;

const storeRatio = (ratio: unknown) =>
  setLocalStorageItem(ISSUE_LIST_RATIO_STORAGE_KEY, ISSUE_LIST_RATIO_STORAGE_VERSION, false, ratio);

const renderSplitPane = ({ withDetail }: { withDetail: boolean }) =>
  renderWithDesignSystem(
    <IssueSplitPane
      list={<div data-testid="issue-list">Issues</div>}
      detail={withDetail ? <div data-testid="issue-detail">Traces</div> : undefined}
    />,
  );

/**
 * The list pane's width is an inline pixel width on the resizable box wrapping the list, so read it
 * off the list's parent. Only the resizable pane sets it — in single-column mode there is none.
 */
const listPaneWidth = () =>
  screen
    .getByTestId('issue-list')
    .closest('.react-resizable')
    ?.getAttribute('style')
    ?.match(/width: ([^;]+)/)?.[1];

beforeAll(() => {
  // jsdom never lays anything out and has no ResizeObserver, so report a fixed container width.
  class FixedWidthResizeObserver {
    callback: ResizeObserverCallback;

    constructor(callback: ResizeObserverCallback) {
      this.callback = callback;
    }

    observe(target: Element) {
      const entry = { target, contentRect: { width: containerWidth, height: 600 } } as ResizeObserverEntry;
      this.callback([entry], this as unknown as ResizeObserver);
    }

    unobserve() {}

    disconnect() {}
  }

  global.ResizeObserver = FixedWidthResizeObserver as unknown as typeof ResizeObserver;
});

beforeEach(() => {
  // Each case owns the stored ratio it renders with, including "nothing stored yet".
  window.localStorage.clear();
  containerWidth = WIDE_CONTAINER;
});

describe('IssueSplitPane', () => {
  it('gives the list the full width, with nothing to drag, until an issue is selected', () => {
    renderSplitPane({ withDetail: false });

    expect(screen.getByTestId('issue-list')).toBeInTheDocument();
    expect(screen.queryByTestId('issue-detail')).not.toBeInTheDocument();
    expect(listPaneWidth()).toBeFalsy();
  });

  it('splits the container one-to-two by default', () => {
    renderSplitPane({ withDetail: true });

    expect(screen.getByTestId('issue-detail')).toBeInTheDocument();
    expect(listPaneWidth()).toBe(`${WIDE_CONTAINER / 3}px`);
  });

  it('restores the ratio a previous session left behind', () => {
    storeRatio(0.45);

    renderSplitPane({ withDetail: true });

    expect(listPaneWidth()).toBe(`${WIDE_CONTAINER * 0.45}px`);
  });

  it('falls back to the default split when the stored ratio is unusable', () => {
    storeRatio('two thirds');

    renderSplitPane({ withDetail: true });

    expect(listPaneWidth()).toBe(`${WIDE_CONTAINER / 3}px`);
  });

  it('resizes with the keyboard, clamps both bounds and persists the ratio', async () => {
    const { unmount } = renderSplitPane({ withDetail: true });
    const separator = screen.getByRole('separator', { name: 'Resize issue list' });
    expect(separator).toHaveAttribute('aria-orientation', 'vertical');
    expect(document.getElementById(separator.getAttribute('aria-controls') ?? '')).toContainElement(
      screen.getByTestId('issue-list'),
    );
    separator.focus();
    await userEvent.keyboard('{ArrowRight}');
    expect(listPaneWidth()).toBe('320px');
    await userEvent.keyboard('{Home}{ArrowLeft}');
    expect(listPaneWidth()).toBe('280px');
    await userEvent.keyboard('{End}{ArrowRight}');
    expect(listPaneWidth()).toBe('420px');
    expect(separator).toHaveFocus();
    expect(separator).toHaveAttribute('aria-valuenow', '420');
    await waitFor(() =>
      expect(localStorage.getItem(`${ISSUE_LIST_RATIO_STORAGE_KEY}_v1`)).toBe(String(420 / WIDE_CONTAINER)),
    );
    unmount();
    renderSplitPane({ withDetail: true });
    expect(listPaneWidth()).toBe('420px');
  });

  it('keeps both panes, list at its minimum, when the container cannot fit both minimums', () => {
    // A ratio stored at a wide window must not widen the list past its minimum here: the detail
    // pane is already below its own minimum, which is what the old fixed grid did at this width.
    storeRatio(0.7);
    containerWidth = NARROW_CONTAINER;

    renderSplitPane({ withDetail: true });

    expect(listPaneWidth()).toBe(`${ISSUE_LIST_MIN_WIDTH}px`);
    expect(screen.getByTestId('issue-detail')).toBeInTheDocument();
  });
});
