import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { render, screen } from '@databricks/web-shared/test-utils/render';
import { fireEvent } from '@testing-library/react';

import { ModelTraceExplorerLargeText } from './ModelTraceExplorerLargeText';

let visibleRows = [0, 1];
const mockScrollToIndex = jest.fn();
const mockMeasureElement = jest.fn();
const mockCopy = jest.fn();
const mockVirtualizer = {
  scrollToIndex: mockScrollToIndex,
  measureElement: mockMeasureElement,
  getTotalSize: () => 2000,
  getVirtualItems: () => visibleRows.map((index) => ({ index, key: index, start: index * 20 })),
};
jest.mock('@tanstack/react-virtual', () => ({ useVirtualizer: () => mockVirtualizer }));
jest.mock('use-clipboard-copy', () => ({ useClipboard: () => ({ copy: mockCopy }) }));

const Providers = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('large trace search rendering', () => {
  beforeEach(() => {
    visibleRows = [0, 1];
    jest.clearAllMocks();
    HTMLElement.prototype.scrollIntoView = jest.fn();
  });

  it('reveals a selection once even when its virtual row unmounts and remounts', () => {
    const text = 'Match\n'.repeat(1000);
    const snippet = (activeMatchIndex?: number, searchFilter = 'match') => (
      <ModelTraceExplorerLargeText text={text} searchFilter={searchFilter} activeMatchIndex={activeMatchIndex} />
    );
    const { rerender } = render(snippet(0), { wrapper: Providers });
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledTimes(1);
    visibleRows = [40, 41];
    rerender(snippet(0));
    visibleRows = [0, 1];
    rerender(snippet(0));
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledTimes(1);
    expect(mockScrollToIndex).toHaveBeenCalledTimes(1);
    rerender(snippet(1));
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledTimes(2);
    rerender(snippet(1, 'Match'));
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledTimes(3);
    rerender(snippet());
    rerender(snippet(1, 'Match'));
    expect(HTMLElement.prototype.scrollIntoView).toHaveBeenCalledTimes(4);
  });

  it('bounds dense highlight DOM while preserving text, a late active occurrence and full Copy', () => {
    const text = 'x'.repeat(100001) + ' café 日本語 🌳';
    const { container } = render(
      <ModelTraceExplorerLargeText text={text} copyText={text} searchFilter="x" activeMatchIndex={1999} />,
      { wrapper: Providers },
    );
    expect(container.querySelectorAll('[data-index] span').length).toBeLessThanOrEqual(42);
    expect(container.querySelector('[data-index="0"]')?.textContent).toBe(text.slice(0, 2000));
    expect(container.querySelector('[data-index="1"]')?.textContent).toBe(text.slice(2000, 4000));
    const reveals = jest.mocked(HTMLElement.prototype.scrollIntoView).mock.instances;
    const active = reveals[reveals.length - 1] as unknown as HTMLElement;
    expect(active).toBeInTheDocument();
    const preceding = document.createRange();
    preceding.setStart(active.parentElement as HTMLElement, 0);
    preceding.setEndBefore(active);
    expect(preceding.toString()).toHaveLength(1999);
    fireEvent.click(screen.getByRole('button', { name: 'Copy' }));
    expect(mockCopy).toHaveBeenCalledWith(text);
  });
});
