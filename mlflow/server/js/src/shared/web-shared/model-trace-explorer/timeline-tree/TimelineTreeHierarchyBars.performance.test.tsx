import { describe, expect, it } from '@jest/globals';
import { render } from '@databricks/web-shared/test-utils/render';
import { DesignSystemProvider } from '@databricks/design-system';

import { TimelineTreeHierarchyBars as LegacyBars } from './TimelineTreeHierarchyBars';
import { TimelineTreeHierarchyBars as RedesignedBars } from '../v2/timeline-tree/TimelineTreeHierarchyBars';

describe.each([
  ['legacy', LegacyBars],
  ['redesigned', RedesignedBars],
] as const)('%s hierarchy rendering', (_name, Bars) => {
  it('renders a root leaf without connector elements', () => {
    const { container } = render(
      <DesignSystemProvider>
        <Bars isActiveSpan={false} isInActiveChain={false} linesToRender={[]} hasChildren={false} isExpanded={false} />
      </DesignSystemProvider>,
    );
    expect(container.querySelectorAll('div')).toHaveLength(1);
  });
  it('uses a fixed number of elements for a deep row', () => {
    const { container } = render(
      <DesignSystemProvider>
        <Bars
          isActiveSpan={false}
          isInActiveChain
          linesToRender={Array.from({ length: 10 }, (_, depth) => ({ shouldRender: true, isActive: depth === 4 }))}
          hasChildren
          isExpanded
        />
      </DesignSystemProvider>,
    );
    expect(container.querySelectorAll('div').length).toBeLessThanOrEqual(4);
  });
});

it('keeps redesigned metric-row connector centers at 24px', () => {
  const { getByTestId } = render(
    <DesignSystemProvider>
      <RedesignedBars
        isActiveSpan={false}
        isInActiveChain
        linesToRender={[{ shouldRender: false, isActive: false }]}
        hasChildren
        isExpanded
        rowHeight={48}
      />
    </DesignSystemProvider>,
  );
  expect(getByTestId('icon-left-connector')).toHaveAttribute('data-row-center', '24');
  expect(getByTestId('icon-bottom-connector')).toHaveAttribute('data-active', 'true');
});
