import { describe, expect, it } from '@jest/globals';
import { renderHook } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import { DesignSystemProvider, useDesignSystemTheme } from '@databricks/design-system';

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

it.each([
  ['legacy', LegacyBars],
  ['redesigned', RedesignedBars],
] as const)('keeps %s connector offsets and both gradient colors aligned across gaps', (_name, Bars) => {
  const { result } = renderHook(
    () => {
      const { theme } = useDesignSystemTheme();
      const element = Bars({
        isActiveSpan: false,
        isInActiveChain: true,
        linesToRender: [
          { shouldRender: true, isActive: false },
          { shouldRender: false, isActive: true },
          { shouldRender: true, isActive: true },
        ],
        hasChildren: true,
        isExpanded: true,
      });
      return { element, colors: theme.colors };
    },
    { wrapper: DesignSystemProvider },
  );
  // Jest's JSX transform leaves css props unprocessed, so inspect the emitted geometry directly.
  const [lines, left, bottom] = result.current.element.props.children;
  const { blue500, border } = result.current.colors;
  expect(lines.props.style).toMatchObject({
    backgroundImage: `linear-gradient(${border}, ${border}), linear-gradient(${blue500}, ${blue500})`,
    backgroundPosition: '8px 0, 40px 0',
  });
  expect(left.props.left).toBe(40);
  expect(bottom.props.left).toBe(48);
});
