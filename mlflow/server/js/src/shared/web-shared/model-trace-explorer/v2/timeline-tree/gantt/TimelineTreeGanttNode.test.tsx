import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { act } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { TimelineTreeGanttNode as RedesignedGantt } from './TimelineTreeGanttNode';
import { TimelineTreeGanttNode as LegacyGantt } from '../../../timeline-tree/gantt/TimelineTreeGanttNode';

afterEach(() => {
  jest.restoreAllMocks();
});

describe.each([
  ['legacy', LegacyGantt],
  ['redesigned', RedesignedGantt],
] as const)('%s Gantt label fitting', (_name, TimelineTreeGanttNode) => {
  it('exposes all labels before measuring and keeps short and long labels in the right place', async () => {
    const visibleAtMeasurement: number[] = [];
    jest.spyOn(HTMLElement.prototype, 'offsetWidth', 'get').mockImplementation(function (this: HTMLElement) {
      if (this.textContent === 'fit-short' || this.textContent === 'fit-long') {
        visibleAtMeasurement.push(
          [...document.querySelectorAll('span')].filter(
            (element) =>
              (element.textContent === 'fit-short' || element.textContent === 'fit-long') &&
              element.style.display === 'inline-flex',
          ).length,
        );
        return this.textContent === 'fit-short' ? 40 : 200;
      }
      return 0;
    });
    const { container } = render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          {['fit-short', 'fit-long'].map((title) => (
            <TimelineTreeGanttNode
              key={title}
              node={{ key: title, title, start: 0, end: 1, attributes: {}, assessments: [], traceId: 'trace' }}
              selectedKey="fit-short"
              leftOffset={0}
              width={120}
              onSelect={undefined}
              expandedKeys={new Set()}
              setExpandedKeys={() => {}}
            />
          ))}
        </DesignSystemProvider>
      </IntlProvider>,
    );
    await act(async () => {});
    expect(visibleAtMeasurement).toEqual([4, 4, 4, 4]);
    for (const title of ['fit-short', 'fit-long']) {
      const displays = [...container.querySelectorAll('span')]
        .filter((element) => element.textContent === title && element.style.display)
        .map((element) => element.style.display);
      expect(displays).toEqual(title === 'fit-short' ? ['inline-flex', 'none'] : ['none', 'inline-flex']);
    }
  });
});
