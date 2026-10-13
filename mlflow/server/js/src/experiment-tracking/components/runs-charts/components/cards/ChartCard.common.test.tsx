import { expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunsChartCardWrapper, RunsChartsChartsDragGroup } from './ChartCard.common';

test('uses the explicit title tooltip when the displayed title is shortened', () => {
  render(
    <DesignSystemProvider>
      <RunsChartCardWrapper
        title="mae"
        titleTooltip="train/losses/grouped_by_x/after_y/mae"
        onEdit={jest.fn()}
        onDelete={jest.fn()}
        dragGroupKey={RunsChartsChartsDragGroup.GENERAL_AREA}
        onReorderWith={jest.fn()}
        canMoveUp={false}
        canMoveDown={false}
        canMoveToTop={false}
        canMoveToBottom={false}
      />
    </DesignSystemProvider>,
  );

  const heading = screen.getByRole('heading', { name: 'train/losses/grouped_by_x/after_y/mae' });
  expect(heading).toHaveTextContent('mae');
  expect(heading).toHaveAttribute('title', 'train/losses/grouped_by_x/after_y/mae');
});
