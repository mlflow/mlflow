import { describe, it, expect } from '@jest/globals';
import { render } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';

import { SegmentedProgressBar } from './SegmentedProgressBar';

const renderWithProviders = (ui: React.ReactElement) => render(<DesignSystemProvider>{ui}</DesignSystemProvider>);

describe('SegmentedProgressBar', () => {
  it('renders nothing when there are no items', () => {
    const { container } = renderWithProviders(<SegmentedProgressBar items={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders one segment per item', () => {
    const { container } = renderWithProviders(
      <SegmentedProgressBar items={[{ color: 'red' }, { color: 'blue' }, { color: 'green' }]} />,
    );
    // A single wrapper div with one child segment per item.
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.children).toHaveLength(3);
  });
});
