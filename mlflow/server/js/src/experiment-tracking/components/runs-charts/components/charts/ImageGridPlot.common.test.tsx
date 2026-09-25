import { DesignSystemProvider } from '@databricks/design-system';
import { describe, expect, it } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { ImageGridRunHeader } from './ImageGridPlot.common';

const renderHeader = ({ showParams }: { showParams?: boolean } = {}) => {
  return render(
    <DesignSystemProvider>
      <ImageGridRunHeader
        displayName="training-run"
        params={{
          lr: { key: 'lr', value: 0.1 },
          batchSize: { key: 'batch_size', value: 4 },
        }}
        showParams={showParams}
      />
    </DesignSystemProvider>,
  );
};

describe('ImageGridRunHeader', () => {
  it('shows run params by default', () => {
    renderHeader();

    expect(screen.getByText('lr=0.1, batch_size=4')).toBeInTheDocument();
  });

  it('hides run params when disabled', () => {
    renderHeader({ showParams: false });

    expect(screen.getByText('training-run')).toBeInTheDocument();
    expect(screen.queryByText('lr=0.1, batch_size=4')).not.toBeInTheDocument();
  });
});
