import type { PropsWithChildren } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { RunArtifactImagePlot } from '../../runs-charts/components/charts/ImageGridPlot.common';
import type { EvaluateCellImage } from '../../../types';
import { EvaluationImageCellRenderer } from './EvaluationImageCellRenderer';

const TestProviders = ({ children }: PropsWithChildren<unknown>) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

jest.mock('../../runs-charts/components/charts/ImageGridPlot.common', () => ({
  RunArtifactImagePlot: jest.fn(() => <div data-testid="run-artifact-image" />),
}));

describe('EvaluationImageCellRenderer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  const renderCell = (value?: EvaluateCellImage, artifactUri?: string) =>
    render(
      <TestProviders>
        <EvaluationImageCellRenderer
          {...({ value, run: { runInfo: { artifactUri } } } as Parameters<typeof EvaluationImageCellRenderer>[0])}
        />
      </TestProviders>,
    );

  it.each([undefined, 's3://bucket/artifacts'])(
    'passes artifact paths and the available root to its loader (%s)',
    (root) => {
      renderCell({ runUuid: 'run-123', filepath: 'images/full.png', compressedFilepath: 'images/thumb.png' }, root);

      expect(screen.getByTestId('run-artifact-image')).toBeInTheDocument();
      expect(jest.mocked(RunArtifactImagePlot).mock.calls[0][0]).toEqual({
        runUuid: 'run-123',
        filepath: 'images/full.png',
        compressedFilepath: 'images/thumb.png',
        artifactRootUri: root,
      });
    },
  );

  it('renders an empty cell without invoking the loader when there is no image', () => {
    renderCell();

    expect(screen.getByText('(empty)')).toBeInTheDocument();
    expect(RunArtifactImagePlot).not.toHaveBeenCalled();
  });
});
