import { DesignSystemProvider } from '@databricks/design-system';
import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import { getArtifactBlob } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import {
  fetchArtifactWithPresignedUrl,
  fetchRunArtifactWithPresignedUrl,
} from '@mlflow/mlflow/src/experiment-tracking/utils/PresignedArtifactUtils';
import { ImageGridRunHeader, RunArtifactImagePlot } from './ImageGridPlot.common';

jest.mock('@mlflow/mlflow/src/experiment-tracking/utils/PresignedArtifactUtils', () => ({
  fetchArtifactWithPresignedUrl: jest.fn(),
  fetchRunArtifactWithPresignedUrl: jest.fn(),
}));

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

describe('RunArtifactImagePlot', () => {
  beforeEach(() => {
    jest.mocked(fetchArtifactWithPresignedUrl).mockResolvedValue(new Blob(['image']));
    jest.mocked(fetchRunArtifactWithPresignedUrl).mockResolvedValue(new Blob(['image']));
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: jest.fn().mockReturnValueOnce('blob:full').mockReturnValueOnce('blob:compressed'),
    });
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: jest.fn(),
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('loads both image variants through the presigned artifact resolver', async () => {
    render(
      <DesignSystemProvider>
        <RunArtifactImagePlot runUuid="run-123" filepath="images/full.png" compressedFilepath="images/thumbnail.png" />
      </DesignSystemProvider>,
    );

    await waitFor(() => expect(fetchRunArtifactWithPresignedUrl).toHaveBeenCalledTimes(2));
    expect(fetchRunArtifactWithPresignedUrl).toHaveBeenNthCalledWith(
      1,
      'run-123',
      'images/full.png',
      expect.stringContaining('path=images%2Ffull.png&run_uuid=run-123'),
      getArtifactBlob,
    );
    expect(fetchRunArtifactWithPresignedUrl).toHaveBeenNthCalledWith(
      2,
      'run-123',
      'images/thumbnail.png',
      expect.stringContaining('path=images%2Fthumbnail.png&run_uuid=run-123'),
      getArtifactBlob,
    );
    await waitFor(() => expect(URL.createObjectURL).toHaveBeenCalledTimes(2));
  });

  it('reuses a supplied artifact root for both image variants', async () => {
    render(
      <DesignSystemProvider>
        <RunArtifactImagePlot
          runUuid="run-123"
          filepath="images/full.png"
          compressedFilepath="images/thumbnail.png"
          artifactRootUri="s3://bucket/run-123/artifacts"
        />
      </DesignSystemProvider>,
    );

    await waitFor(() => expect(fetchArtifactWithPresignedUrl).toHaveBeenCalledTimes(2));
    expect(fetchArtifactWithPresignedUrl).toHaveBeenNthCalledWith(
      1,
      {
        runUuid: 'run-123',
        path: 'images/full.png',
        artifactRootUri: 's3://bucket/run-123/artifacts',
      },
      expect.stringContaining('path=images%2Ffull.png&run_uuid=run-123'),
      getArtifactBlob,
    );
    expect(fetchRunArtifactWithPresignedUrl).not.toHaveBeenCalled();
  });
});
