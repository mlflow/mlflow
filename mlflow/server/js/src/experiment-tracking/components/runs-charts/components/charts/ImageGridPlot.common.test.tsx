import { DesignSystemProvider } from '@databricks/design-system';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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
  const originalImageSrc = Object.getOwnPropertyDescriptor(window.Image.prototype, 'src');
  beforeAll(() => {
    Object.defineProperty(window.Image.prototype, 'src', {
      configurable: true,
      set() {
        Promise.resolve().then(() => this.onload?.());
      },
      get() {
        return '';
      },
    });
  });
  afterAll(() => {
    Object.defineProperty(window.Image.prototype, 'src', originalImageSrc!);
  });
  beforeEach(() => {
    jest.mocked(fetchArtifactWithPresignedUrl).mockResolvedValue(new Blob(['image']));
    jest.mocked(fetchRunArtifactWithPresignedUrl).mockResolvedValue(new Blob(['image']));
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: jest.fn().mockReturnValueOnce('blob:compressed').mockReturnValueOnce('blob:full'),
    });
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: jest.fn() });
  });
  afterEach(() => {
    jest.clearAllMocks();
  });

  it.each([undefined, 's3://bucket/run-123/artifacts'])(
    'fetches only the thumbnail until the preview opens (root: %s)',
    async (artifactRootUri) => {
      const { container, unmount } = render(
        <DesignSystemProvider>
          <RunArtifactImagePlot
            runUuid="run-123"
            filepath="images/full.png"
            compressedFilepath="images/thumbnail.png"
            artifactRootUri={artifactRootUri}
          />
        </DesignSystemProvider>,
      );
      const fetchImage = artifactRootUri ? fetchArtifactWithPresignedUrl : fetchRunArtifactWithPresignedUrl;
      await waitFor(() => expect(container.querySelector('img')).toHaveAttribute('src', 'blob:compressed'));
      expect(fetchImage).toHaveBeenCalledTimes(1);
      if (artifactRootUri) {
        expect(fetchArtifactWithPresignedUrl).toHaveBeenCalledWith(
          { runUuid: 'run-123', path: 'images/thumbnail.png', artifactRootUri },
          expect.stringContaining('path=images%2Fthumbnail.png'),
          getArtifactBlob,
        );
        expect(fetchRunArtifactWithPresignedUrl).not.toHaveBeenCalled();
      } else {
        expect(fetchRunArtifactWithPresignedUrl).toHaveBeenCalledWith(
          'run-123',
          'images/thumbnail.png',
          expect.stringContaining('path=images%2Fthumbnail.png'),
          getArtifactBlob,
        );
      }
      await userEvent.click(container.querySelector('img')!);
      await waitFor(() => expect(fetchImage).toHaveBeenCalledTimes(2));
      expect(URL.createObjectURL).toHaveBeenCalledTimes(2);
      unmount();
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:compressed');
      expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:full');
    },
  );

  it('does not allocate a blob URL when a pending thumbnail completes after unmount', async () => {
    let resolve: (blob: Blob) => void = () => {};
    jest.mocked(fetchRunArtifactWithPresignedUrl).mockReturnValueOnce(
      new Promise<Blob>((r) => {
        resolve = r;
      }),
    );
    const { unmount } = render(
      <DesignSystemProvider>
        <RunArtifactImagePlot runUuid="run" filepath="full.png" compressedFilepath="thumb.png" />
      </DesignSystemProvider>,
    );
    unmount();
    await act(async () => resolve(new Blob(['thumbnail'])));
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });
});
