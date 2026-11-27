import type { PropsWithChildren } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getArtifactBlob } from '@mlflow/mlflow/src/common/utils/ArtifactUtils';
import Utils from '@mlflow/mlflow/src/common/utils/Utils';
import {
  fetchArtifactPreferPresigned,
  fetchRunArtifactPreferPresigned,
} from '@mlflow/mlflow/src/experiment-tracking/utils/PresignedArtifactUtils';
import { ImageGridRunHeader, RunArtifactImagePlot } from './ImageGridPlot.common';

const TestProviders = ({ children }: PropsWithChildren<unknown>) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

jest.mock('@mlflow/mlflow/src/experiment-tracking/utils/PresignedArtifactUtils', () => ({
  fetchArtifactPreferPresigned: jest.fn(),
  fetchRunArtifactPreferPresigned: jest.fn(),
}));

const renderHeader = ({ showParams }: { showParams?: boolean } = {}) => {
  return render(
    <TestProviders>
      <ImageGridRunHeader
        displayName="training-run"
        params={{
          lr: { key: 'lr', value: 0.1 },
          batchSize: { key: 'batch_size', value: 4 },
        }}
        showParams={showParams}
      />
    </TestProviders>,
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
    jest.mocked(fetchArtifactPreferPresigned).mockResolvedValue(new Blob(['image']));
    jest.mocked(fetchRunArtifactPreferPresigned).mockResolvedValue(new Blob(['image']));
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
        <TestProviders>
          <RunArtifactImagePlot
            runUuid="run-123"
            filepath="images/full.png"
            compressedFilepath="images/thumbnail.png"
            artifactRootUri={artifactRootUri}
          />
        </TestProviders>,
      );
      const fetchImage = artifactRootUri ? fetchArtifactPreferPresigned : fetchRunArtifactPreferPresigned;
      await waitFor(() => expect(container.querySelector('img')).toHaveAttribute('src', 'blob:compressed'));
      expect(fetchImage).toHaveBeenCalledTimes(1);
      if (artifactRootUri) {
        expect(fetchArtifactPreferPresigned).toHaveBeenCalledWith(
          { runUuid: 'run-123', path: 'images/thumbnail.png', artifactRootUri },
          expect.stringContaining('path=images%2Fthumbnail.png'),
          getArtifactBlob,
        );
        expect(fetchRunArtifactPreferPresigned).not.toHaveBeenCalled();
      } else {
        expect(fetchRunArtifactPreferPresigned).toHaveBeenCalledWith(
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
    jest.mocked(fetchRunArtifactPreferPresigned).mockReturnValueOnce(
      new Promise<Blob>((r) => {
        resolve = r;
      }),
    );
    const { unmount } = render(
      <TestProviders>
        <RunArtifactImagePlot runUuid="run" filepath="full.png" compressedFilepath="thumb.png" />
      </TestProviders>,
    );
    unmount();
    await act(async () => resolve(new Blob(['thumbnail'])));
    expect(URL.createObjectURL).not.toHaveBeenCalled();
  });

  it('reserves the image dimensions while the thumbnail is loading', () => {
    jest.mocked(fetchRunArtifactPreferPresigned).mockReturnValueOnce(new Promise(() => {}));
    const { container } = render(
      <TestProviders>
        <RunArtifactImagePlot
          runUuid="run"
          filepath="full.png"
          compressedFilepath="thumb.png"
          imageSize={200}
          maxImageSize={250}
        />
      </TestProviders>,
    );
    expect(container.querySelector('[aria-busy="true"]')).toHaveStyle({
      width: '200px',
      height: '200px',
      maxWidth: '250px',
      maxHeight: '250px',
    });
    expect(fetchRunArtifactPreferPresigned).toHaveBeenCalledTimes(1);
  });

  it('shows missing thumbnails within each sized cell without sending global notifications', async () => {
    const notify = jest.spyOn(Utils, 'logErrorAndNotifyUser');
    jest.mocked(fetchRunArtifactPreferPresigned).mockRejectedValue(new Error('Image not found'));
    const { container } = render(
      <TestProviders>
        <RunArtifactImagePlot runUuid="run-1" filepath="full.png" compressedFilepath="thumb.png" imageSize={200} />
        <RunArtifactImagePlot runUuid="run-2" filepath="full.png" compressedFilepath="thumb.png" imageSize={200} />
      </TestProviders>,
    );
    expect(await screen.findAllByText('Image unavailable')).toHaveLength(2);
    for (const frame of container.querySelectorAll('[aria-busy="false"]')) {
      expect(frame).toHaveStyle({ width: '200px', height: '200px' });
    }
    expect(fetchRunArtifactPreferPresigned).toHaveBeenCalledTimes(2);
    expect(URL.createObjectURL).not.toHaveBeenCalled();
    expect(notify).not.toHaveBeenCalled();
    notify.mockRestore();
  });

  it('keeps the thumbnail and shows a local error when its original is missing', async () => {
    const notify = jest.spyOn(Utils, 'logErrorAndNotifyUser');
    jest
      .mocked(fetchRunArtifactPreferPresigned)
      .mockResolvedValueOnce(new Blob(['thumbnail']))
      .mockRejectedValueOnce(new Error('Original not found'));
    const { container, unmount } = render(
      <TestProviders>
        <RunArtifactImagePlot runUuid="run" filepath="full.png" compressedFilepath="thumb.png" imageSize={200} />
      </TestProviders>,
    );
    await waitFor(() => expect(container.querySelector('img')).toHaveAttribute('src', 'blob:compressed'));
    expect(fetchRunArtifactPreferPresigned).toHaveBeenCalledTimes(1);
    await userEvent.click(container.querySelector('img')!);
    expect(await screen.findByText('Full-size image unavailable')).toBeInTheDocument();
    expect(container.querySelector('img')).toHaveAttribute('src', 'blob:compressed');
    expect(container.querySelector('[aria-busy="false"]')).toHaveStyle({ width: '200px', height: '200px' });
    expect(fetchRunArtifactPreferPresigned).toHaveBeenCalledTimes(2);
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(notify).not.toHaveBeenCalled();
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:compressed');
    notify.mockRestore();
  });

  it('does not allocate a blob URL when a pending original completes after unmount', async () => {
    let resolve: (blob: Blob) => void = () => {};
    jest
      .mocked(fetchRunArtifactPreferPresigned)
      .mockResolvedValueOnce(new Blob(['thumbnail']))
      .mockReturnValueOnce(
        new Promise<Blob>((r) => {
          resolve = r;
        }),
      );
    const { container, unmount } = render(
      <TestProviders>
        <RunArtifactImagePlot runUuid="run" filepath="full.png" compressedFilepath="thumb.png" />
      </TestProviders>,
    );
    await waitFor(() => expect(container.querySelector('img')).toHaveAttribute('src', 'blob:compressed'));
    await userEvent.click(container.querySelector('img')!);
    expect(fetchRunArtifactPreferPresigned).toHaveBeenCalledTimes(2);
    unmount();
    await act(async () => resolve(new Blob(['original'])));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:compressed');
  });
});
