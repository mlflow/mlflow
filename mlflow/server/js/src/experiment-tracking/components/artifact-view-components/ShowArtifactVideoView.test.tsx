import { jest, describe, beforeAll, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import ShowArtifactVideoView from './ShowArtifactVideoView';
import { ArtifactPreviewTooLargeError } from '../../../common/utils/ArtifactUtils';

jest.mock('../../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/ArtifactUtils')>('../../../common/utils/ArtifactUtils'),
  getArtifactContent: jest.fn(),
}));

describe('ShowArtifactVideoView', () => {
  const DUMMY_BLOB = new Blob(['unit-test'], { type: 'video/mp4' });
  const getArtifactMock = jest.fn(() => Promise.resolve(DUMMY_BLOB));

  beforeAll(() => {
    jest.spyOn(global.URL, 'createObjectURL').mockImplementation(() => 'blob://dummy-url');
    global.URL.revokeObjectURL = jest.fn();
  });

  it('shows a skeleton placeholder first, then renders the video', async () => {
    render(
      <ShowArtifactVideoView
        runUuid="run-123"
        artifactRootUri="mlflow-artifacts:/1/run-123/artifacts"
        path="foo/bar/video.mp4"
        getArtifact={getArtifactMock}
        isLoggedModelsMode={false}
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        ),
      },
    );

    expect(screen.queryByRole('video')).not.toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByLabelText('video')).toBeInTheDocument();
      expect((screen.getByLabelText('video') as HTMLVideoElement).src).toBe('blob://dummy-url');
    });
  });
  it('shows a local error when a video exceeds the streaming preview limit', async () => {
    const getArtifact = jest.fn<() => Promise<Blob>>().mockRejectedValue(new ArtifactPreviewTooLargeError());
    render(
      <ShowArtifactVideoView
        runUuid="run-123"
        artifactRootUri="mlflow-artifacts:/1/run-123/artifacts"
        path="video.mp4"
        getArtifact={getArtifact}
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        ),
      },
    );
    expect(await screen.findByText('Loading artifact failed')).toBeInTheDocument();
    expect(getArtifact).toHaveBeenCalledTimes(1);
    expect(
      screen.getByText('File is too large to preview. Maximum file size for preview: 50 MiB.'),
    ).toBeInTheDocument();
    expect(screen.queryByLabelText('video')).not.toBeInTheDocument();
  });
});
