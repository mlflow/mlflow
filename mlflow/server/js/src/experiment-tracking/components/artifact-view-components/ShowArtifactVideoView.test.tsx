import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

import ShowArtifactVideoView from './ShowArtifactVideoView';

jest.mock('../../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/ArtifactUtils')>('../../../common/utils/ArtifactUtils'),
  getArtifactContent: jest.fn(),
}));

describe('ShowArtifactVideoView', () => {
  const DUMMY_BLOB = new Blob(['unit-test'], { type: 'video/mp4' });
  const getArtifactMock = jest.fn(() => Promise.resolve(DUMMY_BLOB));

  beforeAll(() => {
    global.URL.createObjectURL = jest.fn(() => 'blob://dummy-url');
    global.URL.revokeObjectURL = jest.fn();
  });

  it('shows a skeleton placeholder first, then renders the video', async () => {
    render(
      <ShowArtifactVideoView
        runUuid="run-123"
        path="foo/bar/video.mp4"
        getArtifact={getArtifactMock}
        isLoggedModelsMode={false}
        experimentId="experiment-123"
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
});
