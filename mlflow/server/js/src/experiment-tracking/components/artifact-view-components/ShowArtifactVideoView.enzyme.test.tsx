import { act } from 'react-dom/test-utils';
import { mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';

import ShowArtifactVideoView from './ShowArtifactVideoView';

describe('ShowArtifactVideoView', () => {
  const DUMMY_BLOB = new Blob(['unit-test'], { type: 'video/mp4' });
  const getArtifactMock = jest.fn(() => Promise.resolve(DUMMY_BLOB));

  beforeAll(() => {
    global.URL.createObjectURL = jest.fn(() => 'blob://dummy-url');
  });

  it('shows a skeleton placeholder first, then renders the video', async () => {
    const wrapper = mountWithIntl(
      <ShowArtifactVideoView
        runUuid="run-123"
        path="foo/bar/video.mp4"
        getArtifact={getArtifactMock}
        isLoggedModelsMode={false}
        experimentId="experiment-123"
      />,
    );

    expect(wrapper.find('LegacySkeleton')).toHaveLength(1);
    expect(wrapper.find('video')).toHaveLength(0);

    await act(async () => Promise.resolve());
    wrapper.update();

    expect(wrapper.find('LegacySkeleton')).toHaveLength(0);
    const videoEl = wrapper.find('video');
    expect(videoEl).toHaveLength(1);
    expect(videoEl.prop('src')).toBe('blob://dummy-url');
  });
});
