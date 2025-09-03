import { render, screen } from '../../../common/utils/TestUtils.react18';
import ShowArtifactAudioView from './ShowArtifactAudioView';

import { IntlProvider } from 'react-intl';
import type { WaveSurferOptions } from 'wavesurfer.js';
import WaveSurfer from 'wavesurfer.js';

jest.mock('wavesurfer.js', () => {
  const mWaveSurfer = {
    load: jest.fn(),
    destroy: jest.fn(),
    on: jest.fn((event, callback) => {
      if (event === 'ready') {
        setTimeout(callback, 0); // Simulate async event
      }
    }),
  };
  return {
    create: jest.fn().mockReturnValue(mWaveSurfer),
  };
});

const minimalProps = {
  path: 'fakepath',
  runUuid: 'fakeUuid',
};

describe('ShowArtifactAudioView tests', () => {
  test('should render with minimal props without exploding', () => {
    render(
      <IntlProvider locale="en">
        <ShowArtifactAudioView {...minimalProps} />
      </IntlProvider>,
    );
    expect(screen.getByTestId('audio-artifact-preview')).toBeInTheDocument();
  });

  test('destroys WaveSurfer on component unmount', async () => {
    const { unmount } = render(
      <IntlProvider locale="en">
        <ShowArtifactAudioView {...minimalProps} />
      </IntlProvider>,
    );

    unmount();

    expect(WaveSurfer.create({} as WaveSurferOptions).destroy).toHaveBeenCalled();
  });
});
