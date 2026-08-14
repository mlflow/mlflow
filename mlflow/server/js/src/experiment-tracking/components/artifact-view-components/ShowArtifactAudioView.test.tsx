import { jest, describe, test, expect, beforeEach } from '@jest/globals';
import { act, render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import ShowArtifactAudioView from './ShowArtifactAudioView';

import { IntlProvider } from 'react-intl';
import WaveSurfer from 'wavesurfer.js';

jest.mock('wavesurfer.js', () => {
  const mWaveSurfer = {
    load: jest.fn(),
    destroy: jest.fn(),
    on: jest.fn((event, callback) => {
      if (event === 'ready') {
        // @ts-expect-error Argument of type 'unknown' is not assignable to parameter of type '() => void'
        setTimeout(callback, 0);
      }
    }),
  };
  return {
    create: jest.fn().mockReturnValue(mWaveSurfer),
  };
});

const fakeBlobUrl = 'blob:http://localhost/fake-audio';
const fakeBlob = new Blob(['fake audio data'], { type: 'audio/wav' });
const mockGetArtifact = jest.fn(() => Promise.resolve(fakeBlob));

beforeEach(() => {
  jest.clearAllMocks();
  global.URL.createObjectURL = jest.fn(() => fakeBlobUrl) as any;
  global.URL.revokeObjectURL = jest.fn() as any;
});

const minimalProps = {
  path: 'fakepath',
  runUuid: 'fakeUuid',
  getArtifact: mockGetArtifact,
};

describe('ShowArtifactAudioView tests', () => {
  test('should render with minimal props without exploding', async () => {
    await act(async () => {
      render(
        <IntlProvider locale="en">
          <ShowArtifactAudioView {...minimalProps} />
        </IntlProvider>,
      );
    });
    expect(screen.getByTestId('audio-artifact-preview')).toBeInTheDocument();
  });

  test('fetches artifact blob and passes blob URL to WaveSurfer', async () => {
    await act(async () => {
      render(
        <IntlProvider locale="en">
          <ShowArtifactAudioView {...minimalProps} />
        </IntlProvider>,
      );
    });

    await waitFor(() => {
      expect(WaveSurfer.create).toHaveBeenCalledWith(
        expect.objectContaining({
          url: fakeBlobUrl,
        }),
      );
    });
  });

  test('destroys WaveSurfer on component unmount', async () => {
    let rendered: ReturnType<typeof render> | undefined;
    await act(async () => {
      rendered = render(
        <IntlProvider locale="en">
          <ShowArtifactAudioView {...minimalProps} />
        </IntlProvider>,
      );
    });

    await waitFor(() => {
      expect(WaveSurfer.create).toHaveBeenCalled();
    });

    const mockWsInstance = jest.mocked(WaveSurfer.create).mock.results[0]?.value as ReturnType<
      typeof WaveSurfer.create
    >;

    act(() => {
      rendered!.unmount();
    });

    expect(mockWsInstance.destroy).toHaveBeenCalled();
  });

  test('shows error state when artifact fetch fails', async () => {
    const failingGetArtifact = jest.fn(() => Promise.reject(new Error('fetch failed')));

    await act(async () => {
      render(
        <IntlProvider locale="en">
          <ShowArtifactAudioView {...minimalProps} getArtifact={failingGetArtifact} />
        </IntlProvider>,
      );
    });

    await waitFor(() => {
      expect(failingGetArtifact).toHaveBeenCalled();
    });

    expect(await screen.findByText('Loading artifact failed')).toBeInTheDocument();
  });
});
