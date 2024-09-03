import { mount, shallow } from 'enzyme';
import ShowArtifactAudioView from './ShowArtifactAudioView';

import { act } from 'react-dom/test-utils';
import { IntlProvider } from 'react-intl';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import WaveSurfer, { WaveSurferOptions } from 'wavesurfer.js';

jest.mock('wavesurfer.js', () => {
  const mWaveSurfer = {
    loadBlob: jest.fn(),
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

describe('ShowArtifactAudioView tests', () => {
  let wrapper: any;
  let minimalProps: any;
  let commonProps;
  let getArtifact: any;

  beforeEach(() => {
    minimalProps = {
      path: 'fakepath',
      runUuid: 'fakeUuid',
    };
    // Mock the `getArtifact` function to avoid spurious network errors
    // during testing
    getArtifact = jest.fn().mockResolvedValue(new Blob(['audio content'], { type: 'audio/mpeg' }));
    commonProps = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactAudioView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactAudioView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render loading skeleton when view is loading', () => {
    expect(wrapper.find(ArtifactViewSkeleton).length).toBe(1);
  });

  test('should render error message when error occurs', async () => {
    const getArtifactError = jest.fn().mockRejectedValue(new Error('my error text'));
    const propsWithError = { ...minimalProps, getArtifact: getArtifactError };
    // Use mount instead of shallow to trigger useEffect
    await act(async () => {
      wrapper = mount(
        <IntlProvider locale="en">
          <ShowArtifactAudioView {...propsWithError} />
        </IntlProvider>,
      );
    });
    wrapper.update();
    expect(wrapper.find(ArtifactViewErrorState).length).toBe(1);
    expect(wrapper.find(ArtifactViewSkeleton).length).toBe(0);
  });

  test('should display audio container when data is loaded', async () => {
    const getArtifact = jest.fn().mockResolvedValue(new Blob(['audio content'], { type: 'audio/mpeg' }));
    commonProps = { ...minimalProps, getArtifact };
    const propsWithError = { ...minimalProps, getArtifact: getArtifact };
    // Use mount instead of shallow to trigger useEffect
    await act(async () => {
      wrapper = mount(
        <IntlProvider locale="en">
          <ShowArtifactAudioView {...propsWithError} />
        </IntlProvider>,
      );
    });
    wrapper.update();

    expect(wrapper.find(ArtifactViewErrorState).length).toBe(0);
    expect(wrapper.find(ArtifactViewSkeleton).length).toBe(0);

    const audioContainerDiv = wrapper.find(ShowArtifactAudioView).find('div').first();
    expect(audioContainerDiv.length).toBe(1);
    expect(audioContainerDiv.getDOMNode()).toHaveStyle({ display: 'block' });
  });

  test('initializes WaveSurfer with correct parameters and calls loadBlob', async () => {
    const props = { ...minimalProps, getArtifact };
    await act(async () => {
      mount(
        <IntlProvider locale="en">
          <ShowArtifactAudioView {...props} />
        </IntlProvider>,
      );
    });

    expect(WaveSurfer.create).toHaveBeenCalledWith(
      expect.objectContaining({
        container: expect.anything(),
        waveColor: '#1890ff',
        progressColor: '#0b3574',
        height: 500,
      }),
    );

    // Since loadBlob is called asynchronously, we need to wait for it to be called
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(WaveSurfer.create({} as WaveSurferOptions).loadBlob).toHaveBeenCalledWith(expect.any(Blob));
  });

  test('destroys WaveSurfer on component unmount', async () => {
    const props = { ...minimalProps, getArtifact };
    const wrapper = mount(
      <IntlProvider locale="en">
        <ShowArtifactAudioView {...props} />
      </IntlProvider>,
    );

    wrapper.unmount();

    expect(WaveSurfer.create({} as WaveSurferOptions).destroy).toHaveBeenCalled();
  });
});
