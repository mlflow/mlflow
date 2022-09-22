import React from 'react';
import { shallow, mount } from 'enzyme';
import ShowArtifactAudioView from './ShowArtifactAudioView';
import WaveformView from './WaveformView';

function minimalWaveFile() {
  const b64 =
    'UklGRiTmAwBXQVZFZm10IBAAAAABAAEA5FcAAMivAAACABAAZGF0YQDmAwDM/3j/ef+k/z0ARgDx/5D/ff/n/w==';
  const str = window.atob(b64);
  const buffer = new ArrayBuffer(str.length);
  const bufView = new Uint8Array(buffer);
  for (let i = 0, strLen = str.length; i < strLen; i++) {
    bufView[i] = str.charCodeAt(i);
  }
  return buffer;
}

describe('ShowArtifactAudioView', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakePath.wav',
      runUuid: 'fakeUuid',
    };
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('some content');
    });
    commonProps = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactAudioView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactAudioView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render waveform from array buffer', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve(minimalWaveFile());
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mount(<ShowArtifactAudioView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find(WaveformView).length).toBe(1);
      done();
    });
  });
});
