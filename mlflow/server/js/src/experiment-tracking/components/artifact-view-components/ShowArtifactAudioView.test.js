import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactAudioView from './ShowArtifactAudioView';

describe('ShowArtifactAudioView', () => {
  let wrapper;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakePath',
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
});
