import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactAudioView from './ShowArtifactAudioView';

describe('ShowArtifactAudioView', () => {
  let wrapper;
  let instance;
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

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-audio-view-loading').length).toBe(1);
  });

  test('should render error text when artifact cannot be loaded or when artifact can be loaded but audio cannot be decoded', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactAudioView {...props} />);
    setImmediate(() => {
      expect(wrapper.find('.artifact-audio-view-error').length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should fetch artifacts on component update', () => {
    instance = wrapper.instance();
    instance.fetchArtifacts = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchArtifacts).toBeCalled();
    expect(instance.props.getArtifact).toBeCalled();
  });
});
