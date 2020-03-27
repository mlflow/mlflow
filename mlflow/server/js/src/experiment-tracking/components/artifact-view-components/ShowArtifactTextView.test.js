import React from 'react';
import { shallow, mount } from 'enzyme';
import ShowArtifactTextView from './ShowArtifactTextView';

describe('ShowArtifactTextView', () => {
  let wrapper;
  let instance;
  let minimalProps;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: "fakepath",
      runUuid: "fakeUuid",
    };
    // Mock the `getArtifact` function to avoid spurious network errors
    // during testing
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve("some content");
    });
    commonProps = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactTextView {...commonProps}/>);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactTextView {...minimalProps}/>);
    expect(wrapper.length).toBe(1);
  });

  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error("my error text"));
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactTextView {...props}/>);
    setImmediate(() => {
      expect(wrapper.find(".artifact-text-view-error").length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.html).toBeUndefined();
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find(".artifact-text-view-loading").length).toBe(1);
  });

  test('should render text content when available', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('my text');
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = mount(<ShowArtifactTextView {...props}/>);
    setImmediate(() => {
      instance = wrapper.instance();
      wrapper.update();
      expect(wrapper.find('.ShowArtifactPage').length).toBe(1);
      expect(wrapper.find('.text-area').length).toBe(1);
      expect(wrapper.find('.text-area').prop('value')).toEqual('my text');
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
