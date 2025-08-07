/* eslint-disable jest/no-standalone-expect */
/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import Iframe from 'react-iframe';

describe('ShowArtifactHtmlView', () => {
  let wrapper: any;
  let instance;
  let minimalProps: any;
  let commonProps;

  beforeEach(() => {
    minimalProps = {
      path: 'fakepath',
      runUuid: 'fakeUuid',
    };
    // Mock the `getArtifact` function to avoid spurious network errors
    // during testing
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('some content');
    });
    commonProps = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactHtmlView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactHtmlView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-html-view-loading').length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactHtmlView {...props} />);
    setImmediate(() => {
      expect(wrapper.find('.artifact-html-view-error').length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.html).toBeFalsy();
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should render html content in IFrame', (done: jest.DoneCallback) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('my text');
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactHtmlView {...props} />);
    setImmediate(() => {
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.html).toBeTruthy();
      expect(wrapper.instance().state.error).toBeUndefined();
      expect(wrapper.find(Iframe).length).toBe(1);
      expect(wrapper.find(Iframe).first().dive().prop('id')).toEqual('html');
      done();
    });
  });

  test('should fetch artifacts on component update', () => {
    instance = wrapper.instance();
    instance.fetchArtifacts = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchArtifacts).toHaveBeenCalled();
    expect(instance.props.getArtifact).toHaveBeenCalled();
  });
});
