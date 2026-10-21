/* eslint-disable jest/no-standalone-expect */
/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactHtmlView from './ShowArtifactHtmlView';
import { ArtifactPreviewTooLargeError } from '../../../common/utils/ArtifactUtils';
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

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render html content in IFrame', (done: any) => {
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
    // @ts-expect-error -- TODO(FEINF-4162)
    jest.spyOn(instance, 'fetchArtifacts').mockImplementation();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchArtifacts).toHaveBeenCalled();
    expect(instance.props.getArtifact).toHaveBeenCalled();
  });
  test('recovers after an oversized preview when a valid artifact is selected', async () => {
    const content = '<p>current</p>';
    const getArtifact = jest
      .fn<() => Promise<string>>()
      .mockRejectedValueOnce(new ArtifactPreviewTooLargeError())
      .mockResolvedValueOnce(content);
    wrapper = shallow(<ShowArtifactHtmlView {...minimalProps} getArtifact={getArtifact} />);
    await new Promise(setImmediate);
    expect(wrapper.find('.artifact-html-view-error')).toHaveLength(1);
    wrapper.setProps({ path: 'valid.html' });
    await new Promise(setImmediate);
    expect(wrapper.instance().state.error).toBeUndefined();
    expect(wrapper.instance().state.loading).toBe(false);
    expect(wrapper.instance().state.html).toBe(content);
  });

  test.each([false, true])('ignores a stale artifact completion (failure: %s)', async (failure) => {
    let resolvePrevious!: (content: string) => void;
    let rejectPrevious!: (error: Error) => void;
    const previous = new Promise<string>((resolve, reject) => {
      resolvePrevious = resolve;
      rejectPrevious = reject;
    });
    const content = '<p>current</p>';
    const getArtifact = jest.fn<() => Promise<string>>().mockReturnValueOnce(previous).mockResolvedValueOnce(content);
    wrapper = shallow(<ShowArtifactHtmlView {...minimalProps} getArtifact={getArtifact} />);
    wrapper.setProps({ path: 'current.html' });
    await new Promise(setImmediate);
    if (failure) {
      rejectPrevious(new ArtifactPreviewTooLargeError());
    } else {
      resolvePrevious('stale');
    }
    await new Promise(setImmediate);
    expect(wrapper.instance().state.error).toBeUndefined();
    expect(wrapper.instance().state.loading).toBe(false);
    expect(wrapper.instance().state.html).toBe(content);
  });

  test.each([{ artifactRootUri: 's3://another-root' }, { isLoggedModelsMode: true, loggedModelId: 'another-model' }])(
    'refetches the same path when its source changes: %j',
    async (source) => {
      const getArtifact = jest.fn<() => Promise<string>>().mockResolvedValue('<p>current</p>');
      wrapper = shallow(<ShowArtifactHtmlView {...minimalProps} getArtifact={getArtifact} />);
      await new Promise(setImmediate);
      wrapper.setProps(source);
      await new Promise(setImmediate);
      expect(getArtifact).toHaveBeenCalledTimes(2);
    },
  );
});
