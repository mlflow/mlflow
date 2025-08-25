/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow, mount } from 'enzyme';
import ShowArtifactTextView, { prettifyArtifactText } from './ShowArtifactTextView';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { mountWithIntl } from '../../../common/utils/TestUtils.enzyme';

describe('ShowArtifactTextView', () => {
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
    commonProps = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactTextView {...commonProps} />).dive();
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactTextView {...minimalProps} />).dive();
    expect(wrapper.length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = shallow(<ShowArtifactTextView {...props} />).dive();
    setImmediate(() => {
      expect(wrapper.find('.artifact-text-view-error').length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.html).toBeUndefined();
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-text-view-loading').length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render text content when available', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('my text');
    });
    const props = { ...minimalProps, getArtifact };
    wrapper = mountWithIntl(<ShowArtifactTextView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.mlflow-ShowArtifactPage').length).toBe(1);
      expect(wrapper.find('code').length).toBe(1);
      expect(wrapper.find('code').text()).toBe('my text');
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('SyntaxHighlighter has an appropriate language prop for a python script', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('print("foo")');
    });
    const props = { path: 'fake.py', runUuid: 'fakeUuid', getArtifact, experimentId: '123' };
    wrapper = shallow(<ShowArtifactTextView {...props} />).dive();
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find(SyntaxHighlighter).first().props().language).toBe('py');
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('SyntaxHighlighter has an appropriate language prop for an MLproject file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('key: value');
    });
    const props = { path: 'MLproject', runUuid: 'fakeUuid', getArtifact, experimentId: '123' };
    wrapper = shallow(<ShowArtifactTextView {...props} />).dive();
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find(SyntaxHighlighter).first().props().language).toBe('yaml');
      done();
    });
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('SyntaxHighlighter has an appropriate language prop for an MLmodel file', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('key: value');
    });
    const props = { path: 'MLmodel', runUuid: 'fakeUuid', getArtifact, experimentId: '123' };
    wrapper = shallow(<ShowArtifactTextView {...props} />).dive();
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find(SyntaxHighlighter).first().props().language).toBe('yaml');
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

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render prettified valid json', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.resolve('{"key1": "val1", "key2": "val2"}');
    });
    const props = { path: 'fake.json', runUuid: 'fakeUuid', getArtifact, experimentId: '123' };
    wrapper = mountWithIntl(<ShowArtifactTextView {...props} />);
    setImmediate(() => {
      wrapper.update();
      expect(wrapper.find('.mlflow-ShowArtifactPage').length).toBe(1);
      expect(wrapper.find('code').length).toBe(1);
      expect(wrapper.find('code').text()).toContain('\n');
      expect(wrapper.find('code').text()).toContain('key1');
      done();
    });
  });

  test('should leave invalid json untouched', () => {
    const outputText = prettifyArtifactText('json', '{"hello');
    expect(outputText).toBe('{"hello');
  });
});
