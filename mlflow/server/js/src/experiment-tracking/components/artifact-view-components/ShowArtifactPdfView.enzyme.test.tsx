/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactPdfView from './ShowArtifactPdfView';
import { setupReactPDFWorker } from './utils/setupReactPDFWorker';

jest.mock('react-pdf', () => ({
  Document: () => null,
  Page: () => null,
  pdfjs: { GlobalWorkerOptions: { workerSrc: '' } },
}));
jest.mock('./utils/setupReactPDFWorker', () => ({
  setupReactPDFWorker: jest.fn(),
}));

describe('ShowArtifactPdfView', () => {
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
    wrapper = shallow(<ShowArtifactPdfView {...commonProps} />);
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ShowArtifactPdfView {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should render loading text when view is loading', () => {
    instance = wrapper.instance();
    instance.setState({ loading: true });
    expect(wrapper.find('.artifact-pdf-view-loading').length).toBe(1);
  });

  // eslint-disable-next-line jest/no-done-callback -- TODO(FEINF-1337)
  test('should render error message when error occurs', (done) => {
    const getArtifact = jest.fn((artifactLocation) => {
      return Promise.reject(new Error('my error text'));
    });
    const props = { ...minimalProps, getArtifact: getArtifact };
    wrapper = shallow(<ShowArtifactPdfView {...props} />);
    setImmediate(() => {
      expect(wrapper.find('.artifact-pdf-view-error').length).toBe(1);
      expect(wrapper.instance().state.loading).toBe(false);
      expect(wrapper.instance().state.html).toBeUndefined();
      expect(wrapper.instance().state.error).toBeDefined();
      done();
    });
  });

  test('should render PDF in container', () => {
    wrapper.setProps({ path: 'fake.pdf', runUuid: 'fakeRunId' });
    wrapper.setState({ loading: false });
    expect(wrapper.find('.mlflow-pdf-outer-container')).toHaveLength(1);
    expect(wrapper.find('.mlflow-pdf-viewer')).toHaveLength(1);
    expect(wrapper.find('.mlflow-paginator')).toHaveLength(1);
    expect(wrapper.find('.mlflow-document')).toHaveLength(1);
  });

  test('should call fetchPdf on component update', () => {
    instance = wrapper.instance();
    instance.fetchPdf = jest.fn();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchPdf).toHaveBeenCalledTimes(1);
  });
});
