/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import React from 'react';
import { shallow } from 'enzyme';
import ShowArtifactPdfView from './ShowArtifactPdfView';
import { ArtifactPreviewTooLargeError } from '../../../common/utils/ArtifactUtils';
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
    // @ts-expect-error -- TODO(FEINF-4162)
    jest.spyOn(instance, 'fetchPdf').mockImplementation();
    wrapper.setProps({ path: 'newpath', runUuid: 'newRunId' });
    expect(instance.fetchPdf).toHaveBeenCalledTimes(1);
  });
  test('recovers after an oversized preview when a valid artifact is selected', async () => {
    const content = new Uint8Array([1, 2, 3]).buffer;
    const getArtifact = jest
      .fn<() => Promise<ArrayBuffer>>()
      .mockRejectedValueOnce(new ArtifactPreviewTooLargeError())
      .mockResolvedValueOnce(content);
    wrapper = shallow(<ShowArtifactPdfView {...minimalProps} getArtifact={getArtifact} />);
    await new Promise(setImmediate);
    expect(wrapper.find('.artifact-pdf-view-error')).toHaveLength(1);
    wrapper.setProps({ path: 'valid.pdf' });
    await new Promise(setImmediate);
    expect(wrapper.instance().state.error).toBeUndefined();
    expect(wrapper.instance().state.loading).toBe(false);
    expect(wrapper.instance().state.pdfData.data).toBe(content);
  });

  test.each([false, true])('ignores a stale artifact completion (failure: %s)', async (failure) => {
    let resolvePrevious!: (content: ArrayBuffer) => void;
    let rejectPrevious!: (error: Error) => void;
    const previous = new Promise<ArrayBuffer>((resolve, reject) => {
      resolvePrevious = resolve;
      rejectPrevious = reject;
    });
    const content = new Uint8Array([1, 2, 3]).buffer;
    const getArtifact = jest
      .fn<() => Promise<ArrayBuffer>>()
      .mockReturnValueOnce(previous)
      .mockResolvedValueOnce(content);
    wrapper = shallow(<ShowArtifactPdfView {...minimalProps} getArtifact={getArtifact} />);
    wrapper.setProps({ path: 'current.pdf' });
    await new Promise(setImmediate);
    if (failure) {
      rejectPrevious(new ArtifactPreviewTooLargeError());
    } else {
      resolvePrevious(new ArrayBuffer(0));
    }
    await new Promise(setImmediate);
    expect(wrapper.instance().state.error).toBeUndefined();
    expect(wrapper.instance().state.loading).toBe(false);
    expect(wrapper.instance().state.pdfData.data).toBe(content);
  });

  test.each([{ artifactRootUri: 's3://another-root' }, { isLoggedModelsMode: true, loggedModelId: 'another-model' }])(
    'refetches the same path when its source changes: %j',
    async (source) => {
      const getArtifact = jest.fn<() => Promise<ArrayBuffer>>().mockResolvedValue(new Uint8Array([1, 2, 3]).buffer);
      wrapper = shallow(<ShowArtifactPdfView {...minimalProps} getArtifact={getArtifact} />);
      await new Promise(setImmediate);
      wrapper.setProps(source);
      await new Promise(setImmediate);
      expect(getArtifact).toHaveBeenCalledTimes(2);
    },
  );
});
