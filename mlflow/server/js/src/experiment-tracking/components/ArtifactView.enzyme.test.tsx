/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { jest, describe, beforeEach, afterEach, test, expect } from '@jest/globals';
import React from 'react';
import { DesignSystemProvider, Typography } from '@databricks/design-system';
import { shallowWithIntl, mountWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.enzyme';
import { ArtifactView, ArtifactViewImpl } from './ArtifactView';
import ShowArtifactTextView from './artifact-view-components/ShowArtifactTextView';
import ShowArtifactImageView from './artifact-view-components/ShowArtifactImageView';
import { LazyShowArtifactMapView } from './artifact-view-components/LazyShowArtifactMapView';
import ShowArtifactHtmlView from './artifact-view-components/ShowArtifactHtmlView';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { mockModelVersionDetailed } from '../../model-registry/test-utils';
import { ModelVersionStatus, Stages } from '../../model-registry/constants';
import { Provider } from 'react-redux';
import { BrowserRouter } from '../../common/utils/RoutingUtils';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import Utils from '../../common/utils/Utils';
import { getArtifactBlob } from '../../common/utils/ArtifactUtils';
import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { MlflowService } from '../sdk/MlflowService';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { ServerInfoProvider } from '../hooks/useServerInfo';

const { Text } = Typography;

// Mock these methods because js-dom doesn't implement window.Request
jest.mock('../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/ArtifactUtils')>('../../common/utils/ArtifactUtils'),
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
  getArtifactContent: jest.fn().mockResolvedValue(),
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
  getArtifactBytesContent: jest.fn().mockResolvedValue(),
  getArtifactBlob: jest
    .fn<() => Promise<Blob>>()
    .mockResolvedValue(new Blob(['dummy content'], { type: 'text/plain' })),
}));

describe('ArtifactView', () => {
  let wrapper;
  let minimalProps: any;
  let minimalStore: any;
  let minimalEntities: any;
  const mockStore = configureStore([thunk, promiseMiddleware()]);
  const getMockStore = (rootNode: any) => {
    return mockStore({
      entities: {
        ...minimalEntities,
        artifactsByRunUuid: { fakeUuid: rootNode },
      },
    });
  };
  const getWrapper = (fakeStore: any, mockProps: any) =>
    mountWithIntl(
      <Provider store={fakeStore}>
        <DesignSystemProvider>
          <BrowserRouter>
            <ArtifactView {...mockProps} />
          </BrowserRouter>
        </DesignSystemProvider>
      </Provider>,
    );
  const getWrapperWithServerInfo = (fakeStore: any, mockProps: any, multipartDownloadsEnabled: boolean) => {
    const queryClient = new QueryClient();
    queryClient.setQueryData(['serverInfo'], {
      store_type: null,
      workspaces_enabled: false,
      trace_archival_enabled: false,
      multipart_uploads_enabled: false,
      multipart_downloads_enabled: multipartDownloadsEnabled,
    });

    return mountWithIntl(
      <QueryClientProvider client={queryClient}>
        <ServerInfoProvider>
          <Provider store={fakeStore}>
            <DesignSystemProvider>
              <BrowserRouter>
                <ArtifactView {...mockProps} />
              </BrowserRouter>
            </DesignSystemProvider>
          </Provider>
        </ServerInfoProvider>
      </QueryClientProvider>,
    );
  };
  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    jest
      .spyOn(global, 'fetch')
      // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
      .mockImplementation(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
    const node = getTestArtifactNode();
    minimalProps = {
      runUuid: 'fakeUuid',
      artifactNode: node,
      artifactRootUri: 'test_root',
      listArtifactsApi: jest.fn(() => Promise.resolve({})),
      modelVersionsBySource: {},
      handleActiveNodeChange: jest.fn(),
    };
    minimalEntities = {
      modelByName: {},
      artifactsByRunUuid: { fakeUuid: node },
      artifactRootUriByRunUuid: { fakeUuid: 'test_root' },
      modelVersionsByModel: {},
    };
    minimalStore = mockStore({
      entities: minimalEntities,
    });

    if (jest.isMockFunction(Utils.isModelRegistryEnabled)) {
      jest.mocked(Utils.isModelRegistryEnabled).mockRestore();
    }
  });
  const getTestArtifactNode = () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const file1 = new ArtifactNode(false, { path: 'file1', is_dir: false, file_size: '159' });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const dir1 = new ArtifactNode(false, { path: 'dir1', is_dir: true });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const dir2 = new ArtifactNode(false, { path: 'dir2', is_dir: true });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const file2 = new ArtifactNode(false, { path: 'dir1/file2', is_dir: false, file_size: '67' });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const file3 = new ArtifactNode(false, { path: 'dir1/file3', is_dir: false, file_size: '123' });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const file4 = new ArtifactNode(false, { path: 'dir2/file4', is_dir: false, file_size: '67' });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const file5 = new ArtifactNode(false, { path: 'dir2/MLmodel', is_dir: false, file_size: '67' });
    dir1.setChildren([file2.fileInfo, file3.fileInfo]);
    dir2.setChildren([file4.fileInfo, file5.fileInfo]);
    rootNode.children = { file1, dir1, dir2 };
    return rootNode;
  };
  test('should render with minimal props without exploding', () => {
    wrapper = shallowWithIntl(<ArtifactViewImpl {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });
  test('should render NoArtifactView when no artifacts are present', () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const emptyNode = new ArtifactNode(true, undefined);
    const props = { ...minimalProps, artifactNode: emptyNode };
    wrapper = getWrapper(getMockStore(emptyNode), props);
    expect(wrapper.find('Empty')).toHaveLength(1);
  });
  test('should render text file in text artifact view', () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const textFile = new ArtifactNode(false, {
      path: 'file1.txt',
      is_dir: false,
      file_size: '159',
    });
    rootNode.setChildren([textFile.fileInfo]);
    wrapper = getWrapper(getMockStore(rootNode), minimalProps);
    const textFileElement = wrapper.find('NodeHeader').at(0);
    textFileElement.simulate('click');
    expect(wrapper.find(ShowArtifactTextView)).toHaveLength(1);
  });
  test('should render image file in image artifact view', () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const imageFile = new ArtifactNode(false, {
      path: 'file1.png',
      is_dir: false,
      file_size: '159',
    });
    rootNode.setChildren([imageFile.fileInfo]);
    wrapper = getWrapper(getMockStore(rootNode), minimalProps);
    const imageFileElement = wrapper.find('NodeHeader').at(0);
    imageFileElement.simulate('click');
    expect(wrapper.find(ShowArtifactImageView)).toHaveLength(1);
  });
  test('should render HTML file in HTML artifact view', () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const htmlFile = new ArtifactNode(false, {
      path: 'file1.html',
      is_dir: false,
      file_size: '159',
    });
    rootNode.setChildren([htmlFile.fileInfo]);
    wrapper = getWrapper(getMockStore(rootNode), minimalProps);
    const htmlFileElement = wrapper.find('NodeHeader').at(0);
    htmlFileElement.simulate('click');
    expect(wrapper.find(ShowArtifactHtmlView)).toHaveLength(1);
  });
  test('should render geojson file in map artifact view', () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 2.
    const geojsonFile = new ArtifactNode(false, {
      path: 'file1.geojson',
      is_dir: false,
      file_size: '159',
    });
    rootNode.setChildren([geojsonFile.fileInfo]);
    wrapper = getWrapper(getMockStore(rootNode), minimalProps);
    const geojsonFileElement = wrapper.find('NodeHeader').at(0);
    geojsonFileElement.simulate('click');
    expect(wrapper.find(LazyShowArtifactMapView)).toHaveLength(1);
  });
  describe('artifact download', () => {
    let assignMock: jest.Mock;
    let originalLocation: Location;
    let originalRevokeObjectURL: any;
    let createObjectURLSpy: any;
    let revokeObjectURLMock: jest.Mock;
    let anchor: any;
    let presignedSpy: any;
    let proxiedPresignedSpy: any;

    const getImplInstance = (props: any = {}, options: { cachedMultipartDownloadsEnabled?: boolean } = {}) => {
      const rootNode = new ArtifactNode(true, undefined, undefined);
      rootNode.isLoaded = true;
      const textFile = new ArtifactNode(
        false,
        {
          path: 'summary.txt',
          is_dir: false,
          file_size: '100',
        },
        undefined,
      );
      rootNode.setChildren([textFile.fileInfo]);
      const mockProps = {
        ...minimalProps,
        artifactRootUri: 's3://bucket/0/fakeUuid/artifacts',
        multipartDownloadsEnabled: true,
        ...props,
      };
      wrapper =
        options.cachedMultipartDownloadsEnabled === undefined
          ? getWrapper(getMockStore(rootNode), mockProps)
          : getWrapperWithServerInfo(getMockStore(rootNode), mockProps, options.cachedMultipartDownloadsEnabled);
      wrapper.find('NodeHeader').at(0).simulate('click');
      wrapper.update();
      // Mock the DOM download plumbing only after enzyme has mounted the component:
      // enzyme itself needs the real document.createElement to create its container.
      createObjectURLSpy = jest.spyOn(URL, 'createObjectURL').mockReturnValue('blob:fake-url');
      // jsdom does not implement URL.revokeObjectURL (and the test setup does not polyfill
      // it), so jest.spyOn cannot be used here; the original value is restored in afterEach.
      revokeObjectURLMock = jest.fn();
      URL.revokeObjectURL = revokeObjectURLMock;
      anchor = { href: '', download: '', click: jest.fn() };
      jest.spyOn(document, 'createElement').mockReturnValue(anchor);
      jest.spyOn(document.body, 'appendChild').mockImplementation(() => anchor);
      jest.spyOn(document.body, 'removeChild').mockImplementation(() => anchor);
      return wrapper.find('ArtifactViewImpl').instance() as any;
    };

    const expectBlobDownload = (expectedUrlPart: string) => {
      expect(getArtifactBlob).toHaveBeenCalledWith(expect.stringContaining(expectedUrlPart));
      expect(createObjectURLSpy).toHaveBeenCalled();
      expect(anchor.download).toBe('summary.txt');
      expect(anchor.click).toHaveBeenCalled();
      expect(revokeObjectURLMock).toHaveBeenCalledWith('blob:fake-url');
    };

    beforeEach(() => {
      originalLocation = window.location;
      originalRevokeObjectURL = (URL as any).revokeObjectURL;
      assignMock = jest.fn();
      // The download navigates the top-level page to a cross-origin (cloud storage) URL,
      // which no router test utility models — mock `location.assign` directly.
      // eslint-disable-next-line @databricks/no-mock-location
      Object.defineProperty(window, 'location', {
        value: { ...originalLocation, assign: assignMock },
        writable: true,
      });
      presignedSpy = jest.spyOn(MlflowService, 'createPresignedDownloadUrl');
      proxiedPresignedSpy = jest.spyOn(MlflowService, 'getMlflowArtifactsPresignedDownloadUrl');
      jest.mocked(getArtifactBlob).mockClear();
    });

    afterEach(() => {
      jest.restoreAllMocks();
      (URL as any).revokeObjectURL = originalRevokeObjectURL;
      // eslint-disable-next-line @databricks/no-mock-location
      Object.defineProperty(window, 'location', { value: originalLocation, writable: true });
    });

    test('should navigate to the presigned URL when the server provides one', async () => {
      presignedSpy.mockResolvedValue({ presigned_url: 'https://s3.example.com/signed', file_size: 100 });

      const implInstance = getImplInstance();
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(presignedSpy).toHaveBeenCalledWith({ run_id: 'fakeUuid', path: 'summary.txt' });
      expect(assignMock).toHaveBeenCalledWith('https://s3.example.com/signed');
      expect(getArtifactBlob).not.toHaveBeenCalled();
    });

    test('should navigate to the proxied artifact presigned URL when server-info enables multipart downloads', async () => {
      proxiedPresignedSpy.mockResolvedValue({ url: 'https://s3.example.com/proxied-signed', file_size: 100 });

      const implInstance = getImplInstance({ artifactRootUri: 'mlflow-artifacts:/0/fakeUuid/artifacts' });
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(proxiedPresignedSpy).toHaveBeenCalledWith('0/fakeUuid/artifacts/summary.txt');
      expect(presignedSpy).not.toHaveBeenCalled();
      expect(assignMock).toHaveBeenCalledWith('https://s3.example.com/proxied-signed');
      expect(getArtifactBlob).not.toHaveBeenCalled();
    });

    test('should ignore the authority for mlflow-artifacts URIs', async () => {
      proxiedPresignedSpy.mockResolvedValue({ url: 'https://s3.example.com/proxied-signed', file_size: 100 });

      const implInstance = getImplInstance({
        artifactRootUri: 'mlflow-artifacts://tracking.example.com/0/fakeUuid/artifacts',
      });
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(proxiedPresignedSpy).toHaveBeenCalledWith('0/fakeUuid/artifacts/summary.txt');
      expect(presignedSpy).not.toHaveBeenCalled();
      expect(assignMock).toHaveBeenCalledWith('https://s3.example.com/proxied-signed');
      expect(getArtifactBlob).not.toHaveBeenCalled();
    });

    test('should use cached server-info when deciding proxied artifact presigned downloads', async () => {
      proxiedPresignedSpy.mockResolvedValue({ url: 'https://s3.example.com/proxied-signed', file_size: 100 });

      const implInstance = getImplInstance(
        {
          artifactRootUri: 'mlflow-artifacts:/0/my%20run/artifacts',
          multipartDownloadsEnabled: undefined,
        },
        { cachedMultipartDownloadsEnabled: true },
      );
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(proxiedPresignedSpy).toHaveBeenCalledWith('0/my run/artifacts/summary.txt');
      expect(presignedSpy).not.toHaveBeenCalled();
      expect(assignMock).toHaveBeenCalledWith('https://s3.example.com/proxied-signed');
      expect(getArtifactBlob).not.toHaveBeenCalled();
    });

    test('should decode and preserve proxied HTTP artifact root paths before route encoding', async () => {
      proxiedPresignedSpy.mockResolvedValue({ url: 'https://s3.example.com/proxied-signed', file_size: 100 });

      const implInstance = getImplInstance({
        artifactRootUri:
          'https://mlflow.example.com/prefix/api/2.0/mlflow-artifacts/artifacts/0/my%20run/api/2.0/mlflow-artifacts/artifacts/artifacts',
      });
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(proxiedPresignedSpy).toHaveBeenCalledWith(
        '0/my run/api/2.0/mlflow-artifacts/artifacts/artifacts/summary.txt',
      );
      expect(presignedSpy).not.toHaveBeenCalled();
      expect(assignMock).toHaveBeenCalledWith('https://s3.example.com/proxied-signed');
      expect(getArtifactBlob).not.toHaveBeenCalled();
    });

    test('should use the run-scoped presigned URL for direct artifact roots when multipart downloads are disabled', async () => {
      presignedSpy.mockResolvedValue({ presigned_url: 'https://s3.example.com/signed', file_size: 100 });

      const implInstance = getImplInstance({ multipartDownloadsEnabled: false });
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(presignedSpy).toHaveBeenCalledWith({ run_id: 'fakeUuid', path: 'summary.txt' });
      expect(proxiedPresignedSpy).not.toHaveBeenCalled();
      expect(assignMock).toHaveBeenCalledWith('https://s3.example.com/signed');
      expect(getArtifactBlob).not.toHaveBeenCalled();
    });

    test('should use the proxied download for proxied artifact roots when server-info disables multipart downloads', async () => {
      presignedSpy.mockResolvedValue({ presigned_url: 'https://s3.example.com/signed', file_size: 100 });
      proxiedPresignedSpy.mockResolvedValue({ url: 'https://s3.example.com/proxied-signed', file_size: 100 });

      const implInstance = getImplInstance({
        artifactRootUri: 'mlflow-artifacts:/0/fakeUuid/artifacts',
        multipartDownloadsEnabled: false,
      });
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(presignedSpy).not.toHaveBeenCalled();
      expect(proxiedPresignedSpy).not.toHaveBeenCalled();
      expect(assignMock).not.toHaveBeenCalled();
      expectBlobDownload('get-artifact?path=summary.txt&run_uuid=fakeUuid');
    });

    test('should use the proxied download for HTTP proxied artifact roots when server-info disables multipart downloads', async () => {
      presignedSpy.mockResolvedValue({ presigned_url: 'https://s3.example.com/signed', file_size: 100 });
      proxiedPresignedSpy.mockResolvedValue({ url: 'https://s3.example.com/proxied-signed', file_size: 100 });

      const implInstance = getImplInstance({
        artifactRootUri: 'https://mlflow.example.com/api/2.0/mlflow-artifacts/artifacts/0/fakeUuid/artifacts',
        multipartDownloadsEnabled: false,
      });
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(presignedSpy).not.toHaveBeenCalled();
      expect(proxiedPresignedSpy).not.toHaveBeenCalled();
      expect(assignMock).not.toHaveBeenCalled();
      expectBlobDownload('get-artifact?path=summary.txt&run_uuid=fakeUuid');
    });

    test.each([400, 404, 501, 503])(
      'should fall back to the proxied download when the presigned request fails with %s',
      async (status) => {
        presignedSpy.mockRejectedValue(new ErrorWrapper('unavailable', status));

        const implInstance = getImplInstance();
        await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

        expect(assignMock).not.toHaveBeenCalled();
        expectBlobDownload('get-artifact?path=summary.txt&run_uuid=fakeUuid');
      },
    );

    test.each([400, 404, 501, 503])(
      'should fall back to the proxied download when the proxied presigned request fails with %s',
      async (status) => {
        proxiedPresignedSpy.mockRejectedValue(new ErrorWrapper('unavailable', status));

        const implInstance = getImplInstance({ artifactRootUri: 'mlflow-artifacts:/0/fakeUuid/artifacts' });
        await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

        expect(presignedSpy).not.toHaveBeenCalled();
        expect(assignMock).not.toHaveBeenCalled();
        expectBlobDownload('get-artifact?path=summary.txt&run_uuid=fakeUuid');
      },
    );

    test('should fail closed without fallback when the presigned request is denied with 403', async () => {
      const notifySpy = jest.spyOn(Utils, 'logErrorAndNotifyUser').mockImplementation(() => {});
      presignedSpy.mockRejectedValue(new ErrorWrapper('permission denied', 403));

      const implInstance = getImplInstance();
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(notifySpy).toHaveBeenCalled();
      expect(assignMock).not.toHaveBeenCalled();
      expect(getArtifactBlob).not.toHaveBeenCalled();
    });

    test('should notify the user when the proxied fallback download itself fails', async () => {
      const notifySpy = jest.spyOn(Utils, 'logErrorAndNotifyUser').mockImplementation(() => {});
      presignedSpy.mockRejectedValue(new ErrorWrapper('older server', 404));

      const implInstance = getImplInstance();
      jest.mocked(getArtifactBlob).mockRejectedValueOnce(new ErrorWrapper('missing artifact', 404));
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(getArtifactBlob).toHaveBeenCalled();
      expect(notifySpy).toHaveBeenCalled();
      expect(assignMock).not.toHaveBeenCalled();
      expect(createObjectURLSpy).not.toHaveBeenCalled();
    });

    test('should use the proxied download when the presigned URL requires request headers', async () => {
      presignedSpy.mockResolvedValue({
        presigned_url: 'https://s3.example.com/signed',
        headers: { 'x-required-header': 'value' },
      });

      const implInstance = getImplInstance();
      await implInstance.onDownloadClick('fakeUuid', 'summary.txt');

      expect(assignMock).not.toHaveBeenCalled();
      expectBlobDownload('get-artifact?path=summary.txt&run_uuid=fakeUuid');
    });

    test('should download logged-model artifacts via the proxied path without a presigned request', async () => {
      const implInstance = getImplInstance();
      await implInstance.onDownloadClick(undefined, 'summary.txt', 'model-123');

      expect(presignedSpy).not.toHaveBeenCalled();
      expect(assignMock).not.toHaveBeenCalled();
      expectBlobDownload('model-123');
    });
  });
});
