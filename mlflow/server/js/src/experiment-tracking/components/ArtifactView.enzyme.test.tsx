/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { Typography } from '@databricks/design-system';
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

const { Text } = Typography;

// Mock these methods because js-dom doesn't implement window.Request
jest.mock('../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/ArtifactUtils')>('../../common/utils/ArtifactUtils'),
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
  getArtifactContent: jest.fn().mockResolvedValue(),
  // @ts-expect-error TS(2554): Expected 1 arguments, but got 0.
  getArtifactBytesContent: jest.fn().mockResolvedValue(),
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
        <BrowserRouter>
          <ArtifactView {...mockProps} />
        </BrowserRouter>
      </Provider>,
    );
  beforeEach(() => {
    // TODO: remove global fetch mock by explicitly mocking all the service API calls
    // @ts-expect-error TS(2322): Type 'Mock<Promise<{ ok: true; status: number; tex... Remove this comment to see the full error message
    global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, text: () => Promise.resolve('') }));
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
});
