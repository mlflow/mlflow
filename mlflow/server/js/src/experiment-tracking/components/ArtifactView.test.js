import React from 'react';
import { shallow, mount } from 'enzyme';
import { ArtifactView } from './ArtifactView';
import ShowArtifactTextView from './artifact-view-components/ShowArtifactTextView';
import ShowArtifactImageView from './artifact-view-components/ShowArtifactImageView';
import ShowArtifactMapView from './artifact-view-components/ShowArtifactMapView';
import ShowArtifactHtmlView from './artifact-view-components/ShowArtifactHtmlView';
import { ArtifactNode } from '../utils/ArtifactUtils';
import {
  mockModelVersionDetailed,
} from '../../model-registry/test-utils';
import { ModelVersionStatus, Stages } from '../../model-registry/constants';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import configureStore from 'redux-mock-store';
import promiseMiddleware from 'redux-promise-middleware';
import thunk from 'redux-thunk';
import Utils from '../../common/utils/Utils';

describe('ArtifactView', () => {
  let wrapper;
  let minimalProps;
  let minimalStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);

  beforeEach(() => {
    const node = getTestArtifactNode();
    minimalProps = {
      runUuid: "fakeUuid",
      artifactNode: node,
      artifactRootUri: "test_root",
      listArtifactsApi: jest.fn(() => Promise.resolve({})),
      modelVersionsBySource: {},
      handleActiveNodeChange: jest.fn(),
    };
    minimalStore = mockStore({
      entities: {
        modelByName: {},
      },
    });
    wrapper = shallow(<ArtifactView {...minimalProps}/>);
  });

  const getTestArtifactNode = () => {
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    const file1 = new ArtifactNode(false, { path: "file1", is_dir: false, file_size: "159" });
    const dir1 = new ArtifactNode(false, { path: "dir1", is_dir: true });
    const file2 = new ArtifactNode(false, { path: "dir1/file2", is_dir: false, file_size: "67" });
    const file3 = new ArtifactNode(false, { path: "dir1/file3", is_dir: false, file_size: "123" });
    dir1.setChildren([file2.fileInfo, file3.fileInfo]);
    rootNode.children = { file1, dir1 };
    return rootNode;
  };

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<ArtifactView {...minimalProps}/>);
    expect(wrapper.length).toBe(1);
  });

  test('should render NoArtifactView when no artifacts are present', () => {
    const emptyNode = new ArtifactNode(true, undefined);
    const props = {...minimalProps, artifactNode: emptyNode};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    expect(wrapper.find('.no-artifacts')).toHaveLength(1);
  });

  test('should render selected file artifact', () => {
    const props = {...minimalProps};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const file1Element = wrapper.find('NodeHeader').at(0);
    file1Element.simulate('click');
    expect(wrapper.find('.artifact-info-path').html()).toContain('test_root/file1');
    expect(wrapper.find('.artifact-info-size').html()).toContain('159B');
    // Selecting a file artifact should display a download link
    expect(wrapper.find('.artifact-info-link')).toHaveLength(1);
  });

  test('should render text file in text artifact view', () => {
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    const textFile = new ArtifactNode(
      false,
      { path: "file1.txt", is_dir: false, file_size: "159" }
    );
    rootNode.setChildren([textFile.fileInfo]);

    const props = {...minimalProps, artifactNode: rootNode};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const textFileElement = wrapper.find('NodeHeader').at(0);
    textFileElement.simulate('click');
    expect(wrapper.find(ShowArtifactTextView)).toHaveLength(1);
  });

  test('should render image file in image artifact view', () => {
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    const imageFile = new ArtifactNode(
      false,
      { path: "file1.png", is_dir: false, file_size: "159" }
    );
    rootNode.setChildren([imageFile.fileInfo]);

    const props = {...minimalProps, artifactNode: rootNode};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const imageFileElement = wrapper.find('NodeHeader').at(0);
    imageFileElement.simulate('click');
    expect(wrapper.find(ShowArtifactImageView)).toHaveLength(1);
  });

  test('should render HTML file in HTML artifact view', () => {
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    const htmlFile = new ArtifactNode(
      false,
      { path: "file1.html", is_dir: false, file_size: "159" }
    );
    rootNode.setChildren([htmlFile.fileInfo]);

    const props = {...minimalProps, artifactNode: rootNode};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const htmlFileElement = wrapper.find('NodeHeader').at(0);
    htmlFileElement.simulate('click');
    expect(wrapper.find(ShowArtifactHtmlView)).toHaveLength(1);
  });

  test('should render geojson file in map artifact view', () => {
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    const geojsonFile = new ArtifactNode(
      false,
      { path: "file1.geojson", is_dir: false, file_size: "159" }
    );
    rootNode.setChildren([geojsonFile.fileInfo]);

    const props = {...minimalProps, artifactNode: rootNode};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const geojsonFileElement = wrapper.find('NodeHeader').at(0);
    geojsonFileElement.simulate('click');
    expect(wrapper.find(ShowArtifactMapView)).toHaveLength(1);
  });

  test('should render selected directory artifact', () => {
    const props = {...minimalProps};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const dir1Element = wrapper.find('NodeHeader').at(1);
    dir1Element.simulate('click');
    expect(wrapper.find('.artifact-info-path').html()).toContain('test_root/dir1');
    // Now that `dir1` has been selected, we expect the visible artifact tree
    // to contain 4 elements: file1, dir1, file2, and file3
    expect(wrapper.find('NodeHeader')).toHaveLength(4);
    // Directories should be displayed as zero bytes in size
    expect(wrapper.find('.artifact-info-size').html()).toContain('0B');
  });

  test('should render register model button for directory when no versions exist', () => {
    expect(Utils.isModelRegistryEnabled()).toEqual(true);
    const props = {...minimalProps};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const dir1Element = wrapper.find('NodeHeader').at(1);
    dir1Element.simulate('click');
    expect(wrapper.find('.artifact-info-path').html()).toContain('test_root/dir1');
    expect(wrapper.find('.register-model-btn-wrapper')).toHaveLength(1);
  });

  test('should not render register model button for directory when registry is disabled', () => {
    const enabledSpy = jest.spyOn(Utils, 'isModelRegistryEnabled').mockImplementation(() => false);
    expect(Utils.isModelRegistryEnabled()).toEqual(false);
    const props = {...minimalProps};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );
    const dir1Element = wrapper.find('NodeHeader').at(1);
    dir1Element.simulate('click');
    expect(wrapper.find('.artifact-info-path').html()).toContain('test_root/dir1');
    expect(wrapper.find('.register-model-btn-wrapper')).toHaveLength(0);

    enabledSpy.mockRestore();
  });

  test('should render model version link for directory when version is present', () => {
    expect(Utils.isModelRegistryEnabled()).toEqual(true);

    const modelVersionsBySource = {
      "test_root/dir1": [
        mockModelVersionDetailed(
          'Model A',
          1,
          Stages.PRODUCTION,
          ModelVersionStatus.READY,
          []
        ),
      ],
    };
    const props = {...minimalProps, modelVersionsBySource};
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactView {...props} />
        </BrowserRouter>
      </Provider>
    );

    const dir1Element = wrapper.find('NodeHeader').at(1);
    dir1Element.simulate('click');
    expect(wrapper.find('.artifact-info-path').html()).toContain('test_root/dir1');
    expect(wrapper.find('.model-version-info')).toHaveLength(1);
    expect(wrapper.find('.model-version-link')).toHaveLength(1);
    expect(wrapper.find('.model-version-link').props().title).toEqual("Model A, v1");
  });
});
