import React from 'react';
import { shallow, mount } from 'enzyme';
import ArtifactPage from './ArtifactPage';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { Provider } from 'react-redux';
import { BrowserRouter } from 'react-router-dom';
import { pending } from '../../common/utils/ActionUtils';
import { SEARCH_MODEL_VERSIONS } from '../../model-registry/actions';
import {
  ModelVersionStatus, Stages, MODEL_VERSION_STATUS_POLL_INTERVAL as POLL_INTERVAL,
} from '../../model-registry/constants';
import Utils from '../../common/utils/Utils';
import {
  mockRegisteredModelDetailed,
} from '../../model-registry/test-utils';
import configureStore from 'redux-mock-store';
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';

describe('ArtifactPage', () => {
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
    };

    minimalStore = mockStore({
      apis: {},
      entities: {
        artifactsByRunUuid: {
          "fakeUuid": node,
        },
        artifactRootUriByRunUuid: {
          "fakeUuid": "8",
        },
        modelVersionsByModel: {
          'Model A': {
            '1': mockRegisteredModelDetailed(
              'Model A',
              1,
              Stages.PRODUCTION,
              ModelVersionStatus.READY,
              [],
            ),
          },
        },
      },
    });
  });

  const getTestArtifactNode = () => {
    const rootNode = new ArtifactNode(true, undefined);
    rootNode.isLoaded = true;
    const dir1 = new ArtifactNode(false, { path: "dir1", is_dir: true });
    const file1 = new ArtifactNode(false, { path: "file1", is_dir: false, file_size: "159" });
    rootNode.children = { dir1, file1 };
    return rootNode;
  };

  test('should render with minimal props without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactPage {...minimalProps} />
        </BrowserRouter>
      </Provider>
    );
    expect(wrapper.length).toBe(1);
  });

  test('should render artifact view', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ArtifactPage {...minimalProps} />
        </BrowserRouter>
      </Provider>
    );
    expect(wrapper.find('.artifact-view').length).toBe(1);
  });

  test('should poll for model versions if registry is enabled and active node is directory', () => {
    jest.useFakeTimers();
    expect(Utils.isModelRegistryEnabled()).toEqual(true);

    const props = {...minimalProps, store: minimalStore};
    wrapper = shallow(
      <ArtifactPage {...props} />
    ).dive();
    wrapper.instance().handleActiveNodeChange(true);
    jest.runTimersToTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(3);
  });

  test('should not poll for model versions if registry is disabled', () => {
    jest.useFakeTimers();
    const enabledSpy = jest.spyOn(Utils, 'isModelRegistryEnabled').mockImplementation(() => false);
    expect(Utils.isModelRegistryEnabled()).toEqual(false);

    const props = {...minimalProps, store: minimalStore};
    wrapper = shallow(
      <ArtifactPage {...props} />
    ).dive();
    wrapper.instance().handleActiveNodeChange(true);
    jest.runTimersToTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(0);

    enabledSpy.mockRestore();
  });

  test('should not poll for model versions if active node is not directory', () => {
    jest.useFakeTimers();

    const props = {...minimalProps, store: minimalStore};
    wrapper = shallow(
      <ArtifactPage {...props} />
    ).dive();
    expect(wrapper.instance().state.activeNodeIsDirectory).toEqual(false);

    jest.runTimersToTime(POLL_INTERVAL * 3);
    const expectedActions = minimalStore.getActions().filter((action) => {
      return action.type === pending(SEARCH_MODEL_VERSIONS);
    });
    expect(expectedActions).toHaveLength(0);
  });
});
