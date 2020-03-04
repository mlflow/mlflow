import { shallow } from 'enzyme';
import { CompareModelVersionsView } from './CompareModelVersionsView';
import ConnectedCompareModelVersionsView from './CompareModelVersionsView';
import React from "react";
import {getModelPageRoute, modelListPageRoute} from "../routes";
import {Link} from "react-router-dom";
import thunk from 'redux-thunk';
import promiseMiddleware from 'redux-promise-middleware';
import configureStore from 'redux-mock-store';
import { Provider } from 'react-redux';
import { mount } from 'enzyme';
import { BrowserRouter } from 'react-router-dom';
import Routes from "../../Routes";


describe('unconnected tests', () => {
  let wrapper;
  let minimumProps;

  beforeEach(() => {
    minimumProps = {
      modelName: 'test',
      runsToVersions: {},
      runUuids: ['runUuid_0', 'runUuid_1'],
      runInfos: [],
      metricLists: [],
      paramLists: [],
      runNames: [],
      runDisplayNames: [],
    };

  });

  test('unconnected should render with minimal props without exploding', () => {
    wrapper = shallow(<CompareModelVersionsView {...minimumProps}/>);
    expect(wrapper.length).toBe(1);
  });

  test('check that the breadcrumb renders correctly', () => {
    wrapper = shallow(<CompareModelVersionsView {...minimumProps}/>);
    const breadcrumbItemClass = 'truncate-text single-line breadcrumb-title';
    const modelName = 'test';
    expect(wrapper.containsAllMatchingElements([
      <Link to={modelListPageRoute} className={breadcrumbItemClass}>Registered Models</Link>,
      <Link to={getModelPageRoute(modelName)} className={breadcrumbItemClass}>{modelName}</Link>,
      <span className={breadcrumbItemClass}>{"Comparing 0 Versions"}</span>
    ])).toEqual(true)
  });

  test('check that the version row has rendered', () => {
    wrapper = shallow(<CompareModelVersionsView {...minimumProps}/>);
    expect(wrapper.contains(
      <th scope="row" className="data-value">Model Version:</th>
    )).toEqual(true)
  });

});

describe('connected tests', () => {
  let wrapper;
  let minimumProps;
  let minimalStore;
  let commonStore;
  const mockStore = configureStore([thunk, promiseMiddleware()]);


  beforeEach(() => {
    minimumProps = {
      modelName: 'test',
      runsToVersions: {}
    };
    minimalStore = mockStore({
      entities: {},
      apis: {},
    });
    commonStore = mockStore({
      entities: {
        runInfosByUuid:
          {"123":{"run_uuid":"123","experiment_id":"0","user_id":"test.user","status":"FINISHED","start_time":"0","end_time":"21","artifact_uri":"./mlruns","lifecycle_stage":"active"}},
        latestMetricsByRunUuid:
          {"123":{"test_metric":{"key":"test_metric","value":0.0,"timestamp":"321","step":"42"}}},
        paramsByRunUuid:
          {"123":{"test_param":{"key":"test_param","value":"0.0"}}},
        tagsByRunUuid:
          {"123":{"test_tag":{"key":"test_tag","value":"test.user"}}}
      },
      apis: {},
    })
  });


  test('connected should render with minimal props and minimal store without exploding', () => {
    wrapper = mount(
      <Provider store={minimalStore}>
        <BrowserRouter>
          <ConnectedCompareModelVersionsView {...minimumProps}/>
        </BrowserRouter>
      </Provider>
    );
    expect(wrapper.find(ConnectedCompareModelVersionsView).length).toBe(1);
    console.log(wrapper.debug())
  });

  test('connected should render with minimal props and common store correctly', () => {
    wrapper = mount(
      <Provider store={commonStore}>
        <BrowserRouter>
          <ConnectedCompareModelVersionsView {...minimumProps}/>
        </BrowserRouter>
      </Provider>
    );
    expect(wrapper.find(ConnectedCompareModelVersionsView).length).toBe(1);
    console.log(wrapper.debug());
    // TODO: find out why commonStore isn't passing its values correctly
  });
});