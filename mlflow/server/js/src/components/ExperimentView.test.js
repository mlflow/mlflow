import React from 'react';
import { shallow } from 'enzyme';
import { ExperimentView } from './ExperimentView';
import KeyFilter from "../utils/KeyFilter";
import {LIFECYCLE_FILTER} from "./ExperimentPage";
import Fixtures from '../test-utils/Fixtures';

test('Entering search filter input updates component state', () => {
  const wrapper = shallow(<ExperimentView
    onSearch={() => {}}
    runInfos={[]}
    experiment={Fixtures.experiments[0]}
    history={[]}
    paramKeyList={[]}
    metricKeyList={[]}
    paramsList={[]}
    metricsList={[]}
    tagsList={[]}
    paramKeyFilter={new KeyFilter("")}
    metricKeyFilter={new KeyFilter("")}
    lifecycleFilter={LIFECYCLE_FILTER.ACTIVE}
    searchInput={""}
  />);
  wrapper.find('.ExperimentView-paramKeyFilter input').first().simulate('change', {target: { value: 'param name'}});
  expect(wrapper.state('paramKeyFilterInput')).toEqual('param name');
});
