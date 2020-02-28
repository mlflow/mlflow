import { shallow } from 'enzyme';
import CompareModelVersionsView from './CompareModelVersionsView';
import { Empty } from 'antd';
import {ParallelCoordinatesPlotPanel} from "../../components/ParallelCoordinatesPlotPanel";
import ParallelCoordinatesPlotView from "../../components/ParallelCoordinatesPlotView";
import React from "react";


describe('unit tests', () => {
    let wrapper;
    let instance;
    let mininumProps;

    beforeEach(() => {
        mininumProps = {
            runUuids: ['runUuid_0', 'runUuid_1'],
            sharedParamKeys: ['param_0', 'param_1'],
            sharedMetricKeys: ['metric_0', 'metric_1'],
        };
        wrapper = shallow(<CompareModelVersionsView {mininumProps}/>)
    });

    test('should render with minimal props without exploding', () => {
        wrapper = shallow(<CompareModelVersionsView {...mininumProps}/>);
        expect(wrapper.length).toBe(1);
    });

    // test('should render empty component when no dimension is selected', () => {
    //     wrapper = shallow(<ParallelCoordinatesPlotPanel {...mininumProps}/>);
    //     instance = wrapper.instance();
    //     expect(wrapper.find(ParallelCoordinatesPlotView)).toHaveLength(1);
    //     expect(wrapper.find(Empty)).toHaveLength(0);
    //     instance.setState({
    //         selectedParamKeys: [],
    //         selectedMetricKeys: [],
    //     });
    //     expect(wrapper.find(ParallelCoordinatesPlotView)).toHaveLength(0);
    //     expect(wrapper.find(Empty)).toHaveLength(1);
    // });
});