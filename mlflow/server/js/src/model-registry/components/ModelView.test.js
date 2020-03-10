import { shallow, mount } from 'enzyme';
import { ModelView } from './ModelView';
import React from "react";
import {BrowserRouter} from "react-router-dom";
import {getCompareModelVersionsPageRoute} from "../routes";


describe('unconnected tests', () => {
  let wrapper;
  let minimumProps;
  let historyMock;

  beforeEach(() => {
    historyMock = jest.fn();
    minimumProps = {
      model: {
        name: 'test_name',
        creation_timestamp: 123456789,
        last_updated_timestamp: 132456789,
      },
      modelVersions: [{'current_stage': 'None'}],
      handleEditDescription: jest.fn(),
      handleDelete: jest.fn(),
      history: { push: historyMock },
    };
  });

  test('unconnected should render with minimal props without exploding', () => {
    wrapper = shallow(<ModelView {...minimumProps}/>);
    expect(wrapper.length).toBe(1);
  });

  test('compare button is disabled when no/1 run selected, active when 2+ runs selected', () => {
    wrapper = mount(
      <BrowserRouter>
        <ModelView {...minimumProps}/>
      </BrowserRouter>
    );

    expect(wrapper.find('.btn').length).toBe(1);
    expect(wrapper.find('.btn').props().disabled).toEqual(true);

    wrapper.find(ModelView).instance().setState({
      runsSelected: {'run_id_1': 'version_1'},
    });
    wrapper.update();
    expect(wrapper.find('.btn').props().disabled).toEqual(true);

    const twoRunsSelected = {'run_id_1': 'version_1', 'run_id_2': 'version_2'};
    wrapper.find(ModelView).instance().setState({
      runsSelected: twoRunsSelected,
    });
    wrapper.update();
    expect(wrapper.find('.btn').props().disabled).toEqual(false);

    wrapper.find('.btn').simulate('click');
    expect(historyMock).toHaveBeenCalledWith(
      getCompareModelVersionsPageRoute(minimumProps['model']['name'], twoRunsSelected));
  });
});
