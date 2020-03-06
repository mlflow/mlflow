import { shallow } from 'enzyme';
import { ModelView } from './ModelView.js'
import React from "react";



describe('unconnected tests', () => {
  let wrapper;
  let minimumProps;

  beforeEach(() => {
    minimumProps = {
      model: {
        name: 'test_name',
        creation_timestamp: 123456789,
        last_updated_timestamp: 132456789
      },
      modelVersions: [{'current_stage': 'None'}],
      handleEditDescription: jest.fn(),
      handleDelete: jest.fn(),
      history: {},
    };
  });

  test('unconnected should render with minimal props without exploding', () => {
    wrapper = shallow(<ModelView {...minimumProps}/>);
    expect(wrapper.length).toBe(1);
  });

  test('compare button is disabled when no/1 runs selected, active when 2+ runs selected', () => {
    wrapper = shallow(<ModelView {...minimumProps}/>);
    console.log(wrapper.debug());
    // debugger;
    expect(wrapper.find('btn-primary')).toEqual(true);
    wrapper.setState({
      runsSelected: {'run_id_1': 'version_1'}
    })

  });
});