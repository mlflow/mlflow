import React from 'react';
import { shallow, mount } from 'enzyme';
import { RegisterModelForm, CREATE_NEW_MODEL_OPTION_VALUE } from './RegisterModelForm';
import { mockRegisteredModelDetailed } from '../test-utils';

describe('RegisterModelForm', () => {
  let wrapper;
  let instance;
  let minimalProps;

  beforeEach(() => {
    minimalProps = {};
  });

  test('should render with minimal props without exploding', () => {
    wrapper = shallow(<RegisterModelForm {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should list "Create New Model" and existing models in dropdown options', () => {
    const modelByName = {
      'Model A': mockRegisteredModelDetailed('Model A', []),
    };
    const props = {
      ...minimalProps,
      modelByName,
    };
    wrapper = shallow(<RegisterModelForm {...props} />).dive();
    expect(wrapper.find('.create-new-model-option').length).toBe(1);
    expect(wrapper.find('[value="Model A"]').length).toBe(1);
  });

  test('should show model name input when user choose "Create New Model"', () => {
    const modelByName = {
      'Model A': mockRegisteredModelDetailed('Model A', []),
    };
    const props = {
      ...minimalProps,
      modelByName,
    };
    wrapper = shallow(<RegisterModelForm {...props} />).dive();
    instance = wrapper.instance();
    instance.setState({ selectedModel: CREATE_NEW_MODEL_OPTION_VALUE });
    expect(wrapper.find('[label="Model Name"]').length).toBe(1);
  });

  test('should search registered model when user types model name', () => {
    const modelByName = {
      'Model A': mockRegisteredModelDetailed('Model A', []),
    };
    const onSearchRegisteredModels = jest.fn(() => Promise.resolve({}));
    const props = {
      ...minimalProps,
      modelByName,
      onSearchRegisteredModels,
    };
    wrapper = mount(<RegisterModelForm {...props} />);
    wrapper.find('input#selectedModel').simulate('change', { target: { value: 'Model B' } });
    expect(onSearchRegisteredModels.mock.calls.length).toBe(1);
  });
});
