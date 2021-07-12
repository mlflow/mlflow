import React from 'react';
import { SchemaTable } from './SchemaTable';
import { Table } from 'antd';
import { BrowserRouter } from 'react-router-dom';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('SchemaTable', () => {
  let wrapper;
  let minimalProps;
  let props;

  beforeEach(() => {
    minimalProps = {
      schema: {
        inputs: [],
        outputs: [],
      },
    };
    props = {
      schema: {
        inputs: [
          { name: 'column1', type: 'string' },
          { name: 'column2', type: 'string' },
        ],
        outputs: [
          { name: 'score1', type: 'long' },
          { name: 'score2', type: 'long' },
        ],
      },
    };
  });

  test('should render with minimal props without exploding', () => {
    wrapper = mountWithIntl(<SchemaTable {...minimalProps} />);
    expect(wrapper.length).toBe(1);
  });

  test('should nested table not be rendered by default', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <SchemaTable {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find(Table).length).toBe(1);
    expect(wrapper.find('.outer-table').find(Table).length).toBe(1);
    expect(wrapper.find('.inner-table').find(Table).length).toBe(0);
    expect(wrapper.html()).toContain('Inputs');
    expect(wrapper.html()).toContain('Outputs');
    expect(wrapper.html()).toContain('Name');
    expect(wrapper.html()).toContain('Type');
    expect(wrapper.html()).not.toContain('column1');
    expect(wrapper.html()).not.toContain('string');
    expect(wrapper.html()).not.toContain('score1');
    expect(wrapper.html()).not.toContain('long');
  });

  test('should inputs table render by click', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <SchemaTable {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find(Table).length).toBe(1);
    // click to render inputs table
    wrapper
      .find('tr.section-header-row')
      .at(0)
      .simulate('click');
    expect(wrapper.find(Table).length).toBe(2);
    expect(wrapper.find('.outer-table').find(Table).length).toBe(2);
    expect(wrapper.find('.inner-table').find(Table).length).toBe(1);
    expect(wrapper.html()).toContain('Inputs');
    expect(wrapper.html()).toContain('Outputs');
    expect(wrapper.html()).toContain('Name');
    expect(wrapper.html()).toContain('Type');
    expect(wrapper.html()).toContain('column1');
    expect(wrapper.html()).toContain('string');
    expect(wrapper.html()).not.toContain('score1');
    expect(wrapper.html()).not.toContain('long');
  });

  test('should outputs table render by click', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <SchemaTable {...props} />
      </BrowserRouter>,
    );
    // click to render outputs table
    expect(wrapper.find(Table).length).toBe(1);
    wrapper
      .find('tr.section-header-row')
      .at(1)
      .simulate('click');
    expect(wrapper.find(Table).length).toBe(2);
    expect(wrapper.find('.outer-table').find(Table).length).toBe(2);
    expect(wrapper.find('.inner-table').find(Table).length).toBe(1);
    expect(wrapper.html()).toContain('Inputs');
    expect(wrapper.html()).toContain('Outputs');
    expect(wrapper.html()).toContain('Name');
    expect(wrapper.html()).toContain('Type');
    expect(wrapper.html()).not.toContain('column1');
    expect(wrapper.html()).not.toContain('string');
    expect(wrapper.html()).toContain('score1');
    expect(wrapper.html()).toContain('long');
  });

  test('should inputs and outputs table render by click', () => {
    wrapper = mountWithIntl(
      <BrowserRouter>
        <SchemaTable {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find(Table).length).toBe(1);
    // click to render inputs and outputs table
    wrapper
      .find('tr.section-header-row')
      .at(0)
      .simulate('click');
    expect(wrapper.find(Table).length).toBe(2);
    wrapper
      .find('tr.section-header-row')
      .at(1)
      .simulate('click');
    expect(wrapper.find(Table).length).toBe(3);
    expect(wrapper.find('.outer-table').find(Table).length).toBe(3);
    expect(wrapper.find('.inner-table').find(Table).length).toBe(2);
    expect(wrapper.html()).toContain('Inputs');
    expect(wrapper.html()).toContain('Outputs');
    expect(wrapper.html()).toContain('Name');
    expect(wrapper.html()).toContain('Type');
    expect(wrapper.html()).toContain('column1');
    expect(wrapper.html()).toContain('string');
    expect(wrapper.html()).toContain('score1');
    expect(wrapper.html()).toContain('long');
  });

  test('Should display tensorSpec as expected', () => {
    props = {
      schema: {
        inputs: [
          {
            name: 'TensorInput',
            type: 'tensor',
            'tensor-spec': { dtype: 'float64', shape: [-1, 28, 28] },
          },
        ],
        outputs: [
          {
            name: 'TensorOutput',
            type: 'tensor',
            'tensor-spec': { dtype: 'float64', shape: [-1] },
          },
        ],
      },
    };
    wrapper = mountWithIntl(
      <BrowserRouter>
        <SchemaTable {...props} />
      </BrowserRouter>,
    );
    expect(wrapper.find(Table).length).toBe(1);
    // click to render inputs and outputs table
    wrapper
      .find('tr.section-header-row')
      .at(0)
      .simulate('click');
    expect(wrapper.find(Table).length).toBe(2);
    wrapper
      .find('tr.section-header-row')
      .at(1)
      .simulate('click');
    expect(wrapper.find(Table).length).toBe(3);
    expect(wrapper.find('.outer-table').find(Table).length).toBe(3);
    expect(wrapper.find('.inner-table').find(Table).length).toBe(2);
    expect(wrapper.html()).toContain('Inputs');
    expect(wrapper.html()).toContain('Outputs');
    expect(wrapper.html()).toContain('Name');
    expect(wrapper.html()).toContain('Type');
    expect(wrapper.html()).toContain('TensorInput');
    expect(wrapper.html()).toContain('Tensor (dtype: float64, shape: [-1,28,28])');
    expect(wrapper.html()).toContain('TensorOutput');
    expect(wrapper.html()).toContain('Tensor (dtype: float64, shape: [-1])');
  });
});
