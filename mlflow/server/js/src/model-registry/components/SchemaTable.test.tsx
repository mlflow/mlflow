/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { shallow, render } from 'enzyme';
import { SchemaTable } from './SchemaTable';
import { Table } from 'antd';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { mountWithIntl } from '../../common/utils/TestUtils';

describe('SchemaTable', () => {
  let wrapper;
  let minimalProps: any;
  let props: any;

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
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
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
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(wrapper.find(Table).length).toBe(1);
    // click to render inputs table
    wrapper.find('tr.section-header-row').at(0).simulate('click');
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

  test('Should display optional input field schema as expected', () => {
    props = {
      schema: {
        // column1 is required but column2 is optional
        inputs: [
          { name: 'column1', type: 'string' },
          { name: 'column2', type: 'float', optional: true },
        ],
        outputs: [{ name: 'score1', type: 'long' }],
      },
    };
    wrapper = mountWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render input schema table
    wrapper.find('tr.section-header-row').at(0).simulate('click');
    expect(wrapper.html()).toContain('column1');
    // the optional input param should have (optional) after the name"
    const col2 = wrapper.find('span').filterWhere((n: any) => n.text().includes('column2'));
    expect(render(col2.get(0)).text()).toContain('column2 (optional)');
    expect(wrapper.html()).toContain('string');
    expect(wrapper.html()).toContain('float');
  });

  test('Should display required input field schema as expected', () => {
    props = {
      schema: {
        // column1 is required but column2 is optional
        inputs: [{ name: 'column', type: 'string', required: true }],
        outputs: [{ name: 'score', type: 'long', required: true }],
      },
    };
    wrapper = mountWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render input schema table
    wrapper.find('tr.section-header-row').at(0).simulate('click');
    expect(wrapper.html()).toContain('column');
    // the optional input param should have (optional) after the name"
    const col2 = wrapper.find('span').filterWhere((n: any) => n.text().includes('column'));
    expect(render(col2.get(0)).text()).toContain('column (required)');
  });

  test('Should display optional output field schema as expected', () => {
    props = {
      schema: {
        inputs: [{ name: 'column1', type: 'string' }],
        // output contains an optional parameter
        outputs: [{ name: 'score1', type: 'long', optional: true }],
      },
    };
    wrapper = mountWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render output schema table
    wrapper.find('tr.section-header-row').at(1).simulate('click');
    // the optional output name should have (optional) after the name
    const score1 = wrapper.find('span').filterWhere((n: any) => n.text().includes('score1'));
    expect(render(score1).text()).toContain('score1 (optional)');
  });

  test('should outputs table render by click', () => {
    wrapper = mountWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render outputs table
    expect(wrapper.find(Table).length).toBe(1);
    wrapper.find('tr.section-header-row').at(1).simulate('click');
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
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(wrapper.find(Table).length).toBe(1);
    // click to render inputs and outputs table
    wrapper.find('tr.section-header-row').at(0).simulate('click');
    expect(wrapper.find(Table).length).toBe(2);
    wrapper.find('tr.section-header-row').at(1).simulate('click');
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
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(wrapper.find(Table).length).toBe(1);
    // click to render inputs and outputs table
    wrapper.find('tr.section-header-row').at(0).simulate('click');
    expect(wrapper.find(Table).length).toBe(2);
    wrapper.find('tr.section-header-row').at(1).simulate('click');
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

  test('should render object/array column types correctly', () => {
    props = {
      schema: {
        // column1 is required but column2 is optional
        inputs: [
          {
            name: 'objectCol',
            type: 'object',
            properties: {
              prop1: { type: 'string', required: true },
            },
            required: true,
          },
          {
            name: 'arrayCol',
            type: 'array',
            items: { type: 'binary', required: true },
            required: true,
          },
        ],
        outputs: [{ name: 'score1', type: 'long', required: true }],
      },
    };

    wrapper = mountWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render input schema table
    wrapper.find('tr.section-header-row').at(0).simulate('click');

    const signatures = wrapper.find('pre');
    expect(signatures).toHaveLength(2);
    expect(shallow(signatures.get(0)).text()).toEqual('{\n  prop1: string\n}');
    expect(shallow(signatures.get(1)).text()).toEqual('Array(binary)');
  });
});
