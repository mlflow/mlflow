/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import userEvent from '@testing-library/user-event-14';

import { SchemaTable } from './SchemaTable';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { renderWithIntl } from '../../common/utils/TestUtils.react18';

async function clickHeaderRow(container: HTMLElement, rowIndex: number): Promise<void> {
  // click to render inputs table
  const rows = container.querySelectorAll('tr.section-header-row');
  if (rows.length < rowIndex + 1) {
    throw new Error("Couldn't find the row to click");
  }
  await userEvent.click(rows[rowIndex]);
}

describe('SchemaTable', () => {
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
    const { container } = renderWithIntl(<SchemaTable {...minimalProps} />);
    expect(container).not.toBeNull();
  });

  test('should nested table not be rendered by default', () => {
    const { container } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(container.querySelector('.outer-table table')).not.toBeNull();
    expect(container.querySelector('.inner-table table')).toBeNull();
    expect(container.innerHTML).toContain('Inputs');
    expect(container.innerHTML).toContain('Outputs');
    expect(container.innerHTML).toContain('Name');
    expect(container.innerHTML).toContain('Type');
    expect(container.innerHTML).not.toContain('column1');
    expect(container.innerHTML).not.toContain('string');
    expect(container.innerHTML).not.toContain('score1');
    expect(container.innerHTML).not.toContain('long');
  });

  test('should inputs table render by click', async () => {
    const { container } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );

    expect(container.getElementsByTagName('table')).toHaveLength(1);
    // click to render inputs table
    await clickHeaderRow(container, 0);

    expect(container.getElementsByTagName('table')).toHaveLength(2);
    expect(container.querySelectorAll('.outer-table table')).toHaveLength(2);
    expect(container.querySelectorAll('.inner-table table')).toHaveLength(1);
    expect(container.innerHTML).toContain('Inputs');
    expect(container.innerHTML).toContain('Outputs');
    expect(container.innerHTML).toContain('Name');
    expect(container.innerHTML).toContain('Type');
    expect(container.innerHTML).toContain('column1');
    expect(container.innerHTML).toContain('string');
    expect(container.innerHTML).not.toContain('score1');
    expect(container.innerHTML).not.toContain('long');
  });

  test('Should display optional input field schema as expected', async () => {
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
    const wrapper = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render input schema table
    await clickHeaderRow(wrapper.container, 0);
    expect(wrapper.container.innerHTML).toContain('column1');
    // the optional input param should have (optional) after the name"
    const col2 = wrapper.getByText('column2');
    expect(col2.textContent).toEqual('column2 (optional)');
    expect(wrapper.container.innerHTML).toContain('string');
    expect(wrapper.container.innerHTML).toContain('float');
  });

  test('Should display required input field schema as expected', async () => {
    props = {
      schema: {
        // column1 is required but column2 is optional
        inputs: [{ name: 'column', type: 'string', required: true }],
        outputs: [{ name: 'score', type: 'long', required: true }],
      },
    };
    const wrapper = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render input schema table
    await clickHeaderRow(wrapper.container, 0);
    expect(wrapper.container.innerHTML).toContain('column');
    // the optional input param should have (optional) after the name"
    const col2 = wrapper.getByText('column');
    expect(col2.textContent).toEqual('column (required)');
  });

  test('Should display optional output field schema as expected', async () => {
    props = {
      schema: {
        inputs: [{ name: 'column1', type: 'string' }],
        // output contains an optional parameter
        outputs: [{ name: 'score1', type: 'long', optional: true }],
      },
    };
    const wrapper = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render output schema table
    await clickHeaderRow(wrapper.container, 1);
    // the optional output name should have (optional) after the name
    const score1 = wrapper.getByText('score1');
    expect(score1.textContent).toEqual('score1 (optional)');
  });

  test('should outputs table render by click', async () => {
    const { container } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render outputs table
    expect(container.getElementsByTagName('table')).toHaveLength(1);
    await clickHeaderRow(container, 1);

    expect(container.getElementsByTagName('table')).toHaveLength(2);
    expect(container.querySelectorAll('.outer-table table')).toHaveLength(2);
    expect(container.querySelectorAll('.inner-table table')).toHaveLength(1);
    expect(container.innerHTML).toContain('Inputs');
    expect(container.innerHTML).toContain('Outputs');
    expect(container.innerHTML).toContain('Name');
    expect(container.innerHTML).toContain('Type');
    expect(container.innerHTML).not.toContain('column1');
    expect(container.innerHTML).not.toContain('string');
    expect(container.innerHTML).toContain('score1');
    expect(container.innerHTML).toContain('long');
  });

  test('should inputs and outputs table render by click', async () => {
    const { container } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(container.getElementsByTagName('table')).toHaveLength(1);
    // click to render inputs and outputs table
    await clickHeaderRow(container, 0);
    expect(container.getElementsByTagName('table')).toHaveLength(2);
    await clickHeaderRow(container, 1);
    expect(container.getElementsByTagName('table')).toHaveLength(3);
    expect(container.querySelectorAll('.outer-table table')).toHaveLength(3);
    expect(container.querySelectorAll('.inner-table table')).toHaveLength(2);
    expect(container.innerHTML).toContain('Inputs');
    expect(container.innerHTML).toContain('Outputs');
    expect(container.innerHTML).toContain('Name');
    expect(container.innerHTML).toContain('Type');
    expect(container.innerHTML).toContain('column1');
    expect(container.innerHTML).toContain('string');
    expect(container.innerHTML).toContain('score1');
    expect(container.innerHTML).toContain('long');
  });

  test('Should display tensorSpec as expected', async () => {
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
    const { container } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(container.getElementsByTagName('table')).toHaveLength(1);
    // click to render inputs and outputs table
    await clickHeaderRow(container, 0);
    expect(container.getElementsByTagName('table')).toHaveLength(2);
    await clickHeaderRow(container, 1);
    expect(container.getElementsByTagName('table')).toHaveLength(3);
    expect(container.querySelectorAll('.outer-table table')).toHaveLength(3);
    expect(container.querySelectorAll('.inner-table table')).toHaveLength(2);
    expect(container.innerHTML).toContain('Inputs');
    expect(container.innerHTML).toContain('Outputs');
    expect(container.innerHTML).toContain('Name');
    expect(container.innerHTML).toContain('Type');
    expect(container.innerHTML).toContain('TensorInput');
    expect(container.innerHTML).toContain('Tensor (dtype: float64, shape: [-1,28,28])');
    expect(container.innerHTML).toContain('TensorOutput');
    expect(container.innerHTML).toContain('Tensor (dtype: float64, shape: [-1])');
  });

  test('should render object/array column types correctly', async () => {
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

    const { container } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );

    // click to render input schema table
    const row = container.querySelector('tr.section-header-row');
    if (row === null) {
      throw new Error("Couldn't find SchemaTable header row");
    }
    await userEvent.click(row);

    const signatures = container.getElementsByTagName('pre');
    expect(signatures).toHaveLength(2);
    expect(signatures[0].textContent).toEqual('{\n  prop1: string\n}');
    expect(signatures[1].textContent).toEqual('Array(binary)');
  });
});
