/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import userEvent from '@testing-library/user-event';

import { SchemaTable } from './SchemaTable';
import { MemoryRouter } from '../../common/utils/RoutingUtils';
import { getByPlaceholderText, renderWithIntl, within } from '../../common/utils/TestUtils.react18';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(60000); // Higher timeout due to testing heavier table

async function clickHeaderRow(container: HTMLElement, name: string | RegExp): Promise<void> {
  const row = within(container).getByRole('row', { name });
  if (!row) {
    throw new Error("Couldn't find the row to click");
  }
  await userEvent.click(row);
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
    const { container, getByRole } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );

    expect(getByRole('table')).toBeInTheDocument();
    // click to render inputs table
    await clickHeaderRow(container, /Inputs/);

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
    await clickHeaderRow(wrapper.container, /Inputs/);
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
    await clickHeaderRow(wrapper.container, /Inputs/);
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
    await clickHeaderRow(wrapper.container, /Outputs/);
    // the optional output name should have (optional) after the name
    const score1 = wrapper.getByText('score1');
    expect(score1.textContent).toEqual('score1 (optional)');
  });

  test('should outputs table render by click', async () => {
    const { container, getByRole } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    // click to render outputs table
    expect(getByRole('table')).toBeInTheDocument();
    await clickHeaderRow(container, /Outputs/);

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
    const { container, getByRole } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(getByRole('table')).toBeInTheDocument();
    // click to render inputs and outputs table
    await clickHeaderRow(container, /Inputs/);
    await clickHeaderRow(container, /Outputs/);
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
    const { container, getByRole } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );
    expect(getByRole('table')).toBeInTheDocument();
    await clickHeaderRow(container, /Inputs/);
    await clickHeaderRow(container, /Outputs/);
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

    const { container, getByRole } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable {...props} />
      </MemoryRouter>,
    );

    expect(getByRole('table')).toBeInTheDocument();
    await clickHeaderRow(container, /Inputs/);

    const signatures = container.getElementsByTagName('pre');
    expect(signatures).toHaveLength(2);
    expect(signatures[0].textContent).toEqual('{\n  prop1: string\n}');
    expect(signatures[1].textContent).toEqual('Array(binary)');
  });

  test('should handle large schemas (>100 columns)', async () => {
    const { container, getByRole, getByText, getByPlaceholderText } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable
          schema={{
            ...props.schema,
            // Render schema with 500 input columns
            inputs: new Array(500)
              .fill({ name: 'column1', type: 'string' })
              .map((item, index) => ({ ...item, name: `column${index}` })),
          }}
        />
      </MemoryRouter>,
    );
    expect(getByRole('table')).toBeInTheDocument();
    await clickHeaderRow(container, /Inputs/);

    expect(
      getByText(
        'Schema is too large to display all rows. Please search for a column name to filter the results. Currently showing 100 results from 500 total rows.',
      ),
    ).toBeInTheDocument();

    await userEvent.click(getByPlaceholderText('Search for a field'));
    await userEvent.paste('column499');

    expect(
      getByText(
        'Schema is too large to display all rows. Please search for a column name to filter the results. Currently showing 1 results from 500 total rows.',
      ),
    ).toBeInTheDocument();
  });

  test('should handle non-string column names', async () => {
    // Create a large schema with > 100 fields to trigger the search functionality
    type Column =
      | { name: string; type: 'string' }
      | { name: number; type: 'integer' | 'float' }
      | { name: boolean; type: 'boolean' };

    const manyInputs: Column[] = Array.from(
      { length: 101 },
      (_, i): Column => ({
        name: `regular_column_${i}`,
        type: 'string',
      }),
    );

    // Add non-string column names
    manyInputs.push(
      { name: 123, type: 'integer' },
      { name: true, type: 'boolean' },
      { name: 0, type: 'integer' },
      { name: 3.14, type: 'float' },
    );

    const { container, getByRole, getByPlaceholderText } = renderWithIntl(
      <MemoryRouter>
        <SchemaTable
          schema={{
            ...props.schema,
            inputs: manyInputs,
          }}
        />
      </MemoryRouter>,
    );
    expect(getByRole('table')).toBeInTheDocument();
    await clickHeaderRow(container, /Inputs/);

    // Verify that search field appears (since we have > 100 columns)
    const searchField = getByPlaceholderText('Search for a field');
    expect(searchField).toBeInTheDocument();

    // Search for the numeric column name
    await userEvent.click(searchField);
    await userEvent.paste('123');

    // The numeric column should be found and displayed
    expect(container.innerHTML).toContain('123');
    expect(container.innerHTML).toContain('integer');

    // Clear search and try a boolean column name
    await userEvent.clear(searchField);
    await userEvent.paste('true');

    // The boolean column should be found and displayed
    expect(container.innerHTML).toContain('true');
    expect(container.innerHTML).toContain('boolean');

    // Clear search and try a float column name
    await userEvent.clear(searchField);
    await userEvent.paste('3.14');

    // The float column should be found and displayed
    expect(container.innerHTML).toContain('3.14');
    expect(container.innerHTML).toContain('float');
  });
});
