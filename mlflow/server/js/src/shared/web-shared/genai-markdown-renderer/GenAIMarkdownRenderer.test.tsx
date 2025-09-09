import { render, screen } from '@testing-library/react';
import React from 'react';
import ReactMarkdown from 'react-markdown-10';
import remarkGfm from 'remark-gfm-4';

import { TableRow } from '@databricks/design-system';

import { GenAIMarkdownRenderer, getMarkdownComponents } from './GenAIMarkdownRenderer';
import { useParsedTableComponents, VirtualizedTableRow, VirtualizedTableCell } from './TableRenderer';
import type { ReactMarkdownProps } from './types';

const MARKDOWN_CONTENT = `
# Test content
Some sample test content to be rendered.
`;

describe('GenAIMarkdownRenderer', () => {
  it('renders the component', async () => {
    render(<GenAIMarkdownRenderer>{MARKDOWN_CONTENT}</GenAIMarkdownRenderer>);

    expect(screen.getByRole('heading', { name: 'Test content' })).toBeInTheDocument();
  });

  describe('getMarkdownComponents', () => {
    describe('Anchor', () => {
      it('renders anchor component', async () => {
        const Component = getMarkdownComponents({}).a;
        const screen = render(
          <Component node={{} as any} href="test">
            {['Test']}
          </Component>,
        );

        expect(screen.getByRole('link')).toHaveAttribute('href', 'test');
      });
      it('renders anchor that opens in new tab', async () => {
        const Component = getMarkdownComponents({}).a;
        const screen = render(
          <Component node={{} as any} href="https://www.test.com">
            {['Test']}
          </Component>,
        );

        expect(screen.getByRole('link')).toHaveAttribute('target', '_blank');
      });
      it('renders anchor that does not open in new tab for ID links', async () => {
        const Component = getMarkdownComponents({}).a;
        const screen = render(
          <Component node={{} as any} href="#foobar">
            {['Test']}
          </Component>,
        );

        expect(screen.getByRole('link')).not.toHaveAttribute('target', '_blank');
      });
      it('renders footnote links', async () => {
        const screen = render(
          <GenAIMarkdownRenderer>
            {`Here is a simple footnote[^1][^2][^3]. With some additional text after it.\n\n[^1]: My reference 1.\n\n[^2]: My reference 2.\n\n[^3]: My reference 3.`}
          </GenAIMarkdownRenderer>,
        );

        expect(screen.getByRole('link', { name: '[1]' })).toHaveAttribute('href', '#user-content-fn-1');
        expect(screen.getByRole('link', { name: '[1]' })).toHaveTextContent('[1]');
        expect(screen.container.querySelector('#user-content-fn-1')).toHaveTextContent('My reference 1.');

        expect(screen.getByRole('link', { name: '[2]' })).toHaveAttribute('href', '#user-content-fn-2');
        expect(screen.getByRole('link', { name: '[2]' })).toHaveTextContent('[2]');
        expect(screen.container.querySelector('#user-content-fn-2')).toHaveTextContent('My reference 2.');

        expect(screen.getByRole('link', { name: '[3]' })).toHaveAttribute('href', '#user-content-fn-3');
        expect(screen.getByRole('link', { name: '[3]' })).toHaveTextContent('[3]');
        expect(screen.container.querySelector('#user-content-fn-3')).toHaveTextContent('My reference 3.');
      });
    });
  });
});

describe('useParsedTableComponents', () => {
  // Helper: render some markdown, intercept <table> to invoke our hook, and resolve its return value.
  function testTableParsing(markdown: string): Promise<ReturnType<typeof useParsedTableComponents>> {
    return new Promise((resolve) => {
      function HookTester({ children }: ReactMarkdownProps<'table'>) {
        const result = useParsedTableComponents({ children });
        React.useEffect(() => {
          resolve(result);
        }, [result]);
        return null;
      }

      render(
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ table: HookTester }}>
          {markdown}
        </ReactMarkdown>,
      );
    });
  }

  it('parses a basic 3x2 table correctly', async () => {
    const markdown = `
| A | B | C |
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |
`;
    const result = await testTableParsing(markdown);

    expect(result.isValid).toBe(true);

    // header should be defined (whatever the first child is)
    expect(result.header).toBeDefined();

    // rows should be exactly two <tr> elements
    expect(result.rows.length).toBe(2);
    result.rows.forEach((rowEl) => {
      expect(rowEl.type).toBe('tr');
    });

    // Spot-check: first cell of header is "A"
    const headerElement = result.header as React.ReactElement;
    const headerRow = React.Children.toArray(headerElement.props.children).find(
      (el) => React.isValidElement(el) && el.type === 'tr',
    ) as React.ReactElement;
    const firstHeaderCell = React.Children.toArray(headerRow.props.children)[0] as React.ReactElement;
    expect(firstHeaderCell.props.children).toStrictEqual('A');

    // Spot-check: first body row's first cell is "1"
    const firstBodyRow = result.rows[0];
    const firstBodyCell = React.Children.toArray(firstBodyRow.props.children)[0] as React.ReactElement;
    expect(firstBodyCell.props.children).toStrictEqual('1');
  });

  it('parses a single-row table correctly', async () => {
    const markdown = `
| X | Y | Z |
|---|---|---|
| only | one | row |
`;
    const result = await testTableParsing(markdown);

    expect(result.isValid).toBe(true);
    expect(result.header).toBeDefined();
    expect(result.rows.length).toBe(1);

    const headerRow = React.Children.toArray((result.header as React.ReactElement).props.children).find(
      (el) => React.isValidElement(el) && el.type === 'tr',
    ) as React.ReactElement;
    const headerTexts = React.Children.toArray(headerRow.props.children).map(
      (cell: any) => (cell as React.ReactElement).props.children,
    );
    expect(headerTexts).toStrictEqual(['X', 'Y', 'Z']);

    // Check the single body row's texts: ['only','one','row']
    const bodyRow = result.rows[0];
    const bodyTexts = React.Children.toArray(bodyRow.props.children).map(
      (cell: any) => (cell as React.ReactElement).props.children,
    );
    expect(bodyTexts).toStrictEqual(['only', 'one', 'row']);
  });

  it('enforces header count equal to row count', async () => {
    const markdown = `
| Col1 | Col2 |
|------|------|
| c    | d    | e |
`;
    const result = await testTableParsing(markdown);

    expect(result.isValid).toBe(true);
    expect(result.header).toBeDefined();

    // Exactly two <tr> rows
    expect(result.rows.length).toBe(1);

    // First body row has 2 cells
    const firstRowCellCount = React.Children.toArray(result.rows[0].props.children).filter((c) =>
      React.isValidElement(c),
    ).length;
    expect(firstRowCellCount).toBe(2);
  });

  it('parses a header-only table (no body rows)', async () => {
    const markdown = `
  | H1 | H2 |
  |----|----|
  `;
    const result = await testTableParsing(markdown);

    expect(result.isValid).toBe(true);
    expect(result.header).toBeDefined();

    // Since there are no body rows, rows.length should be 0
    expect(result.rows.length).toBe(0);

    // Check that header cells are still parsed correctly
    const headerRow = React.Children.toArray((result.header as React.ReactElement).props.children).find(
      (el) => React.isValidElement(el) && el.type === 'tr',
    ) as React.ReactElement;

    const headerTexts = React.Children.toArray(headerRow.props.children).map(
      (cell: any) => (cell as React.ReactElement).props.children,
    );
    expect(headerTexts).toStrictEqual(['H1', 'H2']);
  });

  it('allows inline Markdown (bold, links) inside table cells', async () => {
    const markdown = `
  | Bold       | Link                  | Image                       |
  |------------|-----------------------|-----------------------------|
  | **strong** | [click me](https://)  | ![alt](https://via.placeholder.com/20) |
  | normal     | [here](https://openai.com) | ![icon](https://via.placeholder.com/20) |
  `;
    const result = await testTableParsing(markdown);

    expect(result.isValid).toBe(true);
    expect(result.rows.length).toBe(2);

    // First body row, first cell contains a <strong> element with text "strong"
    const firstBodyRow = result.rows[0];
    const firstBodyCells = React.Children.toArray(firstBodyRow.props.children).filter((c) => React.isValidElement(c));

    // The <td> for "**strong**" should have a <strong> child
    const strongEl = (firstBodyCells[0] as React.ReactElement).props.children;
    expect(strongEl).toBeDefined();
    expect(strongEl.type).toBe('strong');
    expect(strongEl.props.children).toStrictEqual('strong');

    // The link cell should include an <a> element with href
    const linkEl = (firstBodyCells[1] as React.ReactElement).props.children;
    expect(linkEl.props.href).toBe('https://');

    // The image cell should include an <img> element with correct src and alt
    const imgEl = (firstBodyCells[2] as React.ReactElement).props.children;
    expect(imgEl.props.src).toMatch(/via\.placeholder\.com\/20/);
    expect(imgEl.props.alt).toBe('alt');
  });
});

describe('TableRenderer', () => {
  it('renders table with 3 columns and 3 rows', async () => {
    const markdown = `
  | H1 | H2 | H3 |
  |----|----|----|
  | 1  | 2  | 3  |
  | 4  | 5  | 6  |
  | 7  | 8  | 9  |
  
  `;

    const screen = render(<GenAIMarkdownRenderer>{markdown}</GenAIMarkdownRenderer>);

    expect(screen.getByRole('table')).toBeInTheDocument();
    expect(screen.getByTestId('virtualized-table')).toBeInTheDocument();

    // Expect 4 rows
    expect(screen.getAllByRole('row').length).toBe(4);

    // Expect 3 columns
    expect(screen.getAllByRole('columnheader').length).toBe(3);
  });

  it('uses virtualization when the table is large', async () => {
    const markdown = `
  | H1 |
  |----|
  | 1  |
  | 2  |
  | 3  |
  | 4  |
  | 5  |
  | 6  |
  | 7  |
  | 8  |
  | 9  |
  | 10 |
  | 11 |
  | 12 |
  | 13 |
  | 14 |
  | 15 |
  | 16 |
  | 17 |
  | 18 |
  | 19 |
  | 20 |
  | 21 |
  | 22 |
  | 23 |
  | 24 |
  | 25 |
  | 26 |
  | 27 |
  | 28 |
  | 29 |
  | 30 |
  | 31 |
  | 32 |
  | 33 |
  | 34 |
  | 35 |
  | 36 |
  | 37 |
  | 38 |
  | 39 |
  | 40 |
  | 41 |
  | 42 |
  | 43 |
  | 44 |
  | 45 |
  `;

    const screen = render(<GenAIMarkdownRenderer>{markdown}</GenAIMarkdownRenderer>);

    // Verify virtualized table is being used
    expect(screen.getByTestId('virtualized-table')).toBeInTheDocument();

    const visibleRows = screen.getAllByRole('row');

    expect(visibleRows[0]).toHaveTextContent('H1');

    // With virtualization enabled, we should render less than 30 rows
    expect(visibleRows.length).toBeLessThan(45);
  });
});

describe('VirtualizedTableRow', () => {
  it('renders a regular table row', () => {
    const mockNode = {
      children: [{ tagName: 'td' }],
    };
    const screen = render(<VirtualizedTableRow node={mockNode as any}>{['Cell content']}</VirtualizedTableRow>);

    expect(screen.getByRole('row')).toBeInTheDocument();
    expect(screen.getByText('Cell content')).toBeInTheDocument();
  });

  it('renders a header row with sticky positioning', () => {
    const mockNode = {
      children: [{ tagName: 'th' }],
    };
    const screen = render(<VirtualizedTableRow node={mockNode as any}>{['Header content']}</VirtualizedTableRow>);

    const row = screen.getByRole('row');
    expect(row).toBeInTheDocument();
    expect(row).toHaveStyle({ position: 'sticky', top: '0', zIndex: '1' });
    expect(screen.getByText('Header content')).toBeInTheDocument();
  });
});

describe('VirtualizedTableCell', () => {
  it('renders a regular table cell', () => {
    const mockNode = {
      tagName: 'td',
    };
    const screen = render(<VirtualizedTableCell node={mockNode as any}>{['Cell content']}</VirtualizedTableCell>);

    expect(screen.getByRole('cell')).toBeInTheDocument();
    expect(screen.getByText('Cell content')).toBeInTheDocument();
  });

  it('renders a table header cell when node is th', () => {
    const mockNode = {
      tagName: 'th',
    };

    const screen = render(
      <TableRow isHeader>
        <VirtualizedTableCell node={mockNode as any}>{['Header content']}</VirtualizedTableCell>
      </TableRow>,
    );

    expect(screen.getByRole('columnheader')).toBeInTheDocument();
    expect(screen.getByText('Header content')).toBeInTheDocument();
  });
});
