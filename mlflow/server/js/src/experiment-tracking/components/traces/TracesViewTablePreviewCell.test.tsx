import { describe, test, jest, expect } from '@jest/globals';
import { flexRender, getCoreRowModel, useReactTable } from '@tanstack/react-table';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { TracesViewTableRequestPreviewCell, TracesViewTableResponsePreviewCell } from './TracesViewTablePreviewCell';
import { Table, TableCell, TableRow } from '@databricks/design-system';
import userEvent from '@testing-library/user-event';
import { MlflowService } from '../../sdk/MlflowService';

const shortValue = '{"test":"short"}';
const longValue = `{"model_input":[{"query":"What is featured in the last version of MLflow?"}],"system_prompt":"\\nYou are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrastructure administration question related to Databricks. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.Use the following pieces of context to answer the question at the end:\\n","params":{"model_name":"databricks-dbrx-instruct","temperature":0.1,"max_tokens":1000}}`;
const longValueTruncated = `{"model_input":[{"query":"What is featured in the last version of MLflow?"}],"system_prompt":"\\nYou are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrast...`;

const formattedLongValue = JSON.stringify(JSON.parse(longValue), null, 2);

describe('ExperimentViewTracesTablePreviewCell', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  const renderTable = ({
    value,
    preview,
  }: {
    value: string;
    preview: 'request' | 'response';
  }) => {
    const Component = () => {
      const table = useReactTable({
        columns: [
          {
            // @ts-expect-error [FEINF-4084] Type 'ColumnDefTemplate<CellContext<ModelTraceInfoWithRunName, unknown>>' is not assignable to type 'ColumnDefTemplate...
            cell: preview === 'request' ? TracesViewTableRequestPreviewCell : TracesViewTableResponsePreviewCell,
            id: 'test',
          },
        ],
        data: [
          {
            request_metadata: [{ key: preview === 'request' ? 'mlflow.traceInputs' : 'mlflow.traceOutputs', value }],
            request_id: 'test_request_id',
          },
        ],
        getCoreRowModel: getCoreRowModel(),
      });

      return (
        <Table>
          {table.getRowModel().rows.map((row) => (
            <TableRow key={row.id}>
              {row.getAllCells().map((cell) => (
                <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell>
              ))}
            </TableRow>
          ))}
        </Table>
      );
    };

    renderWithDesignSystem(<Component />);
  };

  test('it should expand short values and request more data', async () => {
    jest
      .spyOn(MlflowService, 'getExperimentTraceData')
      .mockImplementation(() => Promise.resolve({ response: longValue }));

    renderTable({ value: longValueTruncated, preview: 'response' });
    expect(screen.queryByRole('button')).toBeInTheDocument();

    expect(screen.getByText(longValueTruncated)).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button'));

    expect(MlflowService.getExperimentTraceData).toHaveBeenCalledWith('test_request_id');

    expect(document.body.textContent).toContain(formattedLongValue);

    await userEvent.click(screen.getByRole('button'));

    expect(document.body.textContent).not.toContain(formattedLongValue);
  });

  test('it should unescape non-ascii characters', async () => {
    jest
      .spyOn(MlflowService, 'getExperimentTraceData')
      .mockImplementation(() => Promise.resolve({ response: longValue }));

    const escapedJson = '{"model_input":"\\uD83D\\uDE42"}';
    const unescapedJson = '{"model_input":"🙂"}';
    renderTable({ value: escapedJson, preview: 'response' });
    expect(screen.getByText(unescapedJson, { collapseWhitespace: false })).toBeInTheDocument();
  });

  test('it should render OTEL gen_ai messages with role labels when expanded', async () => {
    const otelMessages = JSON.stringify([
      { role: 'system', parts: [{ type: 'text', content: 'System instruction' }] },
      { role: 'user', parts: [{ type: 'text', content: 'User prompt' }] },
    ]);

    renderTable({ value: otelMessages, preview: 'response' });

    await userEvent.click(screen.getByRole('button'));

    expect(screen.getByText('System')).toBeInTheDocument();
    expect(screen.getByText('User')).toBeInTheDocument();
    expect(screen.getByText('System instruction')).toBeInTheDocument();
    expect(screen.getByText('User prompt')).toBeInTheDocument();
  });

  test('it should prefer gen_ai.input.messages for request', async () => {
    const inputOtelMessages = [
      { role: 'system', parts: [{ type: 'text', content: 'S' }] },
      { role: 'user', parts: [{ type: 'text', content: 'U' }] },
    ];
    const outputOtelMessages = [{ role: 'assistant', parts: [{ type: 'text', content: 'A' }] }];

    const combined = JSON.stringify({
      'gen_ai.input.messages': inputOtelMessages,
      'gen_ai.output.messages': outputOtelMessages,
    });

    renderTable({ value: combined, preview: 'request' });
    await userEvent.click(screen.getByRole('button'));

    expect(screen.getByText('S')).toBeInTheDocument();
    expect(screen.getByText('U')).toBeInTheDocument();
    expect(screen.queryByText('A')).not.toBeInTheDocument();
  });

  test('it should prefer gen_ai.output.messages for response', async () => {
    const inputOtelMessages = [
      { role: 'system', parts: [{ type: 'text', content: 'S' }] },
      { role: 'user', parts: [{ type: 'text', content: 'U' }] },
    ];
    const outputOtelMessages = [{ role: 'assistant', parts: [{ type: 'text', content: 'A' }] }];

    const combined = JSON.stringify({
      'gen_ai.input.messages': inputOtelMessages,
      'gen_ai.output.messages': outputOtelMessages,
    });

    renderTable({ value: combined, preview: 'response' });
    await userEvent.click(screen.getByRole('button'));

    expect(screen.getByText('A')).toBeInTheDocument();
    expect(screen.queryByText('S')).not.toBeInTheDocument();
    expect(screen.queryByText('U')).not.toBeInTheDocument();
  });
});
