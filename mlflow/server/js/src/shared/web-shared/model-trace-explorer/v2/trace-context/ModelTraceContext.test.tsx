import { describe, test, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import userEvent from '@testing-library/user-event';
import React from 'react';

import { ModelTraceContextProvider, useModelTraceContext } from './ModelTraceContext';
import { MLFLOW_TRACE_SCHEMA_VERSION_KEY, type ModelTraceInferenceTableData } from '../ModelTrace.types';
import { mockSpans, MOCK_TRACE } from '../../ModelTraceExplorer.test-utils';

const MOCK_INFERENCE_DATA: ModelTraceInferenceTableData = {
  app_version_id: 'test',
  start_timestamp: '2024-04-17T07:24:04.050483',
  end_timestamp: '2024-04-17T07:24:04.050483',
  is_truncated: false,
  [MLFLOW_TRACE_SCHEMA_VERSION_KEY]: 2,
  spans: mockSpans.map((span) => ({
    ...span,
    attributes: JSON.stringify(span.attributes),
  })),
};

const NULL_TRACE_DATA = {
  ...MOCK_INFERENCE_DATA,
  // should be caught in null chaining
  spans: undefined,
};

const ERROR_TRACE_DATA = {
  ...MOCK_INFERENCE_DATA,
  // not stringified, which would
  // cause an error in JSON.parse
  spans: mockSpans,
};

const FULL_TRACE = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    tags: {
      [MLFLOW_TRACE_SCHEMA_VERSION_KEY]: '2',
    },
  },
};

const TestComponent = ({ data }: { data: any }) => {
  const { onSelectRow } = useModelTraceContext();

  return (
    <div>
      <h1>Test Child</h1>
      <button data-testid="button" onClick={() => onSelectRow(data)} />
    </div>
  );
};

describe('ModelTraceContext', () => {
  test('should select row and display the trace explorer component', async () => {
    render(
      <ModelTraceContextProvider>
        <TestComponent data={MOCK_INFERENCE_DATA} />
      </ModelTraceContextProvider>,
    );

    expect(screen.getByText('Test Child')).toBeInTheDocument();
    expect(screen.queryByText('document-qa-chain')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('button'));

    // original component should still be there
    expect(screen.getByText('Test Child')).toBeInTheDocument();
    // trace explorer should now be visible
    expect(await screen.findByTitle('Model Trace Explorer')).toBeInTheDocument();
  });

  test.each([
    { data: NULL_TRACE_DATA, label: 'null' },
    { data: ERROR_TRACE_DATA, label: 'error' },
    { data: { data: 'test', info: test }, label: 'not a trace but has data and info' },
  ])('should not blow up on non-trace data ($label)', async ({ data }) => {
    render(
      <ModelTraceContextProvider>
        <TestComponent data={data} />
      </ModelTraceContextProvider>,
    );

    expect(screen.getByText('Test Child')).toBeInTheDocument();
    expect(screen.queryByTitle('Model Trace Explorer')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('button'));

    // things should remain unchanged when clicking
    // on a row that has unexpected data
    expect(screen.getByText('Test Child')).toBeInTheDocument();
    expect(screen.queryByTitle('Model Trace Explorer')).not.toBeInTheDocument();
  });

  test('should display table schema containing full trace', async () => {
    render(
      <ModelTraceContextProvider>
        <TestComponent data={FULL_TRACE} />
      </ModelTraceContextProvider>,
    );

    expect(screen.getByText('Test Child')).toBeInTheDocument();
    expect(screen.queryByText('document-qa-chain')).not.toBeInTheDocument();

    await userEvent.click(screen.getByTestId('button'));

    // original component should still be there
    expect(screen.getByText('Test Child')).toBeInTheDocument();
    // trace explorer should now be visible
    expect(await screen.findByTitle('Model Trace Explorer')).toBeInTheDocument();
  });
});
