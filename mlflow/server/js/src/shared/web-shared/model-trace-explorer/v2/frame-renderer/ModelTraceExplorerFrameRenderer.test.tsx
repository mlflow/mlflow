import { jest, describe, it, expect } from '@jest/globals';
import { screen, act } from '@testing-library/react';
import { render } from '@databricks/web-shared/test-utils/render';
import React from 'react';

import { ModelTraceChildToParentFrameMessage } from '../../frame-renderer/types';

import { ModelTraceExplorerFrameRenderer, getTraceVersion } from './ModelTraceExplorerFrameRenderer';
import type { ModelTrace, ModelTraceInfo, NotebookModelTraceInfo } from '../ModelTrace.types';
import { MLFLOW_TRACE_SCHEMA_VERSION_KEY } from '../ModelTrace.types';
import { MOCK_TRACE } from '../../ModelTraceExplorer.test-utils';
import rendererVersions from '../../ml-model-trace-renderer/library-versions.json';

// backward compatibility tests for all possible trace versions
const TRACE_V1_RUNS_METADATA: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    request_metadata: [{ key: MLFLOW_TRACE_SCHEMA_VERSION_KEY, value: '1' }],
  } as ModelTraceInfo,
};

const TRACE_V1_RUNS_TAGS: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    tags: [{ key: MLFLOW_TRACE_SCHEMA_VERSION_KEY, value: '1' }],
  } as ModelTraceInfo,
};

const TRACE_V1_NOTEBOOK_METADATA: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    request_metadata: { [MLFLOW_TRACE_SCHEMA_VERSION_KEY]: '1' },
  } as NotebookModelTraceInfo,
};

const TRACE_V1_NOTEBOOK_TAGS: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    tags: { [MLFLOW_TRACE_SCHEMA_VERSION_KEY]: '1' },
  } as NotebookModelTraceInfo,
};

const TRACE_V2_RUNS_METADATA: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    request_metadata: [{ key: MLFLOW_TRACE_SCHEMA_VERSION_KEY, value: '2' }],
  } as ModelTraceInfo,
};

const TRACE_V2_RUNS_TAGS: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    tags: [{ key: MLFLOW_TRACE_SCHEMA_VERSION_KEY, value: '2' }],
  } as ModelTraceInfo,
};

const TRACE_V2_NOTEBOOK_METADATA: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    request_metadata: { MLFLOW_TRACE_SCHEMA_VERSION_KEY: '2' },
  } as NotebookModelTraceInfo,
};

const TRACE_V2_NOTEBOOK_TAGS: ModelTrace = {
  ...MOCK_TRACE,
  info: {
    ...MOCK_TRACE.info,
    tags: { MLFLOW_TRACE_SCHEMA_VERSION_KEY: '2' },
  } as NotebookModelTraceInfo,
};

const TRACE_V0: ModelTrace = MOCK_TRACE;

const ALL_TRACES: [string, ModelTrace, string, string][] = [
  // Versions with no corresponding renderer, should use fallback
  ['TRACE_V1_RUNS_METADATA', TRACE_V1_RUNS_METADATA, '1', rendererVersions[2].path],
  ['TRACE_V1_RUNS_TAGS', TRACE_V1_RUNS_TAGS, '1', rendererVersions[2].path],
  ['TRACE_V1_NOTEBOOK_METADATA', TRACE_V1_NOTEBOOK_METADATA, '1', rendererVersions[2].path],
  ['TRACE_V1_NOTEBOOK_TAGS', TRACE_V1_NOTEBOOK_TAGS, '1', rendererVersions[2].path],
  ['TRACE_V0', TRACE_V0, '2', rendererVersions[2].path],

  // Versions with corresponding renderer
  ['TRACE_V2_RUNS_METADATA', TRACE_V2_RUNS_METADATA, '2', rendererVersions[2].path],
  ['TRACE_V2_RUNS_TAGS', TRACE_V2_RUNS_TAGS, '2', rendererVersions[2].path],
  ['TRACE_V2_NOTEBOOK_METADATA', TRACE_V2_NOTEBOOK_METADATA, '2', rendererVersions[2].path],
  ['TRACE_V2_NOTEBOOK_TAGS', TRACE_V2_NOTEBOOK_TAGS, '2', rendererVersions[2].path],
];

describe('ModelTraceExplorerFrameRenderer', () => {
  it.each(ALL_TRACES)('parses the version number correctly (%s)', (_, trace: ModelTrace, expectedVersion: string) => {
    expect(getTraceVersion(trace)).toBe(expectedVersion);
  });

  it.each(ALL_TRACES)(
    'renders all types of traces with correct renderer without crashing (%s)',
    (_, trace, __, expectedRendererPath) => {
      render(<ModelTraceExplorerFrameRenderer modelTrace={trace} />);

      expect(screen.getByTitle('Model Trace Explorer')).toBeInTheDocument();
      expect(screen.getByTitle('Model Trace Explorer')).toHaveAttribute('src', expectedRendererPath);
    },
  );

  it('renders current renderer version', () => {
    render(<ModelTraceExplorerFrameRenderer modelTrace={TRACE_V0} useLatestVersion />);

    expect(screen.getByTitle('Model Trace Explorer')).toBeInTheDocument();
    expect(screen.getByTitle('Model Trace Explorer')).toHaveAttribute('src', rendererVersions.current.path);
  });

  it('logs error when iframe posts LogError message', () => {
    const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    const { container } = render(<ModelTraceExplorerFrameRenderer modelTrace={TRACE_V0} />);
    // eslint-disable-next-line testing-library/no-container, testing-library/no-node-access -- FEINF-5819: migrate render-container DOM queries to RTL queries; remove this disable when migrated
    const iframe = container.querySelector('iframe') as HTMLIFrameElement;

    // Create a mock contentWindow
    const mockContentWindow = {} as Window;
    Object.defineProperty(iframe, 'contentWindow', {
      value: mockContentWindow,
      writable: true,
    });

    // Simulate iframe posting an error message
    const testError = new Error('Test iframe error');
    const messageEvent = new MessageEvent('message', {
      data: {
        type: ModelTraceChildToParentFrameMessage.LogError,
        error: testError,
      },
      source: mockContentWindow,
    });

    act(() => {
      window.dispatchEvent(messageEvent);
    });

    expect(consoleErrorSpy).toHaveBeenCalledWith('MLflow trace renderer error:', testError);
    consoleErrorSpy.mockRestore();
  });
});
