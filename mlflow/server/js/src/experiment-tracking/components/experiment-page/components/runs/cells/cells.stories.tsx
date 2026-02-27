import { ClientSideRowModelModule } from '@ag-grid-community/client-side-row-model';
import '@ag-grid-community/core/dist/styles/ag-grid.css';
import '@ag-grid-community/core/dist/styles/ag-theme-balham.css';
import { AgGridReact } from '@ag-grid-community/react';
import React from 'react';
import { IntlProvider } from 'react-intl';
import { MemoryRouter } from '../../../../../../common/utils/RoutingUtils';
import { EXPERIMENT_RUNS_MOCK_STORE } from '../../../fixtures/experiment-runs.fixtures';
import type {
  RunRowDateAndNestInfo,
  RunRowModelsInfo,
  RunRowType,
  RunRowVersionInfo,
} from '../../../utils/experimentPage.row-types';
import { ColumnHeaderCell } from './ColumnHeaderCell';
import { DateCellRenderer } from './DateCellRenderer';
import { ExperimentNameCellRenderer } from './ExperimentNameCellRenderer';
import { ModelsCellRenderer } from './ModelsCellRenderer';
import { SourceCellRenderer } from './SourceCellRenderer';
import { VersionCellRenderer } from './VersionCellRenderer';

export default {
  title: 'ExperimentView/Table Cells',
  argTypes: {},
};

const MOCK_MODEL = EXPERIMENT_RUNS_MOCK_STORE.entities.modelVersionsByRunUuid['experiment123456789_run4'][0];

const createAgTable = (
  component: React.ComponentType<React.PropsWithChildren<any>>,
  name: string,
  defs?: any[],
  rows?: any[],
) => (
  <div className="ag-theme-balham" style={{ height: 400 }}>
    <MemoryRouter initialEntries={['/']}>
      <IntlProvider locale="en">
        <AgGridReact
          components={{ [name]: component }}
          suppressMovableColumns
          columnDefs={
            defs || [
              {
                field: 'key',
                headerName: 'Key',
                headerComponentParams: {
                  canonicalSortKey: 'key',
                  orderByKey: 'key',
                  orderByAsc: true,
                  enableSorting: true,
                  onSortBy: () => {},
                },
              },
              { field: 'foo', headerName: 'Foo' },
            ]
          }
          rowData={
            rows || [
              { id: 'test-1', key: 'value', foo: 'bar' },
              { id: 'test-2', key: 'other value', foo: 'baz' },
            ]
          }
          modules={[ClientSideRowModelModule]}
          rowModelType="clientSide"
          domLayout="normal"
        />
      </IntlProvider>
    </MemoryRouter>
  </div>
);

export const ColumnHeader = () => {
  return createAgTable(ColumnHeaderCell, 'agColumnHeader');
};
export const DateRenderer = () => {
  const rowsWithDateAndNestInfo: { date: RunRowDateAndNestInfo }[] = [10000, 100000, 1000000].map((timeAgo) => ({
    date: {
      childrenIds: [],
      expanderOpen: false,
      experimentId: '12345',
      hasExpander: false,
      isParent: false,
      level: 0,
      referenceTime: new Date(),
      runStatus: 'FINISHED',
      runUuid: '123',
      startTime: Date.now() - timeAgo,
      belongsToGroup: false,
    },
  }));

  return createAgTable(
    DateCellRenderer,
    'dateCellRenderer',
    [
      {
        field: 'date',
        cellRenderer: 'dateCellRenderer',
      },
    ],
    rowsWithDateAndNestInfo,
  );
};
export const DateRendererWithExpander = () => {
  const rowsWithDateAndNestInfo: { date: Partial<RunRowDateAndNestInfo> }[] = [
    {
      date: {
        childrenIds: ['1001', '1002'],
        expanderOpen: true,
        experimentId: '12345',
        hasExpander: true,
        isParent: true,
        level: 0,
        referenceTime: new Date(),
        runStatus: 'FINISHED',
        runUuid: '1000',
        startTime: Date.now() - 10000,
      },
    },
    {
      date: {
        childrenIds: [],
        expanderOpen: false,
        experimentId: '12345',
        level: 1,
        runStatus: 'FINISHED',
        runUuid: '1001',
        startTime: Date.now() - 10000,
      },
    },
    {
      date: {
        childrenIds: [],
        expanderOpen: false,
        experimentId: '12345',
        level: 1,
        runStatus: 'FINISHED',
        runUuid: '1002',
        startTime: Date.now() - 10000,
      },
    },
  ];

  return createAgTable(
    DateCellRenderer,
    'dateCellRenderer',
    [
      {
        field: 'date',
        cellRenderer: 'dateCellRenderer',
        cellRendererParams: { onExpand: () => {} },
      },
    ],
    rowsWithDateAndNestInfo,
  );
};

export const ExperimentName = () => {
  return createAgTable(
    ExperimentNameCellRenderer,
    'experimentNameCellRenderer',
    [
      {
        field: 'experimentName',
        cellRenderer: 'experimentNameCellRenderer',
        cellRendererParams: {},
      },
    ],
    [
      {
        experimentId: 12345,
        experimentName: { name: 'An experiment name', basename: 'An experiment basename' },
      },
      {
        experimentId: 321,
        experimentName: { name: 'Other experiment name', basename: 'Other experiment basename' },
      },
    ],
  );
};

export const ModelsRenderer = () => {
  const loggedModels = [
    {
      artifactPath: 'model',
      flavors: ['sklearn'],
      utcTimeCreated: 1000,
    },
  ];
  const rowsWithModelInfo: { models: Partial<RunRowModelsInfo> }[] = [
    {
      models: {
        experimentId: '1234',
        registeredModels: [MOCK_MODEL],
        loggedModels: loggedModels,
      },
    },
    {
      models: {
        experimentId: '1234',
        registeredModels: [{ ...MOCK_MODEL, version: '2' }],
        loggedModels: loggedModels,
      },
    },
  ];

  return createAgTable(
    ModelsCellRenderer,
    'modelsCellRenderer',
    [
      {
        field: 'models',
        cellRenderer: 'modelsCellRenderer',
        cellRendererParams: {},
      },
    ],
    rowsWithModelInfo,
  );
};

export const SourceRenderer = () => {
  const rowsWithTagsInfo: Partial<RunRowType>[] = [
    {
      tags: {
        'mlflow.source.name': { key: 'mlflow.source.name', value: 'Notebook name' },
        'mlflow.databricks.notebookID': { key: 'mlflow.databricks.notebookID', value: '123456' },
        'mlflow.source.type': { key: 'mlflow.source.type', value: 'NOTEBOOK' },
      },
    },
    {
      tags: {
        'mlflow.source.name': {
          key: 'mlflow.source.name',
          value: 'https://github.com/xyz/path-to-repo',
        },
        'mlflow.source.type': { key: 'mlflow.source.type', value: 'LOCAL' },
      },
    },
    {
      tags: {
        'mlflow.source.name': {
          key: 'mlflow.source.name',
          value: 'https://github.com/xyz/path-to-repo',
        },
        'mlflow.source.type': { key: 'mlflow.source.type', value: 'PROJECT' },
      },
    },
    {
      tags: {
        'mlflow.source.name': { key: 'mlflow.source.name', value: '1234' },
        'mlflow.databricks.jobID': { key: 'mlflow.databricks.jobID', value: '1234' },
        'mlflow.databricks.jobRunID': { key: 'mlflow.databricks.jobRunID', value: '4321' },
        'mlflow.source.type': { key: 'mlflow.source.type', value: 'JOB' },
      },
    },
  ];

  return createAgTable(
    SourceCellRenderer,
    'sourceCellRenderer',
    [
      {
        field: 'tags',
        cellRenderer: 'sourceCellRenderer',
        cellRendererParams: {},
      },
    ],
    rowsWithTagsInfo,
  );
};

export const VersionRenderer = () => {
  const rowsWithVersionInfo: { version: Partial<RunRowVersionInfo> }[] = [
    {
      version: {
        name: 'https://github.com/xyz/path-to-repo',
        type: 'PROJECT',
        version: '654321',
      },
    },
    {
      version: {
        name: 'Just a version',
        type: 'OTHER',
        version: '123',
      },
    },
  ];

  return createAgTable(
    VersionCellRenderer,
    'versionCellRenderer',
    [
      {
        field: 'version',
        cellRenderer: 'versionCellRenderer',
        cellRendererParams: {},
      },
    ],
    rowsWithVersionInfo,
  );
};
