import { ATTRIBUTE_COLUMN_SORT_LABEL } from '../../../constants';
import { useRunSortOptions } from './useRunSortOptions';

jest.mock('react', () => ({
  ...jest.requireActual('react'),
  // Mock useMemo() so we can use it outside React component
  useMemo: (fn: any) => fn(),
}));

describe('useRunSortOptions', () => {
  test('tests useRunSortOptions without metrics nor params', () => {
    const sortOptions = useRunSortOptions([], []);

    expect(sortOptions).toStrictEqual([
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.DATE,
        value: 'attributes.start_time***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.DATE,
        value: 'attributes.start_time***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.USER,
        value: 'tags.`mlflow.user`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.USER,
        value: 'tags.`mlflow.user`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.RUN_NAME,
        value: 'tags.`mlflow.runName`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.RUN_NAME,
        value: 'tags.`mlflow.runName`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.SOURCE,
        value: 'tags.`mlflow.source.name`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.SOURCE,
        value: 'tags.`mlflow.source.name`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.VERSION,
        value: 'tags.`mlflow.source.git.commit`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.VERSION,
        value: 'tags.`mlflow.source.git.commit`***DESCENDING',
        order: 'DESCENDING',
      },
    ]);
  });

  test('creates RunSortOptions with metrics and params', () => {
    const sortOptions = useRunSortOptions(['metric1', 'metric2'], ['param1', 'param2']);

    expect(sortOptions).toStrictEqual([
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.DATE,
        value: 'attributes.start_time***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.DATE,
        value: 'attributes.start_time***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.USER,
        value: 'tags.`mlflow.user`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.USER,
        value: 'tags.`mlflow.user`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.RUN_NAME,
        value: 'tags.`mlflow.runName`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.RUN_NAME,
        value: 'tags.`mlflow.runName`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.SOURCE,
        value: 'tags.`mlflow.source.name`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.SOURCE,
        value: 'tags.`mlflow.source.name`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.VERSION,
        value: 'tags.`mlflow.source.git.commit`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: ATTRIBUTE_COLUMN_SORT_LABEL.VERSION,
        value: 'tags.`mlflow.source.git.commit`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: 'metric1',
        value: 'metrics.`metric1`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: 'metric1',
        value: 'metrics.`metric1`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: 'metric2',
        value: 'metrics.`metric2`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: 'metric2',
        value: 'metrics.`metric2`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: 'param1',
        value: 'params.`param1`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: 'param1',
        value: 'params.`param1`***DESCENDING',
        order: 'DESCENDING',
      },
      {
        label: 'param2',
        value: 'params.`param2`***ASCENDING',
        order: 'ASCENDING',
      },
      {
        label: 'param2',
        value: 'params.`param2`***DESCENDING',
        order: 'DESCENDING',
      },
    ]);
  });
});
