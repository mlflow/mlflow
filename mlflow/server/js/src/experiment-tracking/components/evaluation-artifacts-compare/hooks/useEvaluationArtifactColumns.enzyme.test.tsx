import { mount } from 'enzyme';
import { useEvaluationArtifactColumns } from './useEvaluationArtifactColumns';
import type { EvaluationArtifactTable } from '../../../types';

describe('useEvaluationArtifactColumns', () => {
  const mountTestComponent = (
    storeData: {
      [runUuid: string]: {
        [artifactPath: string]: EvaluationArtifactTable;
      };
    },
    comparedRunUuids: string[],
    tableNames: string[],
  ) => {
    let hookResult: ReturnType<typeof useEvaluationArtifactColumns>;
    const TestComponent = () => {
      hookResult = useEvaluationArtifactColumns(storeData, comparedRunUuids, tableNames);

      return null;
    };

    const wrapper = mount(<TestComponent />);

    return { wrapper, getHookResult: () => hookResult };
  };

  const MOCK_STORE = {
    // Table with column "col_a"
    run_a: { '/table1': { columns: ['col_a'], entries: [], path: '/table1' } },
    // Table with column "col_b"
    run_b: { '/table1': { columns: ['col_b'], entries: [], path: '/table1' } },
    // Table with columns "col_a" and "col_b"
    run_ab: { '/table1': { columns: ['col_a', 'col_b'], entries: [], path: '/table1' } },
    // Table with columns "col_a", "col_b" and "col_c"
    run_abc: { '/table1': { columns: ['col_a', 'col_b', 'col_c'], entries: [], path: '/table1' } },
    // Table with columns "col_a", "col_b" and "col_c" but also "col_b" and "col_c" in the other table
    run_abc_othertable: {
      '/table1': { columns: ['col_a', 'col_b', 'col_c'], entries: [], path: '/table1' },
      '/table2': { columns: ['col_b', 'col_c'], entries: [], path: '/table2' },
    },
  };

  const getResultsForRuns = (runIds: string[], tableNames: string[]) =>
    mountTestComponent(MOCK_STORE, runIds, tableNames).getHookResult();

  it('properly extracts all column names for a set of runs', () => {
    expect(getResultsForRuns(['run_a'], ['/table1']).columns).toEqual(['col_a']);
    expect(getResultsForRuns(['run_a', 'run_b'], ['/table1']).columns).toEqual(['col_a', 'col_b']);
    expect(getResultsForRuns(['run_a', 'run_b', 'run_ab'], ['/table1']).columns).toEqual(['col_a', 'col_b']);
    expect(getResultsForRuns(['run_a', 'run_b', 'run_abc'], ['/table1']).columns).toEqual(['col_a', 'col_b', 'col_c']);
    expect(getResultsForRuns(['run_a', 'run_abc_othertable'], ['/table1', '/table2']).columns).toEqual([
      'col_a',
      'col_b',
      'col_c',
    ]);
  });

  it('properly extracts columns intersection for a set of runs', () => {
    expect(getResultsForRuns(['run_a'], ['/table1']).columnsIntersection).toEqual(['col_a']);
    expect(getResultsForRuns(['run_a', 'run_b'], ['/table1']).columnsIntersection).toEqual([]);
    expect(getResultsForRuns(['run_a', 'run_ab'], ['/table1']).columnsIntersection).toEqual(['col_a']);
    expect(getResultsForRuns(['run_abc_othertable'], ['/table1', '/table2']).columnsIntersection).toEqual([
      'col_b',
      'col_c',
    ]);
    expect(getResultsForRuns(['run_b', 'run_abc_othertable'], ['/table1', '/table2']).columnsIntersection).toEqual([
      'col_b',
    ]);
  });
});
