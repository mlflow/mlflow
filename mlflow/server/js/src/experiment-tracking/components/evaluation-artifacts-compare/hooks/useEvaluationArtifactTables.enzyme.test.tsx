import { mount } from 'enzyme';
import { useEvaluationArtifactTables } from './useEvaluationArtifactTables';
import type { RunRowType } from '../../experiment-page/utils/experimentPage.row-types';
import { MLFLOW_LOGGED_ARTIFACTS_TAG } from '../../../constants';

describe('useEvaluationArtifactTables', () => {
  const mountTestComponent = (comparedRuns: RunRowType[]) => {
    let hookResult: ReturnType<typeof useEvaluationArtifactTables>;
    const TestComponent = () => {
      hookResult = useEvaluationArtifactTables(comparedRuns);

      return null;
    };

    const wrapper = mount(<TestComponent />);

    return { wrapper, getHookResult: () => hookResult };
  };

  const tagDeclarationAlphaBetaGamma = JSON.stringify([
    { type: 'table', path: '/table_alpha.json' },
    { type: 'table', path: '/table_beta.json' },
    { type: 'table', path: '/table_gamma.json' },
  ]);

  const tagDeclarationAlphaBeta = JSON.stringify([
    { type: 'table', path: '/table_alpha.json' },
    { type: 'table', path: '/table_beta.json' },
  ]);

  const tagDeclarationBeta = JSON.stringify([{ type: 'table', path: '/table_beta.json' }]);
  const tagDeclarationGamma = JSON.stringify([{ type: 'table', path: '/table_gamma.json' }]);

  const createMockRun = (runUuid: string, artifactsDeclaration: string): RunRowType =>
    ({
      runUuid,
      tags: {
        [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
          key: MLFLOW_LOGGED_ARTIFACTS_TAG,
          value: artifactsDeclaration,
        },
      },
    } as any);

  it('properly extracts all table names for a set of runs with all tables', () => {
    const { getHookResult } = mountTestComponent([
      createMockRun('run_1', tagDeclarationAlphaBeta),
      createMockRun('run_2', tagDeclarationGamma),
    ]);

    expect(getHookResult().tables).toEqual(
      expect.arrayContaining(['/table_alpha.json', '/table_beta.json', '/table_gamma.json']),
    );
  });

  it('properly extracts all table names for a set of runs with some tables', () => {
    const { getHookResult } = mountTestComponent([createMockRun('run_1', tagDeclarationAlphaBeta)]);

    expect(getHookResult().tables).toEqual(['/table_alpha.json', '/table_beta.json']);
  });

  it('behaves correctly when where there are no tables reported', () => {
    const { getHookResult } = mountTestComponent([createMockRun('run_1', '[]')]);

    expect(getHookResult().tables).toEqual([]);
  });

  it('properly extracts intersection of table names for a set of runs with single common table', () => {
    const { getHookResult } = mountTestComponent([
      createMockRun('run_1', tagDeclarationAlphaBeta),
      createMockRun('run_2', tagDeclarationBeta),
    ]);

    expect(getHookResult().tablesIntersection).toEqual(['/table_beta.json']);
  });

  it('properly extracts intersection of table names for a set of runs with multiple common tables', () => {
    const { getHookResult } = mountTestComponent([
      createMockRun('run_1', tagDeclarationAlphaBeta),
      createMockRun('run_2', tagDeclarationAlphaBetaGamma),
    ]);

    expect(getHookResult().tablesIntersection).toEqual(['/table_alpha.json', '/table_beta.json']);
  });

  it('properly returns empty intersection of table names for a set of runs with no common tables', () => {
    const { getHookResult } = mountTestComponent([
      createMockRun('run_1', tagDeclarationAlphaBeta),
      createMockRun('run_2', tagDeclarationGamma),
    ]);

    expect(getHookResult().tablesIntersection).toEqual([]);
  });

  it('ignores runs with no table artifact metadata', () => {
    const { getHookResult } = mountTestComponent([
      {
        runUuid: 'run_empty',
        tags: {
          something: {
            key: 'something',
            value: 'something-something',
          },
        },
      } as any,
      createMockRun('run_1', tagDeclarationBeta),
    ]);

    expect(getHookResult().tablesByRun).toEqual({
      run_1: ['/table_beta.json'],
    });
  });

  it('ignores runs with no artifact not being tables', () => {
    const { getHookResult } = mountTestComponent([
      {
        runUuid: 'run_empty',
        tags: {
          [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
            key: MLFLOW_LOGGED_ARTIFACTS_TAG,
            value: '[{"type":"unknownType","path":"/file.json"}]',
          },
        },
      } as any,
      createMockRun('run_1', tagDeclarationBeta),
    ]);

    expect(getHookResult().tablesByRun).toEqual({
      run_1: ['/table_beta.json'],
    });
  });

  it('ignores runs with duplicated reported artifact tables', () => {
    const { getHookResult } = mountTestComponent([
      {
        runUuid: 'run_1',
        tags: {
          [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
            key: MLFLOW_LOGGED_ARTIFACTS_TAG,
            value: '[{"type":"table","path":"/table1.json"},{"type":"table","path":"/table1.json"}]',
          },
        },
      } as any,
    ]);

    expect(getHookResult().tablesByRun).toEqual({
      run_1: ['/table1.json'],
    });
  });

  it('throw an error when the declaration tag JSON is malformed', async () => {
    // Suppress console.error() since we're asserting exception
    // and want to keep the terminal clean
    const spy = jest.spyOn(console, 'error');
    spy.mockImplementation(() => {});

    const mountInvalidComponent = () =>
      mountTestComponent([
        {
          runUuid: 'run_empty',
          tags: {
            [MLFLOW_LOGGED_ARTIFACTS_TAG]: {
              key: MLFLOW_LOGGED_ARTIFACTS_TAG,
              value: '[[malformedData[[[[[{{',
            },
          },
        } as any,
        createMockRun('run_1', tagDeclarationBeta),
      ]);

    expect(mountInvalidComponent).toThrow(SyntaxError);

    spy.mockRestore();
  });
});
