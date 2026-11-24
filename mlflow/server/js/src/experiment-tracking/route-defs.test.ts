import { describe, expect, jest, test } from '@jest/globals';
import { shouldEnableNewEvalRunsUi } from '../common/utils/FeatureUtils';
import { getRouteDefs } from './route-defs';
import { RoutePaths } from './routes';

jest.mock('../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../common/utils/FeatureUtils')>('../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

interface RouteDef {
  path?: string;
  children?: RouteDef[];
}

const collectPaths = (defs: RouteDef[]): string[] =>
  defs.flatMap((def) => [...(def.path ? [def.path] : []), ...collectPaths(def.children ?? [])]);

describe('experiment tracking route definitions', () => {
  test('registers the evaluation run details page under the experiment page with the new eval runs UI', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const experimentPage = (getRouteDefs() as RouteDef[]).find((def) => def.path === RoutePaths.experimentPage);

    expect(collectPaths(experimentPage?.children ?? [])).toContain(RoutePaths.experimentPageTabEvaluationRunDetails);
  });

  test('does not register the evaluation run details page without the new eval runs UI', () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);

    expect(collectPaths(getRouteDefs() as RouteDef[])).not.toContain(RoutePaths.experimentPageTabEvaluationRunDetails);
  });
});
