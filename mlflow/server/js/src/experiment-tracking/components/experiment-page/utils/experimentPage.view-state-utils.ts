import type { ExperimentPageViewState } from '../models/ExperimentPageViewState';

export const mapSelectedRunUuids = (
  runUuids: string[],
): ExperimentPageViewState['runsSelected'] =>
  runUuids.reduce<ExperimentPageViewState['runsSelected']>((selected, runUuid) => {
    selected[runUuid] = true;
    return selected;
  }, {});

export const getSelectedRunsCount = (runsSelected: ExperimentPageViewState['runsSelected']) =>
  Object.values(runsSelected).filter(Boolean).length;

export const shouldShowRunsBulkActions = (
  runsSelected: ExperimentPageViewState['runsSelected'],
  columnSelectorVisible: boolean,
) => getSelectedRunsCount(runsSelected) > 0 && !columnSelectorVisible;

export const createOpenColumnSelectorViewStatePatch = (): Pick<
  ExperimentPageViewState,
  'columnSelectorVisible'
> => ({
  columnSelectorVisible: true,
});
