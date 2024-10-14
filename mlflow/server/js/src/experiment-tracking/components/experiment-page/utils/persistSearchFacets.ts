import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';
import Utils from '../../../../common/utils/Utils';

import { ExperimentPageUIState, createExperimentPageUIState } from '../models/ExperimentPageUIState';
import {
  ExperimentPageSearchFacetsState,
  createExperimentPageSearchFacetsState,
} from '../models/ExperimentPageSearchFacetsState';

/**
 * Loads current view state (UI state, view state) in the local storage.
 */
export function loadExperimentViewState(idKey: string) {
  try {
    const localStorageInstance = LocalStorageUtils.getStoreForComponent('ExperimentPage', idKey);
    return localStorageInstance.loadComponentState();
  } catch {
    Utils.logErrorAndNotifyUser(`Error: malformed persisted search state for experiment(s) ${idKey}`);

    return {
      ...createExperimentPageUIState(),
      ...createExperimentPageSearchFacetsState(),
    };
  }
}

/**
 * Persists view state (UI state, view state) in the local storage.
 */
export function saveExperimentViewState(data: ExperimentPageUIState & ExperimentPageSearchFacetsState, idKey: string) {
  const localStorageInstance = LocalStorageUtils.getStoreForComponent('ExperimentPage', idKey);
  localStorageInstance.saveComponentState(data);
}
