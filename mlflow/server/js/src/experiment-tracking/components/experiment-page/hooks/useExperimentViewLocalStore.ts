import { useMemo } from 'react';
import LocalStorageUtils from '../../../../common/utils/LocalStorageUtils';

/**
 * This hook returns a memoized version of persistable store namespaced for the ExperimentView.
 * It can serve as a store for persisting state for a particular experiment - in this case,
 * the experiment id should be provided as a `identifier` parameter. It can also serve as a store for some
 * general purpose - e.g. you can provide "onboarding" as a identifier to get a store specific
 * for the onboarding section of the experiment view.
 *
 * @param storeIdentifier a unique identifier of created store - can be an experiment id or a general purpose name
 */
export const useExperimentViewLocalStore = (storeIdentifier: string) =>
  useMemo(() => LocalStorageUtils.getStoreForComponent('ExperimentView', storeIdentifier), [storeIdentifier]);
