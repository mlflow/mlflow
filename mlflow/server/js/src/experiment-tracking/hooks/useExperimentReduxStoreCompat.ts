import { useDispatch } from 'react-redux';
import type { ReduxState, ThunkDispatch } from '../../redux-types';
import { useEffect } from 'react';
import { GET_EXPERIMENT_API } from '../actions';
import { fulfilled } from '../../common/utils/ActionUtils';
import type { useGetExperimentQuery } from './useExperimentQuery';
import { get } from 'lodash';

/**
 * A small helper hook that consumes experiment from the GraphQL response and puts it in the redux store.
 * Helps to keep the redux store in sync with the GraphQL data so page transitions are smooth.
 */
export const useExperimentReduxStoreCompat = (experimentResponse: ReturnType<typeof useGetExperimentQuery>['data']) => {
  const dispatch = useDispatch<ThunkDispatch>();

  useEffect(() => {
    const experimentId = get(experimentResponse, 'experimentId');
    if (experimentResponse && experimentId) {
      dispatch((thunkDispatch: ThunkDispatch, getStore: () => ReduxState) => {
        const alreadyStored = Boolean(getStore().entities?.experimentsById?.[experimentId]);
        if (!alreadyStored) {
          thunkDispatch({
            type: fulfilled(GET_EXPERIMENT_API),
            payload: { experiment: experimentResponse },
          });
        }
      });
    }
  }, [experimentResponse, dispatch]);
};
