import { useEffect, useMemo } from 'react';
import { useIntl } from 'react-intl';
import { EXPERIMENT_PAGE_QUERY_PARAM_KEYS, useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { pick } from 'lodash';
import { EXPERIMENT_PAGE_UI_STATE_FIELDS, ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { ExperimentEntity } from '../../../types';
import { useNavigate, useSearchParams } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import {
  EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX,
  EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY,
} from '../../../constants';
import Routes from '../../../routes';

/**
 * Hook that handles loading shared view state from URL and updating the search facets/UI state accordingly
 */
export const useSharedExperimentViewState = (
  uiStateSetter: React.Dispatch<React.SetStateAction<ExperimentPageUIState>>,
  experiment?: ExperimentEntity,
  disabled = false,
) => {
  const [searchParams] = useSearchParams();
  const intl = useIntl();
  const viewStateShareKey = searchParams.get(EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY);

  const isViewStateShared = Boolean(viewStateShareKey);

  const updateSearchFacets = useUpdateExperimentPageSearchFacets();

  const { sharedSearchFacetsState, sharedUiState, sharedStateError, sharedStateErrorMessage } = useMemo(() => {
    if (!viewStateShareKey || !experiment) {
      return {
        sharedSearchFacetsState: null,
        sharedUiState: null,
        sharedStateError: null,
        sharedStateErrorMessage: null,
      };
    }

    // Find the tag with the given share key
    const shareViewTag = experiment.tags.find(
      ({ key }) => key === `${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}${viewStateShareKey}`,
    );

    // If the tag exists, parse the view state from the tag value
    if (shareViewTag) {
      try {
        const parsedSharedViewState = JSON.parse(shareViewTag.value);

        // First, extract search facets part of the shared view state
        const sharedSearchFacetsState = pick(
          parsedSharedViewState,
          EXPERIMENT_PAGE_QUERY_PARAM_KEYS,
        ) as ExperimentPageSearchFacetsState;

        // Then, extract UI state part of the shared view state
        const sharedUiState = pick(parsedSharedViewState, EXPERIMENT_PAGE_UI_STATE_FIELDS) as ExperimentPageUIState;

        return {
          sharedSearchFacetsState,
          sharedUiState,
          sharedStateError: null,
          sharedStateErrorMessage: null,
        };
      } catch (e) {
        return {
          sharedSearchFacetsState: null,
          sharedUiState: null,
          sharedStateError: `Error loading shared view state: share key is invalid`,
          sharedStateErrorMessage: intl.formatMessage({
            defaultMessage: `Error loading shared view state: share key is invalid`,
            description: 'Experiment page > share viewstate > error > share key is invalid',
          }),
        };
      }
    }

    return {
      sharedSearchFacetsState: null,
      sharedUiState: null,
      sharedStateError: `Error loading shared view state: share key ${viewStateShareKey} does not exist`,
      sharedStateErrorMessage: intl.formatMessage(
        {
          defaultMessage: `Error loading shared view state: share key "{viewStateShareKey}" does not exist`,
          description: 'Experiment page > share viewstate > error > share key does not exist',
        },
        {
          viewStateShareKey,
        },
      ),
    };
  }, [experiment, viewStateShareKey, intl]);

  useEffect(() => {
    if (!sharedSearchFacetsState || disabled) {
      return;
    }
    updateSearchFacets(sharedSearchFacetsState, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sharedSearchFacetsState, disabled]);

  useEffect(() => {
    if (!sharedUiState || disabled) {
      return;
    }
    uiStateSetter(sharedUiState);
  }, [uiStateSetter, sharedUiState, disabled]);

  const navigate = useNavigate();

  useEffect(() => {
    if (disabled) {
      return;
    }
    if (sharedStateError && experiment) {
      // If there's an error with share key, remove it from the URL and notify user
      Utils.logErrorAndNotifyUser(new Error(sharedStateError));
      Utils.displayGlobalErrorNotification(sharedStateErrorMessage, 3);
      navigate(Routes.getExperimentPageRoute(experiment.experimentId), { replace: true });
    }
  }, [sharedStateError, sharedStateErrorMessage, experiment, navigate, disabled]);

  return {
    isViewStateShared,
    sharedStateError,
  };
};
