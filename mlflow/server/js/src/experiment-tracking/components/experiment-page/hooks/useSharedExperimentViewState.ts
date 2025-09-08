import { useEffect, useMemo, useState } from 'react';
import { useIntl } from 'react-intl';
import { EXPERIMENT_PAGE_QUERY_PARAM_KEYS, useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { pick } from 'lodash';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { EXPERIMENT_PAGE_UI_STATE_FIELDS } from '../models/ExperimentPageUIState';
import type { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import type { ExperimentEntity } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import { useNavigate, useSearchParams } from '../../../../common/utils/RoutingUtils';
import Utils from '../../../../common/utils/Utils';
import {
  EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX,
  EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY,
} from '../../../constants';
import Routes from '../../../routes';
import { isTextCompressedDeflate, textDecompressDeflate } from '../../../../common/utils/StringUtils';

const deserializePersistedState = async (state: string) => {
  if (isTextCompressedDeflate(state)) {
    return JSON.parse(await textDecompressDeflate(state));
  }
  return JSON.parse(state);
};

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

  const [sharedSearchFacetsState, setSharedSearchFacetsState] = useState<ExperimentPageSearchFacetsState | null>(null);
  const [sharedUiState, setSharedUiState] = useState<ExperimentPageUIState | null>(null);
  const [sharedStateError, setSharedStateError] = useState<string | null>(null);
  const [sharedStateErrorMessage, setSharedStateErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!viewStateShareKey || !experiment) {
      return;
    }

    // Find the tag with the given share key
    const shareViewTag = experiment.tags.find(
      ({ key }) => key === `${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}${viewStateShareKey}`,
    );

    const tryParseSharedStateFromTag = async (shareViewTag: KeyValueEntity) => {
      try {
        const parsedSharedViewState = await deserializePersistedState(shareViewTag.value);
        // First, extract search facets part of the shared view state
        const sharedSearchFacetsState = pick(
          parsedSharedViewState,
          EXPERIMENT_PAGE_QUERY_PARAM_KEYS,
        ) as ExperimentPageSearchFacetsState;

        // Then, extract UI state part of the shared view state
        const sharedUiState = pick(parsedSharedViewState, EXPERIMENT_PAGE_UI_STATE_FIELDS) as ExperimentPageUIState;

        setSharedSearchFacetsState(sharedSearchFacetsState);
        setSharedUiState(sharedUiState);
        setSharedStateError(null);
        setSharedStateErrorMessage(null);
      } catch (e) {
        setSharedSearchFacetsState(null);
        setSharedUiState(null);
        setSharedStateError(`Error loading shared view state: share key is invalid`);
        setSharedStateErrorMessage(
          intl.formatMessage({
            defaultMessage: `Error loading shared view state: share key is invalid`,
            description: 'Experiment page > share viewstate > error > share key is invalid',
          }),
        );
      }
    };

    // If the tag exists, parse the view state from the tag value
    if (!shareViewTag) {
      setSharedSearchFacetsState(null);
      setSharedUiState(null);
      setSharedStateError(`Error loading shared view state: share key ${viewStateShareKey} does not exist`);
      setSharedStateErrorMessage(
        intl.formatMessage(
          {
            defaultMessage: `Error loading shared view state: share key "{viewStateShareKey}" does not exist`,
            description: 'Experiment page > share viewstate > error > share key does not exist',
          },
          {
            viewStateShareKey,
          },
        ),
      );
      return;
    }

    tryParseSharedStateFromTag(shareViewTag);
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
