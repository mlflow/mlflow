import { useEffect, useMemo, useRef, useState } from 'react';
import { useIntl } from 'react-intl';
import { EXPERIMENT_PAGE_QUERY_PARAM_KEYS, useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { omit, pick } from 'lodash';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import {
  EXPERIMENT_PAGE_UI_STATE_FIELDS,
  NON_SHAREABLE_UI_STATE_FIELDS,
  createExperimentPageUIState,
} from '../models/ExperimentPageUIState';
import type { ExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
import { createExperimentPageSearchFacetsState } from '../models/ExperimentPageSearchFacetsState';
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
 * URL-embedded shared links carry the (optionally compressed) view-state blob directly in the
 * viewStateShareKey URL param. Legacy links instead store a hash that points to an experiment
 * tag. A compressed blob carries the deflate header; an uncompressed blob is a JSON object
 * literal — both are distinguishable from a bare hash.
 */
const isUrlEmbeddedShareState = (value: string) => isTextCompressedDeflate(value) || value.trimStart().startsWith('{');

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

  // Tracks the url-embedded key we've already consumed this session. See the guard
  // in the effect below for why this is needed.
  const appliedUrlEmbeddedKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (!viewStateShareKey) {
      return;
    }

    const applyParsedState = (parsedSharedViewState: unknown) => {
      // Merge onto defaults so that fields intentionally omitted from a shared link
      // (e.g. per-run pins/visibility) are restored to valid defaults rather than
      // left undefined, which would break consumers that expect arrays/objects.
      const sharedSearchFacetsState = {
        ...createExperimentPageSearchFacetsState(),
        ...pick(parsedSharedViewState, EXPERIMENT_PAGE_QUERY_PARAM_KEYS),
      } as ExperimentPageSearchFacetsState;

      // Drop the non-shareable fields on read too (the writer already omits them): keeps the
      // filter symmetric so a hand-crafted link or a legacy tag can't smuggle per-run state
      // (e.g. runsHidden keyed by UUIDs that don't exist for the recipient) back into the view.
      const sharedUiState = {
        ...createExperimentPageUIState(),
        ...omit(pick(parsedSharedViewState, EXPERIMENT_PAGE_UI_STATE_FIELDS), NON_SHAREABLE_UI_STATE_FIELDS),
      } as ExperimentPageUIState;

      setSharedSearchFacetsState(sharedSearchFacetsState);
      setSharedUiState(sharedUiState);
      setSharedStateError(null);
      setSharedStateErrorMessage(null);
    };

    const reportInvalidShareState = () => {
      setSharedSearchFacetsState(null);
      setSharedUiState(null);
      setSharedStateError(`Error loading shared view state: share key is invalid`);
      setSharedStateErrorMessage(
        intl.formatMessage({
          defaultMessage: `Error loading shared view state: share key is invalid`,
          description: 'Experiment page > share viewstate > error > share key is invalid',
        }),
      );
    };

    // URL-embedded shared link: the view state is carried directly in the viewStateShareKey param
    if (isUrlEmbeddedShareState(viewStateShareKey)) {
      // Apply a given url-embedded link exactly once. This effect also re-runs when
      // `experiment` gets a new reference (e.g. editing experiment notes/tags refetches
      // the entity); without this guard it would re-apply the URL blob over the user's
      // subsequent edits. A different key (navigating to another shared link) still
      // applies. Latch synchronously so a fast re-run can't slip past the pending async.
      if (appliedUrlEmbeddedKeyRef.current === viewStateShareKey) {
        return;
      }
      appliedUrlEmbeddedKeyRef.current = viewStateShareKey;
      const parseUrlEmbeddedShareState = async () => {
        try {
          const parsedSharedViewState = await deserializePersistedState(viewStateShareKey);
          // Must be a plain object: arrays are typeof 'object' but pick()ing them yields {},
          // which would silently reset the recipient's view to defaults instead of erroring.
          if (
            !parsedSharedViewState ||
            typeof parsedSharedViewState !== 'object' ||
            Array.isArray(parsedSharedViewState)
          ) {
            reportInvalidShareState();
            return;
          }
          applyParsedState(parsedSharedViewState);
        } catch (e) {
          reportInvalidShareState();
        }
      };
      parseUrlEmbeddedShareState();
      return;
    }

    // Legacy shared link: the param is a hash pointing to an experiment tag.
    if (!experiment) {
      return;
    }

    // Find the tag with the given share key
    const shareViewTag = experiment.tags.find(
      ({ key }) => key === `${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}${viewStateShareKey}`,
    );

    const tryParseSharedStateFromTag = async (shareViewTag: KeyValueEntity) => {
      try {
        const parsedSharedViewState = await deserializePersistedState(shareViewTag.value);
        applyParsedState(parsedSharedViewState);
      } catch (e) {
        reportInvalidShareState();
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
    // `experiment` is required by the legacy tag-lookup path below; the url-embedded
    // path above is guarded against re-applying when it re-fires on an experiment change.
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
    // Require the display message too: the error and its message are set in two separate
    // setState calls, and because this runs in an async callback under React 17-compat (no
    // automatic batching) the effect can fire on the intermediate render where the message is
    // still null — which would surface an empty error notification before navigating away.
    if (sharedStateError && sharedStateErrorMessage && experiment) {
      // If there's an error with share key, remove it from the URL and notify user
      Utils.displayGlobalErrorNotification(sharedStateErrorMessage, 3);
      navigate(Routes.getExperimentPageRoute(experiment.experimentId), { replace: true });
    }
  }, [sharedStateError, sharedStateErrorMessage, experiment, navigate, disabled]);

  return {
    isViewStateShared,
    sharedStateError,
  };
};
