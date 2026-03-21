import { useCallback, useEffect, useMemo, useState } from 'react';
import { FormattedMessage } from 'react-intl';
import { GenericSkeleton, Input, Modal } from '@databricks/design-system';
import { useDispatch } from 'react-redux';
import type { ThunkDispatch } from '../../../../../redux-types';
import { setExperimentTagApi } from '../../../../actions';
import Routes from '../../../../routes';
import { CopyButton } from '../../../../../shared/building_blocks/CopyButton';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import { getStringSHA256, textCompressDeflate } from '../../../../../common/utils/StringUtils';
import Utils from '../../../../../common/utils/Utils';
import {
  EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX,
  EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY,
} from '../../../../constants';
import { shouldUseCompressedExperimentViewSharedState } from '../../../../../common/utils/FeatureUtils';
import {
  EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY,
  useExperimentPageViewMode,
} from '../../hooks/useExperimentPageViewMode';
import type { ExperimentViewRunsCompareMode } from '../../../../types';

type GetShareLinkModalProps = {
  onCancel: () => void;
  visible: boolean;
  experimentIds: string[];
  searchFacetsState: ExperimentPageSearchFacetsState;
  uiState: ExperimentPageUIState;
};

type ShareableViewState = ExperimentPageSearchFacetsState & ExperimentPageUIState;

// Typescript-based test to ensure that the keys of the two states are disjoint.
// If they are not disjoint, the state serialization will not work as expected.
const _arePersistedStatesDisjoint: [
  keyof ExperimentPageSearchFacetsState & keyof ExperimentPageUIState extends never ? true : false,
] = [true];

const serializePersistedState = async (state: ShareableViewState) => {
  if (shouldUseCompressedExperimentViewSharedState()) {
    return textCompressDeflate(JSON.stringify(state));
  }
  return JSON.stringify(state);
};

const getShareableUrl = (experimentId: string, shareStateHash: string, viewMode?: ExperimentViewRunsCompareMode) => {
  // As a start, get the route
  const route = Routes.getExperimentPageRoute(experimentId);

  // Begin building the query params
  const queryParams = new URLSearchParams();

  // Add the share state hash
  queryParams.set(EXPERIMENT_PAGE_VIEW_STATE_SHARE_URL_PARAM_KEY, shareStateHash);

  // If the view mode is set, add it to the query params
  if (viewMode) {
    queryParams.set(EXPERIMENT_PAGE_VIEW_MODE_QUERY_PARAM_KEY, viewMode);
  }

  // In regular implementation, build the hash part of the URL
  const params = queryParams.toString();
  const hashParam = `${route}${params?.startsWith('?') ? '' : '?'}${params}`;
  const shareURL = `${window.location.origin}${window.location.pathname}#${hashParam}`;
  return shareURL;
};

/**
 * Modal that displays shareable link for the experiment page.
 * The shareable state is created by serializing the search facets and UI state and storing
 * it as a tag on the experiment.
 */
export const ExperimentGetShareLinkModal = ({
  onCancel,
  visible,
  experimentIds,
  searchFacetsState,
  uiState,
}: GetShareLinkModalProps) => {
  const [sharedStateUrl, setSharedStateUrl] = useState<string>('');
  const [linkInProgress, setLinkInProgress] = useState(true);
  const [generatedState, setGeneratedState] = useState<ShareableViewState | null>(null);
  const [viewMode] = useExperimentPageViewMode();

  const dispatch = useDispatch<ThunkDispatch>();

  const stateToSerialize = useMemo(() => ({ ...searchFacetsState, ...uiState }), [searchFacetsState, uiState]);

  const createSerializedState = useCallback(
    async (state: ShareableViewState) => {
      if (experimentIds.length > 1) {
        setLinkInProgress(false);
        setGeneratedState(state);
        setSharedStateUrl(window.location.href);
        return;
      }
      setLinkInProgress(true);
      const [experimentId] = experimentIds;
      try {
        const data = await serializePersistedState(state);
        const hash = await getStringSHA256(data);

        const tagName = `${EXPERIMENT_PAGE_VIEW_STATE_SHARE_TAG_PREFIX}${hash}`;

        await dispatch(setExperimentTagApi(experimentId, tagName, data));

        setLinkInProgress(false);
        setGeneratedState(state);

        setSharedStateUrl(getShareableUrl(experimentId, hash, viewMode));
      } catch (e) {
        Utils.logErrorAndNotifyUser('Failed to create shareable link for experiment');
        throw e;
      }
    },
    [dispatch, experimentIds, viewMode],
  );

  useEffect(() => {
    if (!visible || generatedState === stateToSerialize) {
      return;
    }
    createSerializedState(stateToSerialize);
  }, [visible, createSerializedState, generatedState, stateToSerialize]);

  return (
    <Modal
      componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentgetsharelinkmodal.tsx_101"
      title={
        <FormattedMessage
          defaultMessage="Get shareable link"
          description='Title text for the experiment "Get link" modal'
        />
      }
      visible={visible}
      onCancel={onCancel}
    >
      <div css={{ display: 'flex', gap: 8 }}>
        {linkInProgress ? (
          <GenericSkeleton css={{ flex: 1 }} />
        ) : (
          <Input
            componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_header_experimentgetsharelinkmodal.tsx_115"
            placeholder="Click button on the right to create shareable state"
            value={sharedStateUrl}
            readOnly
          />
        )}
        <CopyButton loading={linkInProgress} copyText={sharedStateUrl} data-testid="share-link-copy-button" />
      </div>
    </Modal>
  );
};
