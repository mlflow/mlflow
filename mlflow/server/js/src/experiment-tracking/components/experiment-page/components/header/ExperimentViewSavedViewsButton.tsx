import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { Button, DangerModal, DropdownMenu, BookmarkIcon } from '@databricks/design-system';

import Utils from '../../../../../common/utils/Utils';
import { copyToClipboard } from '../../../../../common/utils/copyToClipboard';
import type { ExperimentEntity } from '../../../../types';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import type { SavedViewSummary } from '../../utils/savedViewEnvelope';
import { useSavedViews } from '../../hooks/useSavedViews';
import { SavedViewsMenu, type SavedViewMenuItem } from '../saved-views/SavedViewsMenu';
import { ExperimentGetShareLinkModal, getSavedViewShareUrl } from './ExperimentGetShareLinkModal';

/**
 * "Views" dropdown in the experiment header: browse, search, open, copy-link and delete saved
 * views, plus a "Save current view..." entry point (edit permission only) that reuses the same
 * Save & share modal. Saved views are stored as experiment tags; opening one navigates into the
 * read-only View Mode via the shared-link path.
 *
 * The dropdown body is the shared {@link SavedViewsMenu}; this component owns the runs data source
 * ({@link useSavedViews}), the copy-link clipboard + toast, the delete-confirmation dialog, and the
 * save modal. The trigger label and the menu's active-row check reflect the currently-open view
 * (the one whose id is in the share-key URL param), surfaced by `useSavedViews` as `activeViewId`.
 */
export const ExperimentViewSavedViewsButton = ({
  experiment,
  searchFacetsState,
  uiState,
}: {
  experiment: ExperimentEntity;
  searchFacetsState?: ExperimentPageSearchFacetsState;
  uiState?: ExperimentPageUIState;
}) => {
  const intl = useIntl();
  const { views, canModify, deleteView, openView, activeViewId } = useSavedViews({ experiment });
  const [showSaveModal, setShowSaveModal] = useState(false);
  // Held above the dropdown so the confirm dialog survives the dropdown closing on outside-click:
  // a DangerModal rendered inside DropdownMenu.Content would be torn down when the menu dismisses.
  const [pendingDelete, setPendingDelete] = useState<SavedViewSummary | null>(null);
  // Label the trigger with the open view's name so the header reflects which saved view is applied;
  // falls back to the generic "Views" when no view is active or the active id no longer resolves.
  const activeView = activeViewId ? views.find((view) => view.id === activeViewId) : undefined;

  // Copy the id-referencing link and fire a page-level toast rather than relying on a tooltip
  // (which is easily clipped inside a dropdown menu).
  const handleCopyLink = async (view: SavedViewMenuItem) => {
    const ok = await copyToClipboard(getSavedViewShareUrl(experiment.experimentId, view.id));
    if (ok) {
      Utils.displayGlobalInfoNotification(
        intl.formatMessage(
          {
            defaultMessage: 'Link to "{name}" copied — anyone with access can open this view.',
            description: 'Confirmation toast shown after copying a saved experiment view share link',
          },
          { name: view.name },
        ),
        3,
      );
    } else {
      Utils.displayGlobalErrorNotification(
        intl.formatMessage({
          defaultMessage: 'Copy failed — clipboard unavailable.',
          description: 'Error toast shown when copying a saved experiment view share link fails',
        }),
        3,
      );
    }
  };

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="mlflow.experiment_page.saved_views.trigger"
            icon={<BookmarkIcon />}
            data-testid="saved-views-trigger"
          >
            {activeView ? (
              <span css={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {activeView.name}
              </span>
            ) : (
              <FormattedMessage
                defaultMessage="Views"
                description="Label for the saved views dropdown button in the experiment header"
              />
            )}
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="end">
          <SavedViewsMenu
            componentId="mlflow.experiment_page.saved_views"
            testIdPrefix="saved-views"
            views={views}
            canModify={canModify}
            activeViewId={activeViewId}
            onOpen={openView}
            onCopyLink={handleCopyLink}
            onRequestDelete={setPendingDelete}
            onSaveCurrent={() => setShowSaveModal(true)}
          />
        </DropdownMenu.Content>
      </DropdownMenu.Root>
      <DangerModal
        componentId="mlflow.experiment_page.saved_views.delete_confirm"
        visible={Boolean(pendingDelete)}
        onCancel={() => setPendingDelete(null)}
        onOk={() => {
          if (pendingDelete) {
            deleteView(pendingDelete.id);
          }
          setPendingDelete(null);
        }}
        title={
          <FormattedMessage
            defaultMessage="Delete saved view"
            description="Title of the confirmation dialog for deleting a saved experiment view"
          />
        }
        okText={
          <FormattedMessage defaultMessage="Delete" description="Confirm button for deleting a saved experiment view" />
        }
      >
        <FormattedMessage
          defaultMessage={`Delete "{name}"? This can't be undone.`}
          description="Body of the confirmation dialog for deleting a saved experiment view"
          values={{ name: pendingDelete?.name }}
        />
      </DangerModal>
      <ExperimentGetShareLinkModal
        experimentId={experiment.experimentId}
        searchFacetsState={searchFacetsState}
        uiState={uiState}
        visible={showSaveModal}
        onCancel={() => setShowSaveModal(false)}
      />
    </>
  );
};
