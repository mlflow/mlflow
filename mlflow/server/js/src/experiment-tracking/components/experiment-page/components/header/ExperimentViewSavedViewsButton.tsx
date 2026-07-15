import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Button,
  DangerModal,
  DropdownMenu,
  Input,
  LinkIcon,
  SearchIcon,
  BookmarkIcon,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import Utils from '../../../../../common/utils/Utils';
import { copyToClipboard } from '../../../../../common/utils/copyToClipboard';
import type { ExperimentEntity } from '../../../../types';
import type { ExperimentPageSearchFacetsState } from '../../models/ExperimentPageSearchFacetsState';
import type { ExperimentPageUIState } from '../../models/ExperimentPageUIState';
import type { SavedViewSummary } from '../../utils/savedViewEnvelope';
import { useSavedViews } from '../../hooks/useSavedViews';
import { ExperimentGetShareLinkModal, getSavedViewShareUrl } from './ExperimentGetShareLinkModal';

/**
 * Per-row "Copy link" for a saved view. Copies the id-referencing link and fires a page-level
 * toast rather than relying on a tooltip (which is easily clipped inside a dropdown menu).
 */
const CopyLinkButton = ({ experimentId, view }: { experimentId: string; view: SavedViewSummary }) => {
  const intl = useIntl();
  const copyLinkLabel = intl.formatMessage({
    defaultMessage: 'Copy share link',
    description: 'Label for the button that copies a saved experiment view share link',
  });
  return (
    <Tooltip componentId="mlflow.experiment_page.saved_views.copy_link_tooltip" content={copyLinkLabel}>
      <Button
        componentId="mlflow.experiment_page.saved_views.copy_link"
        icon={<LinkIcon />}
        size="small"
        aria-label={copyLinkLabel}
        data-testid={`saved-views-copy-link-${view.id}`}
        onClick={async (e) => {
          e.stopPropagation();
          const ok = await copyToClipboard(getSavedViewShareUrl(experimentId, view.id));
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
        }}
      />
    </Tooltip>
  );
};

const SavedViewsListPanel = ({
  experimentId,
  views,
  canModify,
  onOpen,
  onRequestDelete,
}: {
  experimentId: string;
  views: SavedViewSummary[];
  canModify: boolean;
  onOpen: (id: string) => void;
  onRequestDelete: (view: SavedViewSummary) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [filter, setFilter] = useState('');

  const filtered = views.filter((v) => v.name.toLowerCase().includes(filter.toLowerCase()));

  return (
    <>
      <div css={{ padding: `${theme.spacing.sm}px ${theme.spacing.md}px ${theme.spacing.xs}px`, width: 320 }}>
        <Input
          componentId="mlflow.experiment_page.saved_views.search"
          data-testid="saved-views-search"
          prefix={<SearchIcon />}
          placeholder={intl.formatMessage({
            defaultMessage: 'Search saved views',
            description: 'Placeholder for the search input in the saved views dropdown',
          })}
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          autoFocus
        />
      </div>
      <div css={{ maxHeight: 320, overflowY: 'auto' }}>
        {filtered.length === 0 ? (
          <div css={{ padding: theme.spacing.md, textAlign: 'center' }}>
            <Typography.Text color="secondary">
              {views.length === 0 ? (
                <FormattedMessage
                  defaultMessage="No saved views yet"
                  description="Empty state shown in the saved views dropdown when no views exist"
                />
              ) : (
                <FormattedMessage
                  defaultMessage="No views match your search"
                  description="Empty state shown in the saved views dropdown when the search matches nothing"
                />
              )}
            </Typography.Text>
          </div>
        ) : (
          filtered.map((view) => (
            <DropdownMenu.Item
              key={view.id}
              componentId="mlflow.experiment_page.saved_views.item"
              onClick={() => onOpen(view.id)}
              css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}
            >
              <div css={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                <Typography.Text bold css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {view.name}
                </Typography.Text>
                <Typography.Text size="sm" color="secondary">
                  {Utils.timeSinceStr(new Date(view.createdAt))}
                </Typography.Text>
              </div>
              <div css={{ display: 'flex', gap: theme.spacing.xs, flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
                <CopyLinkButton experimentId={experimentId} view={view} />
                {canModify && (
                  <Button
                    componentId="mlflow.experiment_page.saved_views.delete"
                    size="small"
                    icon={<TrashIcon />}
                    danger
                    aria-label={intl.formatMessage({
                      defaultMessage: 'Delete saved view',
                      description: 'Accessible label for the button that deletes a saved experiment view',
                    })}
                    data-testid={`saved-views-delete-${view.id}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onRequestDelete(view);
                    }}
                  />
                )}
              </div>
            </DropdownMenu.Item>
          ))
        )}
      </div>
    </>
  );
};

/**
 * "Views" dropdown in the experiment header: browse, search, open, copy-link and delete saved
 * views, plus a "Save current view..." entry point (edit permission only) that reuses the same
 * Save & share modal as the Share button. Saved views are stored as experiment tags; opening one
 * navigates into the read-only View Mode via the shared-link path.
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
  const { views, canModify, deleteView, openView } = useSavedViews({ experiment });
  const [showSaveModal, setShowSaveModal] = useState(false);
  // Held above the dropdown so the confirm dialog survives the dropdown closing on outside-click:
  // a DangerModal rendered inside DropdownMenu.Content would be torn down when the menu dismisses.
  const [pendingDelete, setPendingDelete] = useState<SavedViewSummary | null>(null);

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId="mlflow.experiment_page.saved_views.trigger"
            icon={<BookmarkIcon />}
            data-testid="saved-views-trigger"
          >
            <FormattedMessage
              defaultMessage="Views"
              description="Label for the saved views dropdown button in the experiment header"
            />
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="end">
          <SavedViewsListPanel
            experimentId={experiment.experimentId}
            views={views}
            canModify={canModify}
            onOpen={openView}
            onRequestDelete={setPendingDelete}
          />
          {canModify && (
            <>
              <DropdownMenu.Separator />
              <DropdownMenu.Item
                componentId="mlflow.experiment_page.saved_views.save_current"
                data-testid="saved-views-save-current"
                onClick={() => setShowSaveModal(true)}
              >
                <FormattedMessage
                  defaultMessage="+ Save current view..."
                  description="Menu item that opens the modal to save the current experiment view"
                />
              </DropdownMenu.Item>
            </>
          )}
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
