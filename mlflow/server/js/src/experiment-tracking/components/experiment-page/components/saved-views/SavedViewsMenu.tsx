import { useState } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Button,
  CheckIcon,
  DropdownMenu,
  Input,
  LinkIcon,
  SearchIcon,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import Utils from '../../../../../common/utils/Utils';

export interface SavedViewMenuItem {
  id: string;
  name: string;
  createdAt: number;
}

/**
 * Presentational "Views" dropdown body shared by the runs and traces tabs: a search box, a
 * scrollable list of saved views (name + relative time + per-row copy-link and delete), empty /
 * no-match states, and a "+ Save current view..." footer gated on `canModify`.
 *
 * Deliberately data-source-agnostic — it takes the already-resolved `views` array and callbacks, so
 * each consumer owns its own tag reading (redux slice for runs, Apollo query for traces), clipboard
 * / toast behavior, and delete-confirmation modal. `componentId` (analytics namespace) and
 * `testIdPrefix` (test hooks) are threaded from the consumer so each tab keeps its own registered
 * componentId namespace and its existing test selectors.
 *
 * Render this inside a `DropdownMenu.Content` (it emits `DropdownMenu.Item`s and a `Separator`).
 */
export const SavedViewsMenu = ({
  componentId,
  testIdPrefix,
  views,
  canModify,
  activeViewId,
  onOpen,
  onCopyLink,
  onRequestDelete,
  onSaveCurrent,
}: {
  componentId: string;
  testIdPrefix: string;
  views: SavedViewMenuItem[];
  canModify: boolean;
  // Id of the currently-open view (from the consumer's share-key URL param), or null when none is
  // active. Its row gets a leading checkmark so the list shows which view is applied.
  activeViewId?: string | null;
  onOpen: (id: string) => void;
  onCopyLink: (view: SavedViewMenuItem) => void;
  onRequestDelete: (view: SavedViewMenuItem) => void;
  onSaveCurrent: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [filter, setFilter] = useState('');

  const copyLinkLabel = intl.formatMessage({
    defaultMessage: 'Copy share link',
    description: 'Label for the button that copies a saved view share link',
  });
  const deleteLabel = intl.formatMessage({
    defaultMessage: 'Delete saved view',
    description: 'Accessible label for the button that deletes a saved view',
  });

  const filtered = views.filter((v) => v.name.toLowerCase().includes(filter.toLowerCase()));

  return (
    <>
      <div css={{ padding: `${theme.spacing.sm}px ${theme.spacing.md}px ${theme.spacing.xs}px`, width: 320 }}>
        <Input
          componentId={`${componentId}.search`}
          data-testid={`${testIdPrefix}-search`}
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
              componentId={`${componentId}.item`}
              onClick={() => onOpen(view.id)}
              css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}
              data-testid={`${testIdPrefix}-item-${view.id}`}
            >
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
                <span css={{ width: theme.general.iconFontSize, flexShrink: 0 }}>
                  {view.id === activeViewId && <CheckIcon data-testid={`${testIdPrefix}-active-${view.id}`} />}
                </span>
                <div css={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
                  <Typography.Text bold css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {view.name}
                  </Typography.Text>
                  <Typography.Text size="sm" color="secondary">
                    {Utils.timeSinceStr(new Date(view.createdAt))}
                  </Typography.Text>
                </div>
              </div>
              <div css={{ display: 'flex', gap: theme.spacing.xs, flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
                <Tooltip componentId={`${componentId}.copy_link_tooltip`} content={copyLinkLabel}>
                  <Button
                    componentId={`${componentId}.copy_link`}
                    icon={<LinkIcon />}
                    size="small"
                    aria-label={copyLinkLabel}
                    data-testid={`${testIdPrefix}-copy-link-${view.id}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      onCopyLink(view);
                    }}
                  />
                </Tooltip>
                {canModify && (
                  <Button
                    componentId={`${componentId}.delete`}
                    size="small"
                    icon={<TrashIcon />}
                    danger
                    aria-label={deleteLabel}
                    data-testid={`${testIdPrefix}-delete-${view.id}`}
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
      {canModify && (
        <>
          <DropdownMenu.Separator />
          <DropdownMenu.Item
            componentId={`${componentId}.save_current`}
            data-testid={`${testIdPrefix}-save-current`}
            onClick={onSaveCurrent}
          >
            <FormattedMessage
              defaultMessage="+ Save current view..."
              description="Menu item that opens the modal to save the current view"
            />
          </DropdownMenu.Item>
        </>
      )}
    </>
  );
};
