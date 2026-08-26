import { useState, type ReactNode } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  Button,
  CheckIcon,
  DropdownMenu,
  Input,
  LinkIcon,
  PencilIcon,
  RefreshIcon,
  SearchIcon,
  Tooltip,
  TrashIcon,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

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
  onSelectDefault,
  sharedViewActive,
  onOverrideActive,
  onDiscardActive,
  overrideLabel,
}: {
  componentId: string;
  testIdPrefix: string;
  views: SavedViewMenuItem[];
  canModify: boolean;
  // Id of the currently-open view (from the consumer's share-key URL param), or null when none is
  // active. Its row gets a leading checkmark so the list shows which view is applied.
  activeViewId?: string | null;
  onOpen: (id: string) => void;
  // When provided, a pinned "Default view" row is shown above the list; clicking it returns the table
  // to its default (unfiltered) state. Omit it and the row (and its checkmark) don't render.
  onSelectDefault?: () => void;
  onCopyLink: (view: SavedViewMenuItem) => void;
  onRequestDelete: (view: SavedViewMenuItem) => void;
  onSaveCurrent: () => void;
  // When a shared/previewed view is applied, the menu hosts the persistent Override / Discard
  // actions (so they survive dismissing the banner). Rendered only when `sharedViewActive` is true
  // AND both handlers are provided; consumers that pass none get the menu exactly as before.
  sharedViewActive?: boolean;
  onOverrideActive?: () => void;
  onDiscardActive?: () => void;
  // Per-tab wording for the override entry ("Override my view" on traces, "Override saved view" on
  // runs). Falls back to a generic label when omitted.
  overrideLabel?: ReactNode;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [filter, setFilter] = useState('');
  const showSharedViewActions = Boolean(sharedViewActive && onOverrideActive && onDiscardActive);

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
      <DropdownMenu.Label
        css={{
          textTransform: 'uppercase',
          fontSize: theme.typography.fontSizeMd,
          fontWeight: 600,
          letterSpacing: '0.04em',
        }}
      >
        <FormattedMessage defaultMessage="Saved views" description="Header at the top of the saved views dropdown" />
      </DropdownMenu.Label>
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
          // Stop keystrokes from bubbling to the DropdownMenu, whose built-in typeahead would otherwise
          // steal each letter to jump-focus the first matching row — pulling focus out of this input.
          onKeyDown={(e) => e.stopPropagation()}
          autoFocus
        />
      </div>
      {onSelectDefault && (
        // "Default" is the absence of a saved view, not a stored tag — no copy/delete, pinned above the list.
        <>
          <DropdownMenu.Item
            componentId={`${componentId}.default_item`}
            data-testid={`${testIdPrefix}-default`}
            onClick={onSelectDefault}
            css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: theme.spacing.sm }}
          >
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
              <span css={{ width: theme.general.iconFontSize, flexShrink: 0 }}>
                {!activeViewId && <CheckIcon data-testid={`${testIdPrefix}-active-default`} />}
              </span>
              <Typography.Text>
                <FormattedMessage
                  defaultMessage="Default view"
                  description="Menu item that leaves every saved view and returns the table to its default state"
                />
              </Typography.Text>
            </div>
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Default"
                description="Tag marking the built-in default view row, vs the user-created views below"
              />
            </Typography.Text>
          </DropdownMenu.Item>
          <DropdownMenu.Separator />
        </>
      )}
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
              css={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                gap: theme.spacing.sm,
                // Reveal row actions on hover and Radix keyboard-highlight (so arrow-key users reach
                // them). `opacity` not `display` keeps them in layout + the a11y tree. The active row
                // keeps them visible (no hover affordance of its own). On this rule so it beats the
                // descendant selector below.
                '& .saved-view-row-actions': { opacity: view.id === activeViewId ? 1 : 0 },
                '&:hover .saved-view-row-actions, &[data-highlighted] .saved-view-row-actions': {
                  opacity: 1,
                },
              }}
              data-testid={`${testIdPrefix}-item-${view.id}`}
            >
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, minWidth: 0 }}>
                <span css={{ width: theme.general.iconFontSize, flexShrink: 0 }}>
                  {view.id === activeViewId && <CheckIcon data-testid={`${testIdPrefix}-active-${view.id}`} />}
                </span>
                <Typography.Text css={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {view.name}
                </Typography.Text>
              </div>
              <div
                className="saved-view-row-actions"
                css={{ display: 'flex', gap: theme.spacing.xs, flexShrink: 0 }}
                onClick={(e) => e.stopPropagation()}
              >
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
      {(canModify || showSharedViewActions) && <DropdownMenu.Separator />}
      {showSharedViewActions && (
        <>
          <DropdownMenu.Item
            componentId={`${componentId}.override_active`}
            data-testid={`${testIdPrefix}-override-active`}
            onClick={onOverrideActive}
            css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
          >
            <span css={{ width: theme.general.iconFontSize, flexShrink: 0 }} aria-hidden>
              <PencilIcon />
            </span>
            {overrideLabel ?? (
              <FormattedMessage
                defaultMessage="Override my view"
                description="Menu item that adopts the currently-applied shared view into the user's own view"
              />
            )}
          </DropdownMenu.Item>
          <DropdownMenu.Item
            componentId={`${componentId}.discard_active`}
            data-testid={`${testIdPrefix}-discard-active`}
            onClick={onDiscardActive}
            css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
          >
            <span css={{ width: theme.general.iconFontSize, flexShrink: 0 }} aria-hidden>
              <RefreshIcon />
            </span>
            <FormattedMessage
              defaultMessage="Discard shared view"
              description="Menu item that discards the currently-applied shared view and restores the user's own view"
            />
          </DropdownMenu.Item>
        </>
      )}
      {canModify && (
        <DropdownMenu.Item
          componentId={`${componentId}.save_current`}
          data-testid={`${testIdPrefix}-save-current`}
          onClick={onSaveCurrent}
          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}
        >
          <span css={{ width: theme.general.iconFontSize, flexShrink: 0, textAlign: 'center' }} aria-hidden>
            +
          </span>
          <FormattedMessage
            defaultMessage="Save as new view"
            description="Menu item that opens the modal to save the current view"
          />
        </DropdownMenu.Item>
      )}
    </>
  );
};
