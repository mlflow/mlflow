import { useMemo, useState } from 'react';

import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxEmpty,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxTrigger,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

import { sameUser } from './queuePermissions';

// Unselected users seeded into the list before searching; the rest of the roster
// is reachable by name through the search box.
const DEFAULT_USER_COUNT = 5;
// Cap on matches shown while searching, so a broad query (e.g. "a") doesn't flood
// the list; the rest are reachable by narrowing the search.
const MAX_SEARCH_MATCHES = 20;
// Scroll cap on the option list (matches the reviewer picker's list height).
const LIST_MAX_HEIGHT = 280;

/**
 * Single-select, searchable picker over the assignable-user roster, used to choose
 * a review queue's owner. Mirrors the "Flag for review" user picker and
 * {@link ReviewerChecklistCombobox}, but selects exactly one user: the caller owns
 * the selected value; this owns the search box. The current selection always leads
 * the list and is always present even if it's off-roster (e.g. an owner no longer
 * returned by the roster query), so reassigning it is never the only option.
 */
export const OwnerCombobox = ({
  componentId,
  usernames,
  selectedUser,
  onSelect,
  disabled,
  dropdownZIndex,
  isLoading,
  error,
}: {
  componentId: string;
  usernames: string[];
  selectedUser: string;
  onSelect: (username: string) => void;
  disabled?: boolean;
  dropdownZIndex?: number;
  /** Whether the assignable-user roster is still loading (drives the empty state). */
  isLoading?: boolean;
  /** Set when the roster failed to load; shown in place of the empty state. */
  error?: Error | null;
}) => {
  const intl = useIntl();
  const [search, setSearch] = useState('');
  const query = search.trim().toLowerCase();

  // The current selection leads (and is always present, even if off-roster),
  // followed by a capped set of other roster users as defaults.
  const defaults = useMemo(() => {
    const others = usernames.filter((u) => !sameUser(u, selectedUser));
    return [...(selectedUser ? [selectedUser] : []), ...others.slice(0, DEFAULT_USER_COUNT)];
  }, [usernames, selectedUser]);

  // While searching, match the whole roster (plus the current selection, which may
  // be off-roster); otherwise show the stable default set.
  const { matches, searchTruncated } = useMemo(() => {
    if (!query) {
      return { matches: defaults, searchTruncated: false };
    }
    // Mirror `defaults`: lead with the selection and drop its roster duplicate by
    // the same case-insensitive `sameUser` rule used for `checked`, so an
    // off-roster owner whose casing differs from a roster entry can't show twice.
    const universe = [...(selectedUser ? [selectedUser] : []), ...usernames.filter((u) => !sameUser(u, selectedUser))];
    const filtered = universe.filter((u) => u.toLowerCase().includes(query));
    return { matches: filtered.slice(0, MAX_SEARCH_MATCHES), searchTruncated: filtered.length > MAX_SEARCH_MATCHES };
  }, [defaults, usernames, selectedUser, query]);

  // More roster users exist than the defaults shown, reachable by search.
  const hasMore = !query && usernames.some((u) => !sameUser(u, selectedUser) && !defaults.includes(u));
  const isEmpty = matches.length === 0;

  const handleSelect = (username: string) => {
    onSelect(username);
    setSearch('');
  };

  const renderItem = (username: string) => (
    <DialogComboboxOptionListSelectItem
      key={username}
      value={username}
      checked={sameUser(username, selectedUser)}
      onChange={() => handleSelect(username)}
    >
      {username}
    </DialogComboboxOptionListSelectItem>
  );

  return (
    <DialogCombobox
      componentId={componentId}
      label={intl.formatMessage({
        defaultMessage: 'Select owner',
        description: 'Review queue: owner dropdown label',
      })}
      value={selectedUser ? [selectedUser] : []}
      onOpenChange={(open) => {
        if (!open) {
          setSearch('');
        }
      }}
    >
      <DialogComboboxTrigger
        allowClear={false}
        disabled={disabled}
        placeholder={intl.formatMessage({
          defaultMessage: 'Select an owner',
          description: 'Review queue: owner dropdown placeholder',
        })}
      />
      <DialogComboboxContent matchTriggerWidth style={{ zIndex: dropdownZIndex }}>
        <DialogComboboxOptionList css={{ maxHeight: LIST_MAX_HEIGHT, overflowY: 'auto', overflowX: 'hidden' }}>
          <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
            {isEmpty
              ? // Wrapped in an array (not a bare element): DialogComboboxOptionListSearch
                // keys off `children.length`, and a single child reads as length-less,
                // so it would fall back to its own generic "No results found".
                [
                  <DialogComboboxEmpty
                    key="__empty"
                    emptyText={
                      // Error first: a failed roster load otherwise reads as an
                      // empty roster ("No assignable users"), hiding the error.
                      error ? (
                        <FormattedMessage
                          defaultMessage="Couldn't load users. Try again."
                          description="Review queue: owner roster failed to load"
                        />
                      ) : query ? (
                        <FormattedMessage
                          defaultMessage="No matching users"
                          description="Review queue: no users match the owner search"
                        />
                      ) : isLoading ? (
                        <FormattedMessage
                          defaultMessage="Loading users…"
                          description="Review queue: owner roster loading"
                        />
                      ) : (
                        <FormattedMessage
                          defaultMessage="No assignable users"
                          description="Review queue: no assignable users available to own the queue"
                        />
                      )
                    }
                  />,
                ]
              : // Pass items as a flat array, not a fragment: a wrapping fragment hides
                // them from DialogComboboxOptionListSearch and they fail to render.
                // (Filtering is ours — with `controlledValue` set, the search skips its
                // built-in child filter.)
                [
                  ...matches.map(renderItem),
                  ...(hasMore
                    ? [
                        <DialogComboboxEmpty
                          key="__more"
                          emptyText={
                            <FormattedMessage
                              defaultMessage="Search to find more users"
                              description="Review queue: hint that more users are searchable for the owner"
                            />
                          }
                        />,
                      ]
                    : []),
                  ...(searchTruncated
                    ? [
                        <DialogComboboxEmpty
                          key="__capped"
                          emptyText={
                            <FormattedMessage
                              defaultMessage="Showing the first {count} matches — refine your search"
                              description="Review queue: hint that the owner search results were capped"
                              values={{ count: MAX_SEARCH_MATCHES }}
                            />
                          }
                        />,
                      ]
                    : []),
                ]}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
