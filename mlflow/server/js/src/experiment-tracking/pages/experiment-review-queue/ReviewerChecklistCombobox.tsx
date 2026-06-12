import { useLayoutEffect, useMemo, useRef, useState } from 'react';

import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxEmpty,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxSectionHeader,
  DialogComboboxTrigger,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

// Unselected reviewers shown before the user searches; the rest are reachable by
// name through the search box (mirrors the queues picker's collapsed head).
const COLLAPSED_REVIEWER_COUNT = 3;
// Scroll cap on the option list (matches AddToReviewQueueDropdown's list height).
const LIST_MAX_HEIGHT = 240;

/**
 * Multi-select, searchable checklist of assignable reviewers (usernames), used
 * by the review-queue create/manage modals. Mirrors {@link QuestionChecklistCombobox}
 * and the "Flag for review" user picker: the caller owns which users are checked;
 * this owns the search box. The trigger shows the caller-provided summary (a
 * count), not the joined names — checked state comes from each item's `checked`
 * prop, not from `value`.
 *
 * With no search text, only the first few unselected reviewers are shown (the
 * rest are found by search), and the current selection is grouped at the bottom
 * under a "Selected" header, newest pick first, so it's visible at a glance (an
 * already-selected user that isn't in the roster still shows). Newest-first works
 * because `checkedUsers` is a Set whose insertion order is the order the modal
 * adds picks; reversing it puts the latest on top. Capping the unselected head
 * also keeps unchecking a row from the "Selected" group calm: with more reviewers
 * than the cap, the freshly-unchecked user falls into the hidden (search-only)
 * tail rather than jumping up into a visible row, so the top of the list — and
 * the scroll position — stays put.
 */
export const ReviewerChecklistCombobox = ({
  componentId,
  usernames,
  checkedUsers,
  onToggle,
  triggerValue,
  disabled,
  dropdownZIndex,
}: {
  componentId: string;
  usernames: string[];
  checkedUsers: Set<string>;
  onToggle: (username: string) => void;
  /** Summary string(s) shown in the closed trigger (e.g. "2 reviewers selected"). */
  triggerValue: string[];
  disabled?: boolean;
  dropdownZIndex?: number;
}) => {
  const intl = useIntl();
  const [search, setSearch] = useState('');
  const query = search.trim().toLowerCase();

  // Toggling a row re-renders the list and unmounts the clicked checkbox, which
  // drops focus and lets the list yank the scroll back to the top. Capture the
  // scroll offset on toggle and restore it (keyed on the toggle, not every
  // render) before paint, so the row disappears in place and the scroll holds.
  const listRef = useRef<HTMLDivElement>(null);
  const pendingScrollTop = useRef<number | null>(null);
  const [toggleNonce, setToggleNonce] = useState(0);
  const handleToggle = (username: string) => {
    pendingScrollTop.current = listRef.current?.scrollTop ?? null;
    setToggleNonce((n) => n + 1);
    onToggle(username);
  };
  useLayoutEffect(() => {
    if (pendingScrollTop.current !== null && listRef.current) {
      listRef.current.scrollTop = pendingScrollTop.current;
      pendingScrollTop.current = null;
    }
  }, [toggleNonce]);

  // Live partition so the "Selected" group only ever holds currently-checked
  // reviewers. The searchable universe is the roster ∪ current selection (an
  // existing member may not be in the assignable roster).
  const view = useMemo(() => {
    const all = [...new Set([...usernames, ...checkedUsers])];
    if (query) {
      return { head: all.filter((u) => u.toLowerCase().includes(query)), hasMore: false, selected: [] as string[] };
    }
    // Newest pick first: `checkedUsers` keeps insertion order, so reversing it
    // surfaces the most-recently-added reviewer at the top of the group.
    const selected = [...checkedUsers].reverse();
    const unselected = all.filter((u) => !checkedUsers.has(u));
    return {
      head: unselected.slice(0, COLLAPSED_REVIEWER_COUNT),
      hasMore: unselected.length > COLLAPSED_REVIEWER_COUNT,
      selected,
    };
  }, [usernames, checkedUsers, query]);

  const renderItem = (username: string) => (
    <DialogComboboxOptionListCheckboxItem
      key={username}
      value={username}
      checked={checkedUsers.has(username)}
      onChange={() => handleToggle(username)}
    >
      {username}
    </DialogComboboxOptionListCheckboxItem>
  );

  const isEmpty = view.head.length === 0 && view.selected.length === 0;

  return (
    <DialogCombobox
      componentId={componentId}
      label={intl.formatMessage({
        defaultMessage: 'Select reviewers',
        description: 'Review queue: reviewers dropdown label',
      })}
      multiSelect
      value={triggerValue}
    >
      <DialogComboboxTrigger
        allowClear={false}
        disabled={disabled}
        placeholder={intl.formatMessage({
          defaultMessage: 'Select reviewers',
          description: 'Review queue: reviewers dropdown placeholder',
        })}
      />
      <DialogComboboxContent matchTriggerWidth style={{ zIndex: dropdownZIndex }}>
        <DialogComboboxOptionList
          ref={listRef}
          css={{ maxHeight: LIST_MAX_HEIGHT, overflowY: 'auto', overflowX: 'hidden' }}
        >
          <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={setSearch}>
            {isEmpty
              ? // Wrapped in an array (not a bare element): DialogComboboxOptionListSearch
                // keys off `children.length`, and a single child reads as length-less,
                // so it would fall back to its own generic "No results found".
                [
                  <DialogComboboxEmpty
                    key="__empty"
                    emptyText={
                      query ? (
                        <FormattedMessage
                          defaultMessage="No matching reviewers"
                          description="Review queue: no reviewers match the search"
                        />
                      ) : (
                        <FormattedMessage
                          defaultMessage="No assignable reviewers"
                          description="Review queue: no assignable reviewers available"
                        />
                      )
                    }
                  />,
                ]
              : // Pass the items as a flat array, not a fragment: a wrapping
                // fragment hides them from DialogComboboxOptionListSearch and they
                // fail to render. (Filtering itself is ours — with `controlledValue`
                // set, the search skips its built-in child filter.)
                [
                  ...view.head.map(renderItem),
                  ...(view.hasMore
                    ? [
                        <DialogComboboxEmpty
                          key="__more"
                          emptyText={
                            <FormattedMessage
                              defaultMessage="Search to find more reviewers"
                              description="Review queue: hint that more reviewers are searchable"
                            />
                          }
                        />,
                      ]
                    : []),
                  ...(view.selected.length > 0
                    ? [
                        <DialogComboboxSectionHeader key="__selected-header">
                          <FormattedMessage
                            defaultMessage="Selected"
                            description="Review queue: header above the currently-selected reviewers"
                          />
                        </DialogComboboxSectionHeader>,
                        ...view.selected.map(renderItem),
                      ]
                    : []),
                ]}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
