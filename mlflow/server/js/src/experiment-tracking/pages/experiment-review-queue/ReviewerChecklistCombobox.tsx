import { useEffect, useMemo, useState } from 'react';

import {
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxEmpty,
  DialogComboboxOptionList,
  DialogComboboxOptionListCheckboxItem,
  DialogComboboxOptionListSearch,
  DialogComboboxTrigger,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

// Max reviewers a queue may have assigned. Keep in sync with `MAX_ASSIGNED_USERS`
// in mlflow/genai/review_queues/validation.py — the server rejects more, so the
// picker gates selection at the limit the caller passes (the create modal reserves
// one slot for the creator).
export const MAX_ASSIGNED_USERS = 10;

// Unselected reviewers seeded into the list as defaults before the user searches;
// the rest of the roster is reachable by name through the search box.
const DEFAULT_REVIEWER_COUNT = 3;
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
 * The list is **stable while you read it** and recompacts when you return to it.
 * Recompacting rebuilds it as the selected reviewers (keeping their order, with a
 * reviewer just picked from search leading) followed by a fresh set of unselected
 * defaults — dropping any rows that are no longer selected. It happens when the
 * dropdown opens, when a search is cleared, and when the roster first loads.
 * Between those moments, checking or unchecking a row never moves or removes it.
 */
export const ReviewerChecklistCombobox = ({
  componentId,
  usernames,
  checkedUsers,
  onToggle,
  triggerValue,
  disabled,
  dropdownZIndex,
  isLoading,
  maxSelected,
}: {
  componentId: string;
  usernames: string[];
  checkedUsers: Set<string>;
  onToggle: (username: string) => void;
  /** Summary string(s) shown in the closed trigger (e.g. "2 reviewers selected"). */
  triggerValue: string[];
  disabled?: boolean;
  dropdownZIndex?: number;
  /** Whether the assignable-user roster is still loading (gates the initial seed). */
  isLoading?: boolean;
  /** Max reviewers selectable; unchecked rows disable once this many are checked. */
  maxSelected?: number;
}) => {
  const intl = useIntl();
  const [search, setSearch] = useState('');
  const query = search.trim().toLowerCase();

  // The display order, rebuilt by `recompact` whenever the list view is (re)entered.
  const [order, setOrder] = useState<string[]>([]);
  // Selected reviewers (keeping their current order, newest search-pick leading)
  // followed by a fresh set of unselected defaults; rows no longer selected drop.
  const recompact = () =>
    setOrder((prev) => {
      const selected = [
        ...prev.filter((u) => checkedUsers.has(u)),
        ...[...checkedUsers].filter((u) => !prev.includes(u)),
      ];
      const selectedSet = new Set(selected);
      const defaults = usernames.filter((u) => !selectedSet.has(u)).slice(0, DEFAULT_REVIEWER_COUNT);
      return [...selected, ...defaults];
    });
  useEffect(() => {
    if (!isLoading) {
      recompact();
    }
    // Recompact once the roster resolves; opens and search-clears recompact too.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLoading]);

  const handleToggle = (username: string) => {
    if (query && !checkedUsers.has(username)) {
      // Selecting from search leads the selected group; the list recompacts (drops
      // unselected rows, refreshes defaults) once the search is cleared.
      setOrder((prev) => [username, ...prev.filter((u) => u !== username)]);
    }
    // With no search active, an in-place toggle never moves or removes a row.
    onToggle(username);
  };

  // Recompact when a search is cleared (returning to the list view).
  const handleSearch = (value: string) => {
    if (search.trim() && !value.trim()) {
      recompact();
    }
    setSearch(value);
  };

  // At the cap, keep selected rows toggleable (so you can swap) but block adding more.
  const atLimit = maxSelected !== undefined && checkedUsers.size >= maxSelected;
  const renderItem = (username: string) => {
    const checked = checkedUsers.has(username);
    return (
      <DialogComboboxOptionListCheckboxItem
        key={username}
        value={username}
        checked={checked}
        disabled={!checked && atLimit}
        disabledReason={intl.formatMessage(
          {
            defaultMessage: 'You can assign up to {max} reviewers.',
            description: 'Review queue: reviewer-cap reason on a disabled row',
          },
          { max: maxSelected },
        )}
        onChange={() => handleToggle(username)}
      >
        {username}
      </DialogComboboxOptionListCheckboxItem>
    );
  };

  // While searching, surface roster matches (plus any selected user, who may be
  // off-roster) in a stable roster order so selecting one doesn't reorder the
  // results. With no search, show the stable list.
  const matches = useMemo(() => {
    if (!query) {
      return order;
    }
    const universe = [...new Set([...usernames, ...checkedUsers])];
    return universe.filter((u) => u.toLowerCase().includes(query));
  }, [order, usernames, checkedUsers, query]);

  // More roster reviewers exist than the defaults shown, reachable by search.
  const hasMore = !query && usernames.some((u) => !order.includes(u));
  const isEmpty = matches.length === 0;

  return (
    <DialogCombobox
      componentId={componentId}
      label={intl.formatMessage({
        defaultMessage: 'Select reviewers',
        description: 'Review queue: reviewers dropdown label',
      })}
      multiSelect
      value={triggerValue}
      onOpenChange={(open) => {
        if (open) {
          recompact();
        } else {
          setSearch('');
        }
      }}
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
        <DialogComboboxOptionList css={{ maxHeight: LIST_MAX_HEIGHT, overflowY: 'auto', overflowX: 'hidden' }}>
          <DialogComboboxOptionListSearch controlledValue={search} setControlledValue={handleSearch}>
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
                      ) : isLoading ? (
                        <FormattedMessage
                          defaultMessage="Loading reviewers…"
                          description="Review queue: reviewers roster loading"
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
                  ...matches.map(renderItem),
                  ...(hasMore
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
                ]}
          </DialogComboboxOptionListSearch>
        </DialogComboboxOptionList>
      </DialogComboboxContent>
    </DialogCombobox>
  );
};
