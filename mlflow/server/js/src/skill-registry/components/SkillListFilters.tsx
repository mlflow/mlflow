import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  SimpleSelect,
  SimpleSelectOption,
  TableFilterInput,
  TableFilterLayout,
  ToggleButton,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxRoot,
  useComboboxState,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { SkillSearchInputHelpTooltip } from './SkillSearchInputHelpTooltip';
import { formatSkillSourceLabel, SKILL_CATALOG_SOURCE_TYPE_OPTIONS } from '../utils';
import type { SkillSourceType } from '../types';

const ALL_VALUE = 'all';
const SEARCH_FILTER_WIDTH = 240;
const ORGANIZATION_FILTER_WIDTH = 200;
const SOURCE_FILTER_WIDTH = 180;

const compactFilterInputProps = (width: number) => ({
  ignoreFilterMediaSizing: true,
  css: { width: '100%' },
  containerProps: { style: { width, flex: '0 0 auto' } },
});

const compactSelectStyles = {
  width: SOURCE_FILTER_WIDTH,
  flex: '0 0 auto',
};

const normalizeOrganization = (value: string) => value.trim().replace(/^@/, '');

const SkillOrganizationTypeahead = ({
  organization,
  onOrganizationChange,
  organizations,
}: {
  organization: string;
  onOrganizationChange: (value: string) => void;
  organizations: string[];
}) => {
  const intl = useIntl();
  const items = useMemo(() => organizations.filter(Boolean).map((org) => `@${org}`), [organizations]);
  const [filteredItems, setFilteredItems] = useState(items);

  useEffect(() => {
    setFilteredItems(items);
  }, [items]);

  const selected = organization ? `@${organization}` : undefined;
  const applyOrganization = useCallback(
    (item: string | null | undefined) => {
      const next = typeof item === 'string' ? normalizeOrganization(item) : '';
      if (next === organization) {
        return;
      }
      queueMicrotask(() => onOrganizationChange(next));
    },
    [organization, onOrganizationChange],
  );

  const comboboxState = useComboboxState<string>({
    componentId: 'mlflow.skill_registry.filter.organization',
    allItems: items,
    items: filteredItems,
    setItems: setFilteredItems,
    multiSelect: false,
    allowNewValue: true,
    preventUnsetOnBlur: true,
    itemToString: (item) => item ?? '',
    matcher: (item, query) => {
      const needle = query.replace(/^@/, '').toLowerCase();
      return item?.replace(/^@/, '').toLowerCase().includes(needle) ?? false;
    },
    formValue: selected,
    initialInputValue: selected ?? '',
    formOnChange: applyOrganization,
  });

  return (
    <div css={{ width: ORGANIZATION_FILTER_WIDTH, flex: '0 0 auto' }}>
      <TypeaheadComboboxRoot id="mlflow.skill_registry.filter.organization" comboboxState={comboboxState}>
        <TypeaheadComboboxInput
          placeholder={intl.formatMessage({
            defaultMessage: 'All organizations',
            description: 'Placeholder for Skill Registry organization typeahead when no organization is selected',
          })}
          comboboxState={comboboxState}
          formOnChange={applyOrganization}
          allowClear
          clearInputValueOnFocus={false}
          showComboboxToggleButton
        />
        <TypeaheadComboboxMenu comboboxState={comboboxState} matchTriggerWidth>
          {filteredItems.map((item, index) => (
            <TypeaheadComboboxMenuItem key={item} item={item} index={index} comboboxState={comboboxState}>
              {item}
            </TypeaheadComboboxMenuItem>
          ))}
        </TypeaheadComboboxMenu>
      </TypeaheadComboboxRoot>
    </div>
  );
};

export const SkillListFilters = ({
  searchFilter,
  onSearchFilterChange,
  filterActive,
  onFilterActiveChange,
  organization,
  onOrganizationChange,
  organizations,
  sourceType,
  onSourceTypeChange,
  actions,
}: {
  searchFilter: string;
  onSearchFilterChange: (value: string) => void;
  filterActive: boolean;
  onFilterActiveChange: (checked: boolean) => void;
  organization: string;
  onOrganizationChange: (value: string) => void;
  organizations: string[];
  sourceType: SkillSourceType | '';
  onSourceTypeChange: (value: SkillSourceType | '') => void;
  actions?: ReactNode;
}) => {
  const intl = useIntl();
  const sourceTypeLabel = intl.formatMessage({
    defaultMessage: 'Source',
    description: 'Aria label for Skill Registry source type filter',
  });

  return (
    <TableFilterLayout css={{ marginBottom: '16px', width: '100%' }} actions={actions}>
      <TableFilterInput
        placeholder={intl.formatMessage({
          defaultMessage: 'Search skills',
          description: 'Placeholder for Skill Registry free-text search',
        })}
        componentId="mlflow.skill_registry.search"
        value={searchFilter}
        onChange={(e) => onSearchFilterChange(e.target.value)}
        suffix={<SkillSearchInputHelpTooltip />}
        {...compactFilterInputProps(SEARCH_FILTER_WIDTH)}
      />
      <ToggleButton
        componentId="mlflow.skill_registry.filter.active"
        pressed={filterActive}
        onPressedChange={(pressed) => onFilterActiveChange(pressed)}
        aria-label={intl.formatMessage({
          defaultMessage: 'Filter by active status',
          description: 'Aria label for Skill Registry active status filter toggle',
        })}
      >
        <FormattedMessage
          defaultMessage="Active"
          description="Filter toggle for skills with an active latest version"
        />
      </ToggleButton>
      <SkillOrganizationTypeahead
        key={organization || 'all'}
        organization={organization}
        onOrganizationChange={onOrganizationChange}
        organizations={organizations}
      />
      <SimpleSelect
        id="mlflow.skill_registry.filter.source_type"
        componentId="mlflow.skill_registry.filter.source_type"
        aria-label={sourceTypeLabel}
        value={sourceType || ALL_VALUE}
        onChange={({ target }) =>
          onSourceTypeChange(target.value === ALL_VALUE ? '' : (target.value as SkillSourceType))
        }
        css={compactSelectStyles}
      >
        <SimpleSelectOption value={ALL_VALUE}>
          <FormattedMessage
            defaultMessage="All sources"
            description="Skill catalog source filter option for any source"
          />
        </SimpleSelectOption>
        {SKILL_CATALOG_SOURCE_TYPE_OPTIONS.map((option) => (
          <SimpleSelectOption key={option} value={option}>
            {formatSkillSourceLabel(option)}
          </SimpleSelectOption>
        ))}
      </SimpleSelect>
    </TableFilterLayout>
  );
};
