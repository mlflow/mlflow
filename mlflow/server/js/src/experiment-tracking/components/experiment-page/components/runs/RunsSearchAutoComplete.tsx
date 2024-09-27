import { isEqual } from 'lodash';
import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import {
  AutoComplete,
  Input,
  SearchIcon,
  LegacyTooltip,
  InfoIcon,
  Button,
  CloseIcon,
  useDesignSystemTheme,
  Tooltip,
  InfoFillIcon,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ErrorWrapper } from '../../../../../common/utils/ErrorWrapper';
import { useExperimentViewLocalStore } from '../../hooks/useExperimentViewLocalStore';
import { RunsSearchTooltipContent } from './RunsSearchTooltipContent';
import {
  Entity,
  EntityNameGroup,
  getEntitiesAndIndices,
  getEntityNamesFromRunsData,
  getFilteredOptionsFromEntityName,
  getOptionsFromEntityNames,
  OptionGroup,
} from './RunsSearchAutoComplete.utils';
import { createQuickRegexpSearchFilter, detectSqlSyntaxInSearchQuery } from '../../utils/experimentPage.fetch-utils';
import { shouldUseRegexpBasedAutoRunsSearchFilter } from '../../../../../common/utils/FeatureUtils';

// A default placeholder for the search box
const SEARCH_BOX_PLACEHOLDER = 'metrics.rmse < 1 and params.model = "tree"';
const TOOLTIP_COOKIE_KEY = 'tooltipLastPopup';
const WEEK_IN_SECONDS = 604800;

export type RunsSearchAutoCompleteProps = {
  runsData: ExperimentRunsSelectorResult;
  searchFilter: string;
  onSearchFilterChange: (newValue: string) => void;
  onClear: () => void;
  requestError: ErrorWrapper | null;
};

/**
 * Autocomplete component that provides suggestions for MLflow search entity names.
 */
export const RunsSearchAutoComplete = (props: RunsSearchAutoCompleteProps) => {
  const { runsData, searchFilter, requestError, onSearchFilterChange, onClear } = props;
  const { theme, getPrefixedClassName } = useDesignSystemTheme();

  const dropdownRef = useRef<HTMLDivElement>(null);
  const intl = useIntl();

  const [text, setText] = useState<string>('');
  const [autocompleteEnabled, setAutocompleteEnabled] = useState<boolean | undefined>(undefined);
  const [focused, setFocused] = useState(false);
  const onFocus = () => setFocused(true);
  const onBlur = () => setFocused(false);
  // Determines whether the text was changed by making a selection in the autocomplete
  // dialog, as opposed to by typing.
  const [lastSetBySelection, setLastSetBySelection] = useState(false);
  const existingEntityNamesRef = useRef<EntityNameGroup>({
    metricNames: [],
    paramNames: [],
    tagNames: [],
  });
  // How many suggestions should be shown per entity group before the group is ellipsized.
  const [suggestionLimits, setSuggestionLimits] = useState({
    Metrics: 10,
    Parameters: 10,
    Tags: 10,
  });
  // List of entities parsed from `text`.
  const currentEntitiesRef = useRef<Entity[]>([]);
  const [entityBeingEdited, setEntityBeingEdited] = useState<Entity | undefined>(undefined);

  // Each time we're setting search filter externally, update it here as well
  useEffect(() => {
    setText(searchFilter);
  }, [searchFilter]);

  const baseOptions = useMemo<OptionGroup[]>(() => {
    const existingEntityNames = existingEntityNamesRef.current;
    const mergedEntityNames = getEntityNamesFromRunsData(runsData, existingEntityNames);
    existingEntityNamesRef.current = mergedEntityNames;
    return getOptionsFromEntityNames(mergedEntityNames);
    // existingEntityNamesRef is only set here. Omit from dependencies to avoid infinite loop
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runsData]);

  useEffect(() => {
    const previousEntities = currentEntitiesRef.current;
    const newEntities = getEntitiesAndIndices(text);
    currentEntitiesRef.current = newEntities;

    if (lastSetBySelection) {
      setLastSetBySelection(false);
      return;
    }
    const currentEntitiesNames = newEntities.map((e) => e.name);
    const previousEntitiesNames = previousEntities.map((e) => e.name);
    if (!isEqual(currentEntitiesNames, previousEntitiesNames) && newEntities.length >= previousEntities.length) {
      let i = 0;
      while (i < newEntities.length) {
        if (i >= previousEntities.length || newEntities[i].name.trim() !== previousEntities[i].name.trim()) {
          setAutocompleteEnabled(true);
          setEntityBeingEdited(newEntities[i]);
          return;
        }
        i++;
      }
    }
    // If here, no entity is being edited
    setAutocompleteEnabled(false);
    // currentEntitiesRef is not used anywhere else and state setters are safe to
    // omit from hook dependencies as per react docs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  const filteredOptions = useMemo(() => {
    if (!entityBeingEdited) {
      return [];
    }
    return getFilteredOptionsFromEntityName(baseOptions, entityBeingEdited, suggestionLimits);
  }, [baseOptions, entityBeingEdited, suggestionLimits]);

  /**
   * Called when an option is picked from the autocomplete dropdown, either by hitting enter
   * when selected, or clicking on it
   * @param value
   */
  const onSelect = useCallback(
    (value: string, option: any) => {
      if (!entityBeingEdited) {
        return;
      }
      if (value.startsWith('...')) {
        // Keep the dialog open as this is not a real selection
        setAutocompleteEnabled(true);
        const groupName = option.value.split('_')[1];
        setSuggestionLimits((prevSuggestionLimits) => ({
          ...prevSuggestionLimits,
          [groupName]: (prevSuggestionLimits as any)[groupName] + 10,
        }));
      } else {
        const prefix = text.substring(0, entityBeingEdited.startIndex);
        const suffix = text.substring(entityBeingEdited.endIndex);
        setText(prefix + value + ' ' + suffix);
        setLastSetBySelection(true);
        setAutocompleteEnabled(false);
      }
    },
    [text, setText, entityBeingEdited, setAutocompleteEnabled],
  );

  const localStorageInstance = useExperimentViewLocalStore(TOOLTIP_COOKIE_KEY);

  const [showTooltipOnError, setShowTooltipOnError] = useState(() => {
    const currentTimeSecs = Math.floor(Date.now() / 1000);
    const storedItem = localStorageInstance.getItem(TOOLTIP_COOKIE_KEY);
    // Show tooltip again if it was last shown 1 week ago or older
    return !storedItem || parseInt(storedItem, 10) < currentTimeSecs - WEEK_IN_SECONDS;
  });
  const tooltipIcon = React.useRef<HTMLButtonElement>(null);

  const quickRegexpFilter = useMemo(() => {
    if (shouldUseRegexpBasedAutoRunsSearchFilter() && text.length > 0 && !detectSqlSyntaxInSearchQuery(text)) {
      return createQuickRegexpSearchFilter(text);
    }
    return undefined;
  }, [text]);

  // If requestError has changed and there is an error, pop up the tooltip
  useEffect(() => {
    if (requestError && showTooltipOnError) {
      const currentTimeSecs = Math.floor(Date.now() / 1000);
      localStorageInstance.setItem(TOOLTIP_COOKIE_KEY, currentTimeSecs);
      setShowTooltipOnError(false);
      tooltipIcon.current?.click();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestError]);

  const noMatches = filteredOptions.flatMap((o) => o.options).length === 0;
  const open = autocompleteEnabled && focused && !noMatches;

  // Callback fired when key is pressed on the input
  const triggerSearch: React.KeyboardEventHandler<HTMLInputElement> = useCallback(
    (e) => {
      // Get the class name for the active item in the dropdown
      const activeItemClass = getPrefixedClassName('select-item-option-active');
      const dropdownContainsActiveItem = Boolean(dropdownRef.current?.querySelector(`.${activeItemClass}`));

      if (e.key === 'Enter') {
        // If the autocomplete dialog is open, close it
        if (open) {
          setAutocompleteEnabled(false);
        }
        // If the autocomplete dialog is closed or user didn't select any item, trigger search
        if (!open || !dropdownContainsActiveItem) {
          onSearchFilterChange(text);
        }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        if (open) {
          setAutocompleteEnabled(false);
        }
      }
    },
    [open, text, onSearchFilterChange, getPrefixedClassName],
  );

  return (
    <div
      css={{
        display: 'flex',
        gap: theme.spacing.sm,
        width: 430,
        [theme.responsive.mediaQueries.xs]: {
          width: 'auto',
        },
      }}
    >
      <AutoComplete
        dropdownMatchSelectWidth={560}
        css={{
          width: 560,
          [theme.responsive.mediaQueries.xs]: {
            width: 'auto',
          },
        }}
        defaultOpen={false}
        defaultActiveFirstOption={!shouldUseRegexpBasedAutoRunsSearchFilter()}
        open={open}
        options={filteredOptions}
        onSelect={onSelect}
        value={text}
        data-test-id="runs-search-autocomplete"
        dropdownRender={(menu) => (
          <div
            css={{
              '.du-bois-light-select-item-option-active:not(.du-bois-light-select-item-option-disabled)': {
                // TODO: ask the design team about the color existing in the palette
                backgroundColor: '#e6f1f5',
              },
            }}
            ref={dropdownRef}
          >
            {menu}
          </div>
        )}
      >
        <Input
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_runssearchautocomplete.tsx_236"
          value={text}
          prefix={
            <SearchIcon
              css={{
                svg: {
                  width: theme.general.iconFontSize,
                  height: theme.general.iconFontSize,
                  color: theme.colors.textSecondary,
                },
              }}
            />
          }
          onKeyDown={triggerSearch}
          onClick={onFocus}
          onBlur={onBlur}
          onChange={(e) => setText(e.target.value)}
          placeholder={SEARCH_BOX_PLACEHOLDER}
          data-test-id="search-box"
          suffix={
            <div css={{ display: 'flex', gap: 4, alignItems: 'center' }}>
              {text && (
                <Button
                  componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_runssearchautocomplete.tsx_212"
                  onClick={onClear}
                  type="link"
                  data-test-id="clear-button"
                >
                  <CloseIcon />
                </Button>
              )}
              {quickRegexpFilter ? (
                <Tooltip
                  content={
                    <FormattedMessage
                      defaultMessage="Using regular expression quick filter. The following query will be used: {filterSample}"
                      description="Experiment page > control bar > search filter > a label displayed when user has entered a simple query that will be automatically transformed into RLIKE SQL query before being sent to the API"
                      values={{
                        filterSample: (
                          <div>
                            <code>{quickRegexpFilter}</code>
                          </div>
                        ),
                      }}
                    />
                  }
                  delayDuration={0}
                >
                  <InfoFillIcon
                    aria-label={intl.formatMessage(
                      {
                        defaultMessage:
                          'Using regular expression quick filter. The following query will be used: {filterSample}',
                        description:
                          'Experiment page > control bar > search filter > a label displayed when user has entered a simple query that will be automatically transformed into RLIKE SQL query before being sent to the API',
                      },
                      {
                        filterSample: quickRegexpFilter,
                      },
                    )}
                    css={{
                      svg: {
                        width: theme.general.iconFontSize,
                        height: theme.general.iconFontSize,
                        color: theme.colors.actionPrimaryBackgroundDefault,
                      },
                    }}
                  />
                </Tooltip>
              ) : (
                <LegacyTooltip
                  title={<RunsSearchTooltipContent />}
                  placement="right"
                  dangerouslySetAntdProps={{
                    overlayInnerStyle: { width: '150%' },
                    trigger: ['focus', 'click'],
                  }}
                >
                  <Button
                    size="small"
                    ref={tooltipIcon}
                    componentId="mlflow.experiment_page.search_filter.tooltip"
                    type="link"
                    css={{ marginLeft: -theme.spacing.xs, marginRight: -theme.spacing.xs }}
                    icon={
                      <InfoIcon
                        css={{
                          svg: {
                            width: theme.general.iconFontSize,
                            height: theme.general.iconFontSize,
                            color: theme.colors.textSecondary,
                          },
                        }}
                      />
                    }
                  />
                </LegacyTooltip>
              )}
            </div>
          }
        />
      </AutoComplete>
    </div>
  );
};
