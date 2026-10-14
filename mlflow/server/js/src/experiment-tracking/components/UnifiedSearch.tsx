import type { ReactNode, RefObject } from 'react';
import { useLayoutEffect, useRef, useState } from 'react';
import { useIntl } from 'react-intl';
import {
  AutoComplete,
  Button,
  CloseIcon,
  FilterIcon,
  getDarkModePortalStyles,
  Input,
  type InputRef,
  SearchIcon,
  SparkleIcon,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';

export interface UnifiedSearchOption {
  id: string;
  label: ReactNode;
  accessibleLabel: string;
  textSearch?: boolean;
  aiSearch?: boolean;
  select: () => void;
}

export interface UnifiedSearchChip {
  text: string;
  label: ReactNode;
  onEdit: () => void;
  onRemove: () => void;
}

interface UnifiedSearchProps {
  componentId: string;
  value: string;
  open: boolean;
  setOpen: (open: boolean) => void;
  isAIMode: boolean;
  placeholder: string;
  ariaLabel: string;
  submitDisabled: boolean;
  onSubmit: () => void;
  onChange: (value: string) => void;
  onClear: () => void;
  onOpenFilter: () => void;
  onFocusChange?: (focused: boolean) => void;
  onInputClick?: () => void;
  recentOptions: UnifiedSearchOption[];
  suggestions: UnifiedSearchOption[];
  chips: UnifiedSearchChip[];
  suffix?: ReactNode;
  inputRef?: RefObject<InputRef>;
  pillRailRef?: RefObject<HTMLDivElement>;
  alignDropdownToContainer?: boolean;
  showEmptySuggestionsHeader?: boolean;
}

/** Search chrome shared by traces and evaluation runs; query semantics stay with each caller. */
export const UnifiedSearch = ({
  componentId,
  value,
  open,
  setOpen,
  isAIMode,
  placeholder,
  ariaLabel,
  submitDisabled,
  onSubmit,
  onChange,
  onClear,
  onOpenFilter,
  onFocusChange,
  onInputClick,
  recentOptions,
  suggestions,
  chips,
  suffix,
  inputRef,
  pillRailRef,
  alignDropdownToContainer,
  showEmptySuggestionsHeader,
}: UnifiedSearchProps) => {
  const intl = useIntl();
  const { theme, classNamePrefix } = useDesignSystemTheme();
  const searchBarRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [dropdownWidth, setDropdownWidth] = useState(0);
  const [dropdownOffset, setDropdownOffset] = useState(0);
  useLayoutEffect(() => {
    if (!alignDropdownToContainer) return;
    const bar = searchBarRef.current;
    const trigger = bar?.lastElementChild;
    if (!bar || !trigger) return;
    const measure = () => {
      const barBounds = bar.getBoundingClientRect();
      setDropdownWidth(barBounds.width);
      setDropdownOffset(barBounds.left - trigger.getBoundingClientRect().left);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(bar);
    observer.observe(trigger);
    return () => observer.disconnect();
  }, [alignDropdownToContainer]);
  const visibleOptions = [...recentOptions, ...suggestions];
  const suggestionsHeader = intl.formatMessage({
    defaultMessage: 'Suggestions',
    description: 'Search suggestions section',
  });
  const optionProps = (option: UnifiedSearchOption) => ({
    value: option.id,
    'aria-label': option.accessibleLabel,
    label: (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        {option.aiSearch ? <SparkleIcon color="ai" /> : option.textSearch ? <SearchIcon /> : <FilterIcon />}
        <span css={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{option.label}</span>
      </div>
    ),
  });

  const renderDropdown = (menu: ReactNode) => {
    if (!open) return <></>;
    return (
      <div
        ref={dropdownRef}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false);
        }}
      >
        {showEmptySuggestionsHeader && !suggestions.length && (
          <div css={{ padding: theme.spacing.sm, color: theme.colors.textSecondary }}>{suggestionsHeader}</div>
        )}
        <div hidden={!visibleOptions.length}>{menu}</div>
        <div
          css={{ borderTop: `1px solid ${theme.colors.actionDefaultBorderDefault}`, padding: theme.spacing.xs }}
          onMouseDown={(event) => event.preventDefault()}
        >
          <Button
            componentId="mlflow.unified-search.add-filter"
            type="tertiary"
            icon={<FilterIcon />}
            onClick={onOpenFilter}
            css={{
              display: 'flex',
              justifyContent: 'flex-start',
              textAlign: 'left',
              '&:focus:not(:focus-visible)': { outline: 'none', boxShadow: 'none' },
            }}
          >
            {intl.formatMessage({ defaultMessage: 'Add a Filter', description: 'Open the search filter editor' })}
          </Button>
        </div>
      </div>
    );
  };

  return (
    <div
      ref={searchBarRef}
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: 4,
        width: '100%',
        minWidth: 0,
        height: theme.general.heightSm,
        boxSizing: 'border-box',
        padding: '0 4px',
        border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
        borderRadius: theme.borders.borderRadiusMd,
        '&:focus-within': { borderColor: theme.colors.actionDefaultBorderFocus },
      }}
    >
      <Button
        componentId="mlflow.unified-search.submit"
        type="link"
        size="small"
        icon={<SearchIcon />}
        disabled={submitDisabled}
        onClick={onSubmit}
        aria-label={intl.formatMessage({ defaultMessage: 'Search', description: 'Submit search' })}
        css={{ flex: '0 0 24px', width: 24, padding: 0 }}
      />
      <div
        ref={pillRailRef}
        css={{
          display: chips.length ? 'flex' : 'none',
          flex: '0 1 auto',
          alignItems: 'center',
          gap: 4,
          minWidth: 0,
          overflowX: 'auto',
          overflowY: 'hidden',
          scrollbarWidth: 'none',
          '&::-webkit-scrollbar': { display: 'none' },
        }}
      >
        {chips.map((chip, index) => (
          <span
            key={index}
            css={{
              display: 'inline-flex',
              alignItems: 'center',
              flex: '0 0 auto',
              maxWidth: 200,
              height: 24,
              boxSizing: 'border-box',
              minWidth: 0,
              overflow: 'hidden',
              backgroundColor: theme.colors.backgroundSecondary,
              borderRadius: theme.borders.borderRadiusMd,
            }}
          >
            <Tooltip componentId="mlflow.unified-search.edit-filter-tooltip" content={chip.text} delayDuration={0}>
              <Button
                componentId="mlflow.unified-search.edit-filter"
                type="tertiary"
                size="small"
                title={chip.text}
                aria-label={intl.formatMessage(
                  { defaultMessage: 'Edit filter {filter}', description: 'Edit an applied search filter' },
                  { filter: chip.text },
                )}
                onClick={chip.onEdit}
                css={{
                  flex: '1 1 0%',
                  minWidth: 0,
                  overflow: 'hidden',
                  '&&': { height: 22, fontSize: 11, padding: '0 4px', lineHeight: '20px' },
                  '& code': {
                    display: 'block',
                    fontSize: 'inherit',
                    maxWidth: '100%',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  },
                  '& > span': { minWidth: 0, overflow: 'hidden' },
                }}
              >
                {chip.label}
              </Button>
            </Tooltip>
            <Button
              componentId="mlflow.unified-search.remove-filter"
              type="tertiary"
              size="small"
              icon={<CloseIcon />}
              aria-label={intl.formatMessage(
                { defaultMessage: 'Remove filter {filter}', description: 'Remove an applied search filter' },
                { filter: chip.text },
              )}
              onClick={chip.onRemove}
              css={{
                flex: '0 0 22px',
                zIndex: 1,
                backgroundColor: theme.colors.backgroundSecondary,
                '&&': { width: 22, minWidth: 22, height: 22, padding: 0, fontSize: 12 },
              }}
            />
          </span>
        ))}
      </div>
      <AutoComplete
        size="small"
        css={{
          flex: '1 1 180px',
          minWidth: 140,
          [`&&& .${classNamePrefix}-select-selector, &&& .${classNamePrefix}-select-focused .${classNamePrefix}-select-selector, &&& .${classNamePrefix}-select-selection-search`]:
            {
              border: '0 !important',
              boxShadow: 'none !important',
              outline: 'none !important',
            },
          [`&&& .db-input, &&& .${classNamePrefix}-input, &&& .${classNamePrefix}-input-affix-wrapper, &&& .ant-input, &&& .ant-input-affix-wrapper`]:
            {
              border: 0,
              backgroundColor: 'transparent',
            },
          [`&&& .db-input:hover, &&& .db-input:focus, &&& .db-input:focus-within, &&& .db-input:focus-visible, &&& .${classNamePrefix}-input:hover, &&& .${classNamePrefix}-input:focus, &&& .${classNamePrefix}-input:focus-within, &&& .${classNamePrefix}-input:focus-visible, &&& .${classNamePrefix}-input-affix-wrapper:hover, &&& .${classNamePrefix}-input-affix-wrapper:focus, &&& .${classNamePrefix}-input-affix-wrapper:focus-within, &&& .${classNamePrefix}-input-affix-wrapper-focused, &&& .ant-input:hover, &&& .ant-input:focus, &&& .ant-input-affix-wrapper:hover, &&& .ant-input-affix-wrapper-focused`]:
            {
              border: '0 !important',
              borderColor: 'transparent !important',
              boxShadow: 'none !important',
              outline: 'none !important',
              outlineColor: 'transparent !important',
              outlineStyle: 'none !important',
              outlineWidth: '0 !important',
              backgroundColor: 'transparent',
            },
          '& input::-webkit-search-cancel-button': { WebkitAppearance: 'none', display: 'none' },
        }}
        value={value}
        open={open && !isAIMode}
        defaultActiveFirstOption={false}
        virtual={false}
        dropdownMatchSelectWidth={alignDropdownToContainer ? dropdownWidth || true : 560}
        dropdownAlign={alignDropdownToContainer ? { offset: [dropdownOffset, 4] } : undefined}
        dropdownStyle={{
          boxShadow: theme.general.shadowLow,
          ...getDarkModePortalStyles(theme),
          zIndex: theme.options.zIndexBase + 20,
          minWidth: 0,
          maxWidth: 'calc(100vw - 32px)',
        }}
        options={[
          ...(recentOptions.length
            ? [
                {
                  label: intl.formatMessage({ defaultMessage: 'Recent', description: 'Recent searches section' }),
                  options: recentOptions.map(optionProps),
                },
              ]
            : []),
          ...(suggestions.length ? [{ label: suggestionsHeader, options: suggestions.map(optionProps) }] : []),
          ...(!visibleOptions.length ? [{ value: 'empty', disabled: true, style: { display: 'none' } }] : []),
        ]}
        dropdownRender={renderDropdown}
        onSearch={(next) => {
          onChange(next);
          setOpen(!isAIMode);
        }}
        onSelect={(id) => visibleOptions.find((option) => option.id === id)?.select()}
        onFocus={() => {
          onFocusChange?.(true);
          setOpen(!isAIMode);
        }}
        onBlur={(event) => {
          onFocusChange?.(false);
          if (!dropdownRef.current?.contains(event.relatedTarget as Node)) setOpen(false);
        }}
      >
        <Input
          ref={inputRef}
          size="small"
          componentId={componentId}
          placeholder={placeholder}
          aria-label={ariaLabel}
          allowClear
          onClear={onClear}
          onClick={(event) => {
            if (event.target instanceof HTMLInputElement) onInputClick?.();
          }}
          onKeyDownCapture={(event) => {
            if (event.key === 'Tab' && !event.shiftKey && open && event.target instanceof HTMLInputElement) {
              const footer = dropdownRef.current?.querySelector<HTMLButtonElement>(
                '[data-component-id="mlflow.unified-search.add-filter"]',
              );
              if (footer) {
                event.preventDefault();
                footer.focus();
                return;
              }
            }
            if (event.key === 'Escape') setOpen(false);
            if (event.key !== 'Enter' || event.nativeEvent.isComposing) return;
            const activeId =
              event.target instanceof HTMLInputElement ? event.target.getAttribute('aria-activedescendant') : null;
            if (open && activeId && document.getElementById(activeId)) return;
            event.preventDefault();
            event.stopPropagation();
            onSubmit();
          }}
          suffix={suffix}
        />
      </AutoComplete>
    </div>
  );
};
