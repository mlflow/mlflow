import { sortedIndexOf } from 'lodash';
import React, { useMemo, useRef, useState } from 'react';
import type { Control } from 'react-hook-form';
import { useController } from 'react-hook-form';
import { useIntl } from 'react-intl';

import { PlusIcon, LegacySelect, LegacyTooltip, useDesignSystemTheme } from '@databricks/design-system';
import type { KeyValueEntity } from '../types';

/**
 * Will show an extra row at the bottom of the dropdown menu to create a new tag when
 * The user has typed something in the search input
 * and either
 * 1. The search input is not an exact match for an existing tag name
 * 2. There are no tags available based on search input
 */

function DropdownMenu(menu: React.ReactElement, allAvailableTags: string[]) {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();
  const searchValue = menu.props.searchValue.toLowerCase();

  const resolvedMenu = useMemo(() => {
    if (!searchValue) return menu;

    const doesTagExists = sortedIndexOf(allAvailableTags, searchValue) >= 0;
    if (doesTagExists) return menu;

    const isValidTagKey = /^[^,.:/=\-\s]+$/.test(searchValue);

    // Overriding the menu to add a new option at the top
    return React.cloneElement(menu, {
      flattenOptions: [
        {
          data: {
            value: searchValue,
            disabled: !isValidTagKey,
            style: {
              color: isValidTagKey ? theme.colors.actionTertiaryTextDefault : theme.colors.actionDisabledText,
            },
            children: (
              <LegacyTooltip
                title={
                  isValidTagKey
                    ? undefined
                    : intl.formatMessage({
                        defaultMessage: ', . : / - = and blank spaces are not allowed',
                        description:
                          'Key-value tag editor modal > Tag dropdown Manage Modal > Invalid characters error',
                      })
                }
                placement="right"
              >
                <span css={{ display: 'block' }}>
                  <PlusIcon css={{ marginRight: theme.spacing.sm }} />
                  {intl.formatMessage(
                    {
                      defaultMessage: 'Add tag "{tagKey}"',
                      description: 'Key-value tag editor modal > Tag dropdown Manage Modal > Add new tag button',
                    },
                    {
                      tagKey: searchValue,
                    },
                  )}
                </span>
              </LegacyTooltip>
            ),
          },
          key: searchValue,
          groupOption: false,
        },
        ...menu.props.flattenOptions,
      ],
    });
  }, [allAvailableTags, menu, searchValue, intl, theme]);

  return resolvedMenu;
}

function getDropdownMenu(allAvailableTags: string[]) {
  return (menu: React.ReactElement) => DropdownMenu(menu, allAvailableTags);
}

/**
 * Used in tag edit feature, allows selecting existing / adding new tag value
 */
export function TagKeySelectDropdown({
  allAvailableTags,
  control,
  onKeyChangeCallback,
}: {
  allAvailableTags: string[];
  control: Control<KeyValueEntity>;
  onKeyChangeCallback?: (key?: string) => void;
}) {
  const intl = useIntl();
  const [isOpen, setIsOpen] = useState(false);
  const selectRef = useRef<{ blur: () => void; focus: () => void }>(null);

  const { field, fieldState } = useController({
    control: control,
    name: 'key',
    rules: {
      required: {
        message: intl.formatMessage({
          defaultMessage: 'A tag key is required',
          description: 'Key-value tag editor modal > Tag dropdown > Tag key required error message',
        }),
        value: true,
      },
    },
  });

  const handleDropdownVisibleChange = (visible: boolean) => {
    setIsOpen(visible);
  };

  const handleClear = () => {
    field.onChange(undefined);
    onKeyChangeCallback?.(undefined);
  };

  const handleSelect = (key: string) => {
    field.onChange(key);
    onKeyChangeCallback?.(key);
  };

  return (
    <LegacySelect
      allowClear
      ref={selectRef}
      dangerouslySetAntdProps={{
        showSearch: true,
        dropdownRender: getDropdownMenu(allAvailableTags),
      }}
      css={{ width: '100%' }}
      placeholder={intl.formatMessage({
        defaultMessage: 'Type a key',
        description: 'Key-value tag editor modal > Tag dropdown > Tag input placeholder',
      })}
      value={field.value}
      defaultValue={field.value}
      open={isOpen}
      onDropdownVisibleChange={handleDropdownVisibleChange}
      filterOption={(input, option) => option?.value.toLowerCase().includes(input.toLowerCase())}
      onSelect={handleSelect}
      onClear={handleClear}
      validationState={fieldState.error ? 'error' : undefined}
    >
      {allAvailableTags.map((tag) => (
        <LegacySelect.Option value={tag} key={tag}>
          {tag}
        </LegacySelect.Option>
      ))}
    </LegacySelect>
  );
}
