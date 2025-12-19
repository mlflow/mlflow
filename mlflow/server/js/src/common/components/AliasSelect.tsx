import type { Dispatch } from 'react';
import { useCallback, useState } from 'react';

import { LegacySelect, useDesignSystemTheme } from '@databricks/design-system';

import { AliasTag } from './AliasTag';
import { FormattedMessage, useIntl } from 'react-intl';

/**
 * A specialized <LegacySelect> component used for adding and removing aliases from model versions
 */
export const AliasSelect = ({
  renderKey,
  setDraftAliases,
  existingAliases,
  draftAliases,
  version,
  aliasToVersionMap,
  disabled,
}: {
  renderKey: any;
  disabled: boolean;
  setDraftAliases: Dispatch<React.SetStateAction<string[]>>;
  existingAliases: string[];
  draftAliases: string[];
  version: string;
  aliasToVersionMap: Record<string, string>;
}) => {
  const intl = useIntl();
  const [dropdownVisible, setDropdownVisible] = useState(false);

  const { theme } = useDesignSystemTheme();

  const removeFromEditedAliases = useCallback(
    (alias: string) => {
      setDraftAliases((aliases) => aliases.filter((existingAlias) => existingAlias !== alias));
    },
    [setDraftAliases],
  );

  const updateEditedAliases = useCallback(
    (aliases: string[]) => {
      const sanitizedAliases = aliases
        // Remove all characters that are not alphanumeric, underscores or hyphens
        .map((alias) =>
          alias
            .replace(/[^\w-]/g, '')
            .toLowerCase()
            .substring(0, 255),
        )
        // After sanitization, filter out invalid aliases
        // so we won't get empty values
        .filter((alias) => alias.length > 0);

      // Remove duplicates that might result from varying letter case
      const uniqueAliases = Array.from(new Set(sanitizedAliases));
      setDraftAliases(uniqueAliases);
      setDropdownVisible(false);
    },
    [setDraftAliases],
  );

  return (
    // For the time being, we will use <LegacySelect /> under the hood,
    // while <TypeaheadCombobox /> is still in the design phase.
    <LegacySelect
      disabled={disabled}
      filterOption={(val, opt) => opt?.value.toLowerCase().startsWith(val.toLowerCase())}
      placeholder={intl.formatMessage({
        defaultMessage: 'Enter aliases (champion, challenger, etc)',
        description: 'Model registry > model version alias select > Alias input placeholder',
      })}
      allowClear
      css={{ width: '100%' }}
      mode="tags"
      // There's a bug with current <LegacySelect /> implementation that causes the dropdown
      // to detach from input vertically when its position on screen changes (in this case, it's
      // caused by the conflict alerts). A small key={} hack ensures that the component is recreated
      // and the dropdown is repositioned each time the alerts below are changed.
      key={JSON.stringify(renderKey)}
      onChange={updateEditedAliases}
      dangerouslySetAntdProps={{
        dropdownMatchSelectWidth: true,
        tagRender: ({ value }) => (
          <AliasTag
            compact
            css={{ marginTop: 2 }}
            closable
            onClose={() => removeFromEditedAliases(value.toString())}
            value={value.toString()}
          />
        ),
      }}
      onDropdownVisibleChange={setDropdownVisible}
      open={dropdownVisible}
      value={draftAliases || []}
    >
      {existingAliases.map((alias) => (
        <LegacySelect.Option key={alias} value={alias} data-testid="model-alias-option">
          <div key={alias} css={{ display: 'flex', marginRight: theme.spacing.xs }}>
            <div css={{ flex: 1 }}>{alias}</div>
            <div>
              <FormattedMessage
                defaultMessage="This version"
                description="Model registry > model version alias select > Indicator for alias of selected version"
              />
            </div>
          </div>
        </LegacySelect.Option>
      ))}
      {Object.entries(aliasToVersionMap)
        .filter(([, otherVersion]) => otherVersion !== version)
        .map(([alias, aliasedVersion]) => (
          <LegacySelect.Option key={alias} value={alias} data-testid="model-alias-option">
            <div key={alias} css={{ display: 'flex', marginRight: theme.spacing.xs }}>
              <div css={{ flex: 1 }}>{alias}</div>
              <div>
                <FormattedMessage
                  defaultMessage="Version {version}"
                  description="Model registry > model version alias select > Indicator for alias of a particular version"
                  values={{ version: aliasedVersion }}
                />
              </div>
            </div>
          </LegacySelect.Option>
        ))}
    </LegacySelect>
  );
};
