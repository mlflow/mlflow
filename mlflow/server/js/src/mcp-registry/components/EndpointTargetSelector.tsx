import { SimpleSelect, SimpleSelectOption, SimpleSelectOptionGroup } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

const ALIAS_PREFIX = 'alias:';
const VERSION_PREFIX = 'version:';

export { ALIAS_PREFIX, VERSION_PREFIX };

export const EndpointTargetSelector = ({
  value,
  onChange,
  disabled,
  scopedVersion,
  scopedAliases,
  aliases,
  versions,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled: boolean;
  scopedVersion?: string;
  scopedAliases?: string[];
  aliases: { alias: string }[];
  versions: { version: string }[];
}) => {
  const intl = useIntl();

  return (
    <SimpleSelect
      id="mcp-registry-endpoint-target"
      componentId="mlflow.mcp_registry.endpoint_modal.target"
      value={value}
      onChange={({ target }) => onChange(target.value)}
      disabled={disabled}
    >
      {scopedVersion ? (
        <>
          <SimpleSelectOptionGroup
            label={intl.formatMessage({
              defaultMessage: 'Version',
              description: 'MCP registry endpoint modal version group label',
            })}
          >
            <SimpleSelectOption value={`${VERSION_PREFIX}${scopedVersion}`}>{scopedVersion}</SimpleSelectOption>
          </SimpleSelectOptionGroup>
          {scopedAliases && scopedAliases.length > 0 && (
            <SimpleSelectOptionGroup
              label={intl.formatMessage({
                defaultMessage: 'Aliases',
                description: 'MCP registry endpoint modal aliases group label',
              })}
            >
              {scopedAliases.map((alias) => (
                <SimpleSelectOption key={alias} value={`${ALIAS_PREFIX}${alias}`}>
                  @{alias}
                </SimpleSelectOption>
              ))}
            </SimpleSelectOptionGroup>
          )}
        </>
      ) : (
        <>
          <SimpleSelectOptionGroup
            label={intl.formatMessage({
              defaultMessage: 'Aliases',
              description: 'MCP registry endpoint modal aliases group label',
            })}
          >
            <SimpleSelectOption value={`${ALIAS_PREFIX}latest`}>
              <FormattedMessage defaultMessage="@latest" description="MCP registry latest alias option" />
            </SimpleSelectOption>
            {aliases.map((a) => (
              <SimpleSelectOption key={a.alias} value={`${ALIAS_PREFIX}${a.alias}`}>
                @{a.alias}
              </SimpleSelectOption>
            ))}
          </SimpleSelectOptionGroup>
          {versions.length > 0 && (
            <SimpleSelectOptionGroup
              label={intl.formatMessage({
                defaultMessage: 'Versions',
                description: 'MCP registry endpoint modal versions group label',
              })}
            >
              {versions.map((v) => (
                <SimpleSelectOption key={v.version} value={`${VERSION_PREFIX}${v.version}`}>
                  {v.version}
                </SimpleSelectOption>
              ))}
            </SimpleSelectOptionGroup>
          )}
        </>
      )}
    </SimpleSelect>
  );
};
