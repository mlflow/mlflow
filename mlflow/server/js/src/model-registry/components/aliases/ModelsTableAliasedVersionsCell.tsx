import { first, sortBy } from 'lodash';
import type { ModelEntity } from '../../../experiment-tracking/types';
import { AliasTag } from '../../../common/components/AliasTag';
import { Button, DropdownMenu, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import { ModelRegistryRoutes } from '../../routes';
import { FormattedMessage, defineMessage } from 'react-intl';

const versionLabel = defineMessage({
  defaultMessage: 'Version {version}',
  description: 'Model registry > models table > aliases column > version indicator',
});

interface ModelsTableAliasedVersionsCellProps {
  model: ModelEntity;
}

export const ModelsTableAliasedVersionsCell = ({ model }: ModelsTableAliasedVersionsCellProps) => {
  const { aliases } = model;
  const { theme } = useDesignSystemTheme();

  if (!aliases?.length) {
    return null;
  }

  // Sort alias entries by version, descending
  const aliasesByVersionSorted = sortBy(aliases, ({ version }) => parseInt(version, 10) || 0).reverse();

  const latestVersionAlias = first(aliasesByVersionSorted);

  // Return nothing if there's not a single alias present
  if (!latestVersionAlias) {
    return null;
  }

  const otherAliases = aliasesByVersionSorted.filter((alias) => alias !== latestVersionAlias);

  return (
    <div>
      <Link to={ModelRegistryRoutes.getModelVersionPageRoute(model.name, latestVersionAlias.version)}>
        <AliasTag value={latestVersionAlias.alias} css={{ marginRight: 0, cursor: 'pointer' }} />
        : <FormattedMessage {...versionLabel} values={{ version: latestVersionAlias.version }} />
      </Link>
      {otherAliases.length > 0 && (
        <DropdownMenu.Root modal={false}>
          <DropdownMenu.Trigger asChild>
            <Button
              componentId="codegen_mlflow_app_src_model-registry_components_aliases_modelstablealiasedversionscell.tsx_47"
              size="small"
              css={{ borderRadius: 12, marginLeft: theme.spacing.xs }}
            >
              +{aliases.length - 1}
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align="start">
            {otherAliases.map(({ alias, version }) => (
              <DropdownMenu.Item
                componentId="codegen_mlflow_app_src_model-registry_components_aliases_modelstablealiasedversionscell.tsx_57"
                key={alias}
              >
                <Link to={ModelRegistryRoutes.getModelVersionPageRoute(model.name, version)}>
                  <AliasTag value={alias} css={{ marginRight: 0, cursor: 'pointer' }} />:{' '}
                  <span css={{ color: theme.colors.actionTertiaryTextDefault }}>
                    <FormattedMessage {...versionLabel} values={{ version }} />
                  </span>
                </Link>
              </DropdownMenu.Item>
            ))}
          </DropdownMenu.Content>
        </DropdownMenu.Root>
      )}
    </div>
  );
};
