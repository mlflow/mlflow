import { first, sortBy } from 'lodash';
import { ModelEntity } from '../../../experiment-tracking/types';
import { ModelVersionAliasTag } from './ModelVersionAliasTag';
import { Button, DropdownMenu, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import { getModelVersionPageRoute } from '../../routes';
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
  const aliasesByVersionSorted = sortBy(
    aliases,
    ({ version }) => parseInt(version, 10) || 0,
  ).reverse();

  const latestVersionAlias = first(aliasesByVersionSorted);

  // Return nothing if there's not a single alias present
  if (!latestVersionAlias) {
    return null;
  }

  const otherAliases = aliasesByVersionSorted.filter((alias) => alias !== latestVersionAlias);

  return (
    <div>
      <ModelVersionAliasTag value={latestVersionAlias.alias} css={{ marginRight: 0 }} />:{' '}
      <Link to={getModelVersionPageRoute(model.name, latestVersionAlias.version)}>
        <FormattedMessage {...versionLabel} values={{ version: latestVersionAlias.version }} />
      </Link>
      {otherAliases.length > 0 && (
        <DropdownMenu.Root modal={false}>
          <DropdownMenu.Trigger asChild>
            <Button size='small' css={{ borderRadius: 12, marginLeft: theme.spacing.xs }}>
              +{aliases.length - 1}
            </Button>
          </DropdownMenu.Trigger>
          <DropdownMenu.Content align='start'>
            {otherAliases.map(({ alias, version }) => (
              <DropdownMenu.Item key={alias}>
                <Link to={getModelVersionPageRoute(model.name, version)}>
                  <ModelVersionAliasTag value={alias} css={{ marginRight: 0 }} />:{' '}
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
