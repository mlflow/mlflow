import { FormattedMessage } from 'react-intl';
import { Link } from '../../../common/utils/RoutingUtils';
import { useState } from 'react';
import {
  Button,
  ChevronDoubleDownIcon,
  ChevronDoubleUpIcon,
  LegacyTooltip,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { ModelRegistryRoutes } from '../../routes';
import type { KeyValueEntity } from '../../../common/types';
import { MLFLOW_INTERNAL_PREFIX } from '../../../common/utils/TagUtils';

const EmptyCell = () => <>&mdash;</>;

export const ModelListTagsCell = ({ tags }: { tags: KeyValueEntity[] }) => {
  const tagsToShowInitially = 3;
  const { theme } = useDesignSystemTheme();
  const [showMore, setShowMore] = useState(false);

  const validTags = tags?.filter((tag) => !tag.key.startsWith(MLFLOW_INTERNAL_PREFIX));

  const tagsToDisplay = validTags?.slice(0, showMore ? undefined : tagsToShowInitially);

  if (!validTags?.length) {
    return <EmptyCell />;
  }

  const noValue = (
    <em>
      <FormattedMessage description="Models table > tags column > no value" defaultMessage="(empty)" />
    </em>
  );

  return (
    <div>
      {tagsToDisplay.map((tag) => (
        <LegacyTooltip
          key={tag.key}
          title={
            <>
              {tag.key}: {tag.value || noValue}
            </>
          }
          placement="left"
        >
          <div
            key={tag.key}
            css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}
            data-testid="models-table-tag-entry"
          >
            <Typography.Text bold>{tag.key}</Typography.Text>: {tag.value || noValue}
          </div>
        </LegacyTooltip>
      ))}
      {tags.length > tagsToShowInitially && (
        <Button
          componentId="codegen_mlflow_app_src_model-registry_components_model-list_modeltablecellrenderers.tsx_65"
          css={{ marginTop: theme.spacing.sm }}
          size="small"
          onClick={() => setShowMore(!showMore)}
          icon={showMore ? <ChevronDoubleUpIcon /> : <ChevronDoubleDownIcon />}
          data-testid="models-table-show-more-tags"
        >
          {showMore ? (
            <FormattedMessage
              defaultMessage="Show less"
              description="Models table > tags column > show less toggle button"
            />
          ) : (
            <FormattedMessage
              defaultMessage="{value} more"
              description="Models table > tags column > show more toggle button"
              values={{ value: validTags.length - tagsToShowInitially }}
            />
          )}
        </Button>
      )}
    </div>
  );
};

/**
 * Renders model version with the link in the models table
 */
export const ModelListVersionLinkCell = ({ versionNumber, name }: { versionNumber?: string; name: string }) => {
  if (!versionNumber) {
    return <EmptyCell />;
  }
  return (
    <FormattedMessage
      defaultMessage="<link>Version {versionNumber}</link>"
      description="Row entry for version columns in the registered model page"
      values={{
        versionNumber,
        link: (text: any) => <Link to={ModelRegistryRoutes.getModelVersionPageRoute(name, versionNumber)}>{text}</Link>,
      }}
    />
  );
};
