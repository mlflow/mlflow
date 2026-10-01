import { Breadcrumb, Header, Spacer, PuzzleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { ScrollablePageWrapper } from '../../common/components/ScrollablePageWrapper';
import { Link, useParams } from '../../common/utils/RoutingUtils';
import { withErrorBoundary } from '../../common/utils/withErrorBoundary';
import ErrorUtils from '../../common/utils/ErrorUtils';
import SkillRegistryRoutes from '../routes';
import { formatSkillIdentity, parseSkillRouteParams } from '../utils';
import { headerIconStyles } from '../styles';

const SkillDetailPage = () => {
  const { theme } = useDesignSystemTheme();
  const params = useParams<{ organization?: string; skillName?: string }>();
  const { name, organization } = parseSkillRouteParams(params);
  const identity = formatSkillIdentity(name, organization);

  return (
    <ScrollablePageWrapper css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <Spacer shrinks={false} />
      <Header
        breadcrumbs={
          <Breadcrumb>
            <Breadcrumb.Item>
              <Link
                componentId="mlflow.skill_registry.detail.breadcrumb_back"
                to={SkillRegistryRoutes.skillRegistryPageRoute}
              >
                <FormattedMessage
                  defaultMessage="Skills"
                  description="Breadcrumb link back to the Skill Registry catalog"
                />
              </Link>
            </Breadcrumb.Item>
          </Breadcrumb>
        }
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span css={headerIconStyles(theme)}>
              <PuzzleIcon />
            </span>
            {identity}
          </span>
        }
      />
      <Spacer shrinks={false} />
      <Typography.Hint>
        <FormattedMessage
          defaultMessage="Skill details will appear here."
          description="Placeholder copy on the Skill detail route boundary"
        />
      </Typography.Hint>
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.SKILL_REGISTRY, SkillDetailPage);
