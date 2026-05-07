import { Empty, Header, PlayIcon, Spacer, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ScrollablePageWrapper } from '../../../common/components/ScrollablePageWrapper';
import ErrorUtils from '../../../common/utils/ErrorUtils';
import { withErrorBoundary } from '../../../common/utils/withErrorBoundary';

const PlaygroundPage = () => {
  const { theme } = useDesignSystemTheme();

  return (
    <ScrollablePageWrapper css={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
      <Spacer shrinks={false} />
      <Header
        title={
          <span css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
            <span
              css={{
                display: 'flex',
                borderRadius: theme.borders.borderRadiusSm,
                backgroundColor: theme.colors.backgroundSecondary,
                padding: theme.spacing.sm,
              }}
            >
              <PlayIcon />
            </span>
            <FormattedMessage defaultMessage="Playground" description="Title of the LLM playground page" />
          </span>
        }
      />
      <Spacer shrinks={false} />
      <Empty
        title={
          <FormattedMessage
            defaultMessage="Playground coming soon"
            description="Title shown on the Playground page placeholder before its features are wired up"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Soon you'll be able to test AI Gateway endpoints and registered prompts here."
            description="Placeholder description shown on the Playground page before its features are wired up"
          />
        }
      />
    </ScrollablePageWrapper>
  );
};

export default withErrorBoundary(ErrorUtils.mlflowServices.EXPERIMENTS, PlaygroundPage);
