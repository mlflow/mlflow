import { useState } from 'react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  Spacer,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useQuery } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { RegisteredPromptsApi } from '../../prompts/api';
import { getPromptContentTagValue } from '../../prompts/utils';
import type { RegisteredPromptVersion } from '../../prompts/types';
import { PROMPT_VERSION_QUERY_PARAM } from '../../prompts/utils';

interface ExpandablePromptSectionProps {
  promptUri: string;
  experimentId: string;
  title: React.ReactNode;
}

/**
 * Parse a prompt URI in the format "prompts:/name/version" or "prompts:/name/version@alias"
 */
const parsePromptUri = (uri: string): { name: string; version: string } | null => {
  // Format: prompts:/name/version or prompts:/name/version@alias
  const match = uri.match(/^prompts:\/([^/]+)\/(\d+)(?:@.*)?$/);
  if (match) {
    return { name: match[1], version: match[2] };
  }
  return null;
};

/**
 * Expandable section that displays a prompt with a link to the prompt details page.
 * When expanded, fetches and displays the prompt content.
 */
export const ExpandablePromptSection = ({ promptUri, experimentId, title }: ExpandablePromptSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const [isExpanded, setIsExpanded] = useState(false);

  const parsedUri = parsePromptUri(promptUri);

  // Fetch prompt versions when expanded
  const { data: versionsData, isLoading } = useQuery<{ model_versions?: RegisteredPromptVersion[] }, Error>(
    ['prompt_version_for_optimization', parsedUri?.name, parsedUri?.version],
    {
      queryFn: () => RegisteredPromptsApi.getPromptVersions(parsedUri!.name),
      enabled: isExpanded && !!parsedUri,
      retry: false,
    },
  );

  // Find the specific version
  const promptVersion = versionsData?.model_versions?.find((v) => v.version === parsedUri?.version);
  const promptContent = promptVersion ? getPromptContentTagValue(promptVersion) : undefined;

  // Build link to prompt details page with version selected
  const buildPromptLink = () => {
    if (!parsedUri) return null;

    const basePath = Routes.getPromptDetailsPageRoute(encodeURIComponent(parsedUri.name), experimentId);
    const searchParams = new URLSearchParams();
    searchParams.set(PROMPT_VERSION_QUERY_PARAM, parsedUri.version);

    return `${basePath}?${searchParams.toString()}`;
  };

  const promptLink = buildPromptLink();

  return (
    <div
      css={{
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.general.borderRadiusBase,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        css={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.sm,
          background: theme.colors.backgroundSecondary,
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
          '&:hover': {
            background: theme.colors.actionTertiaryBackgroundHover,
          },
        }}
      >
        {isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        <Typography.Text bold>{title}</Typography.Text>
        {promptLink && (
          <Link
            to={promptLink}
            onClick={(e) => e.stopPropagation()}
            css={{
              marginLeft: 'auto',
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            {promptUri}
          </Link>
        )}
        {!promptLink && (
          <Typography.Text
            color="secondary"
            css={{
              marginLeft: 'auto',
              fontFamily: 'monospace',
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            {promptUri}
          </Typography.Text>
        )}
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div css={{ padding: theme.spacing.md }}>
          {isLoading ? (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Spinner size="small" />
              <Typography.Text color="secondary">
                <FormattedMessage defaultMessage="Loading prompt..." description="Loading prompt message" />
              </Typography.Text>
            </div>
          ) : promptContent ? (
            <pre
              css={{
                background: theme.colors.backgroundSecondary,
                padding: theme.spacing.md,
                borderRadius: theme.general.borderRadiusBase,
                overflow: 'auto',
                maxHeight: 400,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
                fontSize: theme.typography.fontSizeSm,
                fontFamily: 'monospace',
                margin: 0,
              }}
            >
              {promptContent}
            </pre>
          ) : (
            <Typography.Text color="secondary">
              <FormattedMessage
                defaultMessage="Prompt content not available. The prompt may have been deleted or the URI format is not recognized."
                description="No prompt content message"
              />
            </Typography.Text>
          )}
        </div>
      )}
    </div>
  );
};
