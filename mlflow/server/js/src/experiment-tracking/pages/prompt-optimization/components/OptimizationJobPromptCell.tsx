import { useMemo } from 'react';
import { Link, useParams } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';

interface OptimizationJobPromptCellProps {
  promptUri?: string;
}

/**
 * Parses a prompt URI like "models:/prompt-name/1" or "prompts:/prompt-name/1"
 * and returns the prompt name and version.
 */
const parsePromptUri = (uri: string): { promptName: string; version: string } | null => {
  // Match patterns like "models:/prompt-name/1" or "prompts:/prompt-name/1"
  const match = uri.match(/^(?:models|prompts):\/([^/]+)\/(\d+)$/);
  if (match) {
    return { promptName: match[1], version: match[2] };
  }
  return null;
};

export const OptimizationJobPromptCell = ({ promptUri }: OptimizationJobPromptCellProps) => {
  const { experimentId } = useParams<{ experimentId: string }>();

  const parsed = useMemo(() => {
    if (!promptUri) return null;
    return parsePromptUri(promptUri);
  }, [promptUri]);

  if (!promptUri) {
    return <span>-</span>;
  }

  if (!parsed) {
    // If we can't parse it, just show the raw URI
    return <span>{promptUri}</span>;
  }

  const displayText = `${parsed.promptName}/${parsed.version}`;

  if (!experimentId) {
    return <span>{displayText}</span>;
  }

  // Stop propagation to prevent row click when clicking the link
  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
  };

  // Link to the prompt details page
  const promptDetailsRoute = Routes.getPromptDetailsPageRoute(parsed.promptName, experimentId);

  return (
    <Link to={promptDetailsRoute} onClick={handleClick}>
      {displayText}
    </Link>
  );
};
