import { CodeIcon, GavelIcon, SparkleDoubleIcon, UserIcon } from '@databricks/design-system';

/** Judge, human or code icon for an assessment source; a gavel when the source is unknown or mixed. */
export const AssessmentSourceIcon = ({ sourceType, testIdPrefix }: { sourceType?: string; testIdPrefix: string }) => {
  const iconCss = { fontSize: 16, flexShrink: 0 };
  switch (sourceType) {
    case 'AI_JUDGE':
    case 'LLM_JUDGE':
      return <SparkleDoubleIcon aria-hidden data-testid={`${testIdPrefix}-judge-icon`} css={iconCss} />;
    case 'HUMAN':
      return <UserIcon aria-hidden data-testid={`${testIdPrefix}-human-icon`} css={iconCss} />;
    case 'CODE':
      return <CodeIcon aria-hidden data-testid={`${testIdPrefix}-code-icon`} css={iconCss} />;
    default:
      return <GavelIcon aria-hidden data-testid={`${testIdPrefix}-icon`} css={iconCss} />;
  }
};
