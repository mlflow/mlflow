import { useState } from 'react';
import ShowArtifactPage from './artifact-view-components/ShowArtifactPage';
import type { RunInfoEntity } from '../types';
import { useRunsArtifacts } from './experiment-page/hooks/useRunsArtifacts';
import { getCommonArtifacts } from './experiment-page/utils/getCommonArtifacts';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ArtifactViewTree } from './ArtifactViewTree';
import { getBasename } from '../../common/utils/FileUtils';

/**
 * Minimum height for the artifact comparison container.
 * Ensures embedded viewers (HTML iframes, PDF viewers) have enough
 * vertical space to render meaningful content.
 */
const ARTIFACT_VIEW_MIN_HEIGHT = 320;

/**
 * Maximum height for the artifact comparison container expressed as a
 * viewport-height fraction.  Using 70vh instead of 100vh prevents the
 * artifact panel from overflowing the collapsible section that wraps it
 * in the compare-runs page.
 */
const ARTIFACT_VIEW_MAX_HEIGHT = '70vh';

export const CompareRunArtifactView = ({
  runUuids,
  runInfos,
}: {
  runUuids: string[];
  runInfos: RunInfoEntity[];
}) => {
  const { theme } = useDesignSystemTheme();
  const [artifactPath, setArtifactPath] = useState<string | undefined>();

  const { artifactsKeyedByRun } = useRunsArtifacts(runUuids);
  const commonArtifacts = getCommonArtifacts(artifactsKeyedByRun);

  if (commonArtifacts.length === 0) {
    return (
      <h2>
        <FormattedMessage
          defaultMessage="No common artifacts to display."
          description="Text shown when there are no common artifacts between the runs"
        />
      </h2>
    );
  }
  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        minHeight: ARTIFACT_VIEW_MIN_HEIGHT,
        maxHeight: ARTIFACT_VIEW_MAX_HEIGHT,
        height: ARTIFACT_VIEW_MAX_HEIGHT,
        border: `1px solid ${theme.colors.borderDecorative}`,
        borderRadius: theme.borders.borderRadiusMd,
        overflow: 'hidden',
      }}
    >
      {/* ── Tree sidebar: intrinsic width, full height ── */}
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          color: theme.colors.textPrimary,
          flex: '0 0 auto',
          whiteSpace: 'nowrap',
          borderRight: `1px solid ${theme.colors.borderDecorative}`,
          overflowY: 'auto',
        }}
      >
        <ArtifactViewTree
          data={commonArtifacts.map((path: string) => ({
            id: path,
            active: artifactPath === path,
            name: getBasename(path),
          }))}
          onToggleTreebeard={({ id }) => setArtifactPath(id)}
        />
      </div>

      {/* ── Artifact content area: fills remaining space ── */}
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minWidth: 0,
          overflow: 'hidden',
        }}
      >
        {/* Run‑identifier header row */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            borderBottom: `1px solid ${theme.colors.borderDecorative}`,
            backgroundColor: theme.colors.backgroundSecondary,
          }}
        >
          {runUuids.map((runUuid, index) => (
            <div
              key={`header-${runUuid}`}
              css={{
                flex: '1 1 0%',
                minWidth: 0,
                padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`,
                borderLeft: index > 0 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                fontWeight: theme.typography.typographyBoldFontWeight,
                fontSize: theme.typography.fontSizeSm,
                color: theme.colors.textSecondary,
              }}
              title={runUuid}
            >
              <FormattedMessage
                defaultMessage="Run {index}"
                description="Header label for a run column in the artifact comparison view"
                values={{ index: index + 1 }}
              />
              {': '}
              <span css={{ fontWeight: 'normal' }}>{runUuid.slice(0, 8)}</span>
            </div>
          ))}
        </div>

        {/* Artifact viewer columns */}
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            flex: 1,
            minHeight: 0,
            width: '100%',
          }}
        >
          {runUuids.map((runUuid, index) => (
            <div
              key={runUuid}
              css={{
                flex: '1 1 0%',
                minWidth: 0,
                borderLeft: index > 0 ? `1px solid ${theme.colors.borderDecorative}` : 'none',
                padding: !artifactPath ? theme.spacing.md : 0,
                overflow: 'auto',
                display: 'flex',
                flexDirection: 'column',
              }}
            >
              <ShowArtifactPage
                runUuid={runUuid}
                artifactRootUri={runInfos[index].artifactUri}
                path={artifactPath}
                experimentId={runInfos[index].experimentId}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
