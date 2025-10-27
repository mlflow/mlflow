import { useState } from 'react';
import ShowArtifactPage from './artifact-view-components/ShowArtifactPage';
import type { RunInfoEntity } from '../types';
import { useRunsArtifacts } from './experiment-page/hooks/useRunsArtifacts';
import { getCommonArtifacts } from './experiment-page/utils/getCommonArtifacts';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ArtifactViewTree } from './ArtifactViewTree';
import { getBasename } from '../../common/utils/FileUtils';

export const CompareRunArtifactView = ({
  runUuids,
  runInfos,
  colWidth,
}: {
  runUuids: string[];
  runInfos: RunInfoEntity[];
  colWidth: number;
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
        height: '100vh',
      }}
    >
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          color: theme.colors.textPrimary,
          flex: '1 1 0%',
          whiteSpace: 'nowrap',
          border: `1px solid ${theme.colors.grey300}`,
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
      <div
        css={{
          border: `1px solid ${theme.colors.grey300}`,
          borderLeft: 'none',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap' }}>
          {runUuids.map((runUuid, index) => (
            <div
              key={runUuid}
              style={{
                width: `${colWidth}px`,
                borderBottom: `1px solid ${theme.colors.grey300}`,
                padding: !artifactPath ? theme.spacing.md : 0,
                overflow: 'auto',
                whiteSpace: 'nowrap',
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
