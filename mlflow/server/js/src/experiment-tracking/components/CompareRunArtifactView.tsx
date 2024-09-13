import { useState } from 'react';
import ShowArtifactPage from './artifact-view-components/ShowArtifactPage';
import { RunInfoEntity } from '../types';
import { useRunsArtifacts } from './experiment-page/hooks/useRunsArtifacts';
import { getCommonArtifacts } from './experiment-page/utils/getCommonArtifacts';
import { CompareRunArtifactViewSidebar } from './CompareRunArtifactViewSidebar';
import { useDesignSystemTheme } from '@databricks/design-system';
import './CompareRunArtifactView.css';
import { FormattedMessage } from 'react-intl';

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
  const [artifactPath, setArtifactPath] = useState<string>();

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
    <div className="artifact-container">
      <CompareRunArtifactViewSidebar artifacts={commonArtifacts} onSelectArtifact={setArtifactPath} />
      <div
        className="artifact-right"
        css={{
          border: `1px solid ${theme.colors.grey300}`,
          borderLeft: 'none',
        }}
      >
        <div css={{ display: 'flex', flexDirection: 'row' }}>
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
                path={artifactPath ?? '/'}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
