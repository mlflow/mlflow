import React, { useState } from 'react';
import {
  DropdownMenu,
  Button,
  ChevronDownIcon,
  PlayIcon,
  CodeIcon,
  BracketsCurlyIcon,
  Modal,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { COMPONENT_ID_PREFIX } from './constants';
import type { ScheduledScorer } from './types';

interface JudgeUseDropdownProps {
  selectedScorers: ScheduledScorer[];
  experimentId: string;
}

const JudgeUseDropdown: React.FC<JudgeUseDropdownProps> = ({ selectedScorers, experimentId }) => {
  const [snippetModal, setSnippetModal] = useState<'sdk' | 'api' | null>(null);

  const scorerNames = selectedScorers.map((s) => s.name);
  const namesStr = scorerNames.map((n) => `"${n}"`).join(', ');

  const sdkSnippet = `import mlflow

# Run selected judges on traces
mlflow.evaluate(
    experiment_id="${experimentId}",
    scorers=[${namesStr}],
)`;

  const apiSnippet = `POST /api/2.0/mlflow/traces/evaluate
{
  "experiment_id": "${experimentId}",
  "scorer_names": [${namesStr}]
}`;

  return (
    <>
      <DropdownMenu.Root>
        <DropdownMenu.Trigger asChild>
          <Button
            componentId={`${COMPONENT_ID_PREFIX}.use-dropdown-trigger`}
            type="primary"
            size="small"
            endIcon={<ChevronDownIcon />}
          >
            <FormattedMessage defaultMessage="Use" description="Label for use judges dropdown button" />
          </Button>
        </DropdownMenu.Trigger>
        <DropdownMenu.Content align="end" minWidth={180}>
          <DropdownMenu.Item
            componentId={`${COMPONENT_ID_PREFIX}.use-in-ui`}
            onClick={() => {
              // Future: open judge config modal
            }}
          >
            <DropdownMenu.IconWrapper>
              <PlayIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage defaultMessage="Use in UI" description="Use judges in the UI option" />
          </DropdownMenu.Item>
          <DropdownMenu.Item
            componentId={`${COMPONENT_ID_PREFIX}.use-python-sdk`}
            onClick={() => setSnippetModal('sdk')}
          >
            <DropdownMenu.IconWrapper>
              <CodeIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage defaultMessage="Python SDK" description="Show Python SDK code snippet" />
          </DropdownMenu.Item>
          <DropdownMenu.Item
            componentId={`${COMPONENT_ID_PREFIX}.use-rest-api`}
            onClick={() => setSnippetModal('api')}
          >
            <DropdownMenu.IconWrapper>
              <BracketsCurlyIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage defaultMessage="REST API" description="Show REST API code snippet" />
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
      <Modal
        componentId={`${COMPONENT_ID_PREFIX}.use-snippet-modal`}
        title={
          snippetModal === 'sdk' ? (
            <FormattedMessage defaultMessage="Python SDK" description="SDK snippet modal title" />
          ) : (
            <FormattedMessage defaultMessage="REST API" description="API snippet modal title" />
          )
        }
        visible={snippetModal !== null}
        onCancel={() => setSnippetModal(null)}
        onOk={() => setSnippetModal(null)}
        footer={null}
      >
        <pre
          css={{
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            fontFamily: 'monospace',
            fontSize: 13,
          }}
        >
          {snippetModal === 'sdk' ? sdkSnippet : apiSnippet}
        </pre>
      </Modal>
    </>
  );
};

export default JudgeUseDropdown;
