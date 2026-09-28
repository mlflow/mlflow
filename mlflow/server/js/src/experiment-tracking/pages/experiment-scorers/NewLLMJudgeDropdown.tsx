import React from 'react';
import { Button, DropdownMenu, PlusIcon, SparkleDoubleIcon, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { ScorerEvaluationScope } from './constants';
import { useTemplateOptions } from './llmScorerUtils';
import { LLM_TEMPLATE } from './types';

interface NewLLMJudgeDropdownProps {
  onSelect: (template: LLM_TEMPLATE, scope: ScorerEvaluationScope) => void;
}

const NewLLMJudgeDropdown: React.FC<NewLLMJudgeDropdownProps> = ({ onSelect }) => {
  const { theme } = useDesignSystemTheme();
  const { templateOptions: traceTemplates } = useTemplateOptions(ScorerEvaluationScope.TRACES);
  const { templateOptions: sessionTemplates } = useTemplateOptions(ScorerEvaluationScope.SESSIONS);

  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button type="primary" icon={<PlusIcon />} componentId="mlflow.experiment-scorers.new-llm-judge-button">
          <FormattedMessage defaultMessage="New LLM judge" description="Button text to create a new LLM judge" />
        </Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Content css={{ maxHeight: 400, overflowY: 'auto' }}>
        <DropdownMenu.Item
          componentId="mlflow.experiment-scorers.new-custom-llm-judge-menu-item"
          onClick={() => onSelect(LLM_TEMPLATE.CUSTOM, ScorerEvaluationScope.TRACES)}
          css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}
        >
          <PlusIcon />
          <FormattedMessage defaultMessage="Custom judge" description="Menu item to create a custom LLM judge" />
        </DropdownMenu.Item>
        <DropdownMenu.Separator />
        <DropdownMenu.Label>
          <FormattedMessage defaultMessage="Trace judges" description="Heading for trace-level LLM judges" />
        </DropdownMenu.Label>
        {traceTemplates
          .filter(({ value }) => value !== LLM_TEMPLATE.CUSTOM)
          .map(({ value, label }) => (
            <DropdownMenu.Item
              key={value}
              componentId="mlflow.experiment-scorers.new-trace-llm-judge-menu-item"
              onClick={() => onSelect(value, ScorerEvaluationScope.TRACES)}
              css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}
            >
              <SparkleDoubleIcon />
              {label}
            </DropdownMenu.Item>
          ))}
        <DropdownMenu.Separator />
        <DropdownMenu.Label>
          <FormattedMessage defaultMessage="Session judges" description="Heading for session-level LLM judges" />
        </DropdownMenu.Label>
        {sessionTemplates
          .filter(({ value }) => value !== LLM_TEMPLATE.CUSTOM)
          .map(({ value, label }) => (
            <DropdownMenu.Item
              key={value}
              componentId="mlflow.experiment-scorers.new-session-llm-judge-menu-item"
              onClick={() => onSelect(value, ScorerEvaluationScope.SESSIONS)}
              css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}
            >
              <SparkleDoubleIcon />
              {label}
            </DropdownMenu.Item>
          ))}
      </DropdownMenu.Content>
    </DropdownMenu.Root>
  );
};

export default NewLLMJudgeDropdown;
