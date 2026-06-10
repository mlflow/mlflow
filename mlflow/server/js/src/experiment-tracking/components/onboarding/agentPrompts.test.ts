import { describe, it, expect } from '@jest/globals';
import { buildInstrumentPrompt, buildInstrumentAssistantPrompt } from '../traces/quickstart/tracesAgentPrompt';
import {
  buildEvaluatePrompt,
  buildEvaluateAssistantPrompt,
} from '../../pages/experiment-evaluation-runs/evalRunsAgentPrompt';
import {
  buildCreatePromptPrompt,
  buildCreatePromptAssistantPrompt,
} from '../../pages/prompts/components/promptsAgentPrompt';

describe('coding-agent prompt builders', () => {
  it('build_instrument_prompt interpolates experiment name and references both skills', () => {
    const out = buildInstrumentPrompt('my-exp');
    expect(out).toContain('Target experiment: my-exp.');
    expect(out).toContain('instrumenting-with-mlflow-tracing');
    expect(out).toContain('searching-mlflow-docs');
    expect(out).toContain('github.com/mlflow/skills');
    expect(out).not.toContain('.databrickscfg');
  });

  it('build_evaluate_prompt interpolates experiment ID into both the prose and the Python call', () => {
    const out = buildEvaluatePrompt('42');
    expect(out).toContain('Target experiment ID: 42.');
    expect(out).toContain('experiment_id="42"');
    expect(out).toContain('searching-mlflow-docs');
    expect(out).toContain('mlflow.genai.evaluate');
    expect(out).not.toContain('.databrickscfg');
  });

  it('build_create_prompt_prompt walks all 6 conversational steps and references the docs skill', () => {
    const out = buildCreatePromptPrompt();
    expect(out).toMatch(/1\.\s/);
    expect(out).toMatch(/6\.\s/);
    expect(out).toContain('registered prompt');
    expect(out).toContain('searching-mlflow-docs');
  });
});

describe('assistant prompt builders', () => {
  it('build_instrument_assistant_prompt interpolates experiment and does NOT reference local skills', () => {
    const out = buildInstrumentAssistantPrompt('my-exp');
    expect(out).toContain('Target experiment: my-exp.');
    expect(out).not.toContain('searching-mlflow-docs');
    expect(out).not.toContain('instrumenting-with-mlflow-tracing');
    expect(out).toContain('Ask');
  });

  it('build_evaluate_assistant_prompt does NOT reference local skills', () => {
    const out = buildEvaluateAssistantPrompt('42');
    expect(out).toContain('Target experiment ID: 42.');
    expect(out).not.toContain('searching-mlflow-docs');
    expect(out).toContain('Ask');
  });

  it('build_create_prompt_assistant_prompt walks the 6 steps and does NOT reference local skills', () => {
    const out = buildCreatePromptAssistantPrompt();
    expect(out).toMatch(/1\.\s/);
    expect(out).toMatch(/6\.\s/);
    expect(out).not.toContain('searching-mlflow-docs');
  });
});
