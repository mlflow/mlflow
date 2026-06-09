import { describe, it, expect } from '@jest/globals';
import { buildInstrumentPrompt, buildInstrumentAssistantPrompt } from '../traces/quickstart/tracesAgentPrompt';
import {
  buildEvaluatePrompt,
  buildEvaluateAssistantPrompt,
} from '../../pages/experiment-evaluation-runs/evalRunsAgentPrompt';
import { buildCreateExperimentPrompt, buildCreateExperimentAssistantPrompt } from '../experimentListAgentPrompt';
import {
  buildCreatePromptPrompt,
  buildCreatePromptAssistantPrompt,
} from '../../pages/prompts/components/promptsAgentPrompt';

const URI = 'http://localhost:5000';

describe('coding-agent prompt builders', () => {
  it('build_instrument_prompt interpolates URI and experiment name and references both skills', () => {
    const out = buildInstrumentPrompt(URI, 'my-exp');
    expect(out).toContain(URI);
    expect(out).toContain('my-exp');
    expect(out).toContain('instrumenting-with-mlflow-tracing');
    expect(out).toContain('searching-mlflow-docs');
    expect(out).toContain('github.com/mlflow/skills');
  });

  it('build_evaluate_prompt interpolates URI + experiment ID and references the docs skill', () => {
    const out = buildEvaluatePrompt(URI, '42');
    expect(out).toContain(URI);
    expect(out).toContain('experiment_id="42"');
    expect(out).toContain('searching-mlflow-docs');
    expect(out).toContain('mlflow.genai.evaluate');
  });

  it('build_create_experiment_prompt interpolates URI and references the docs skill', () => {
    const out = buildCreateExperimentPrompt(URI);
    expect(out).toContain(URI);
    expect(out).toContain('mlflow.set_experiment');
    expect(out).toContain('searching-mlflow-docs');
  });

  it('build_create_prompt_prompt walks all 6 conversational steps and references the docs skill', () => {
    const out = buildCreatePromptPrompt(URI);
    expect(out).toContain(URI);
    expect(out).toMatch(/1\.\s/);
    expect(out).toMatch(/6\.\s/);
    expect(out).toContain('Prompt Registry');
    expect(out).toContain('searching-mlflow-docs');
  });
});

describe('assistant prompt builders', () => {
  it('build_instrument_assistant_prompt interpolates URI + experiment and does NOT reference local skills', () => {
    const out = buildInstrumentAssistantPrompt(URI, 'my-exp');
    expect(out).toContain(URI);
    expect(out).toContain('my-exp');
    expect(out).not.toContain('searching-mlflow-docs');
    expect(out).not.toContain('instrumenting-with-mlflow-tracing');
    expect(out).toContain('Ask me');
  });

  it('build_evaluate_assistant_prompt does NOT reference local skills', () => {
    const out = buildEvaluateAssistantPrompt(URI, '42');
    expect(out).toContain('42');
    expect(out).not.toContain('searching-mlflow-docs');
    expect(out).toContain('Ask me');
  });

  it('build_create_experiment_assistant_prompt does NOT reference local skills', () => {
    const out = buildCreateExperimentAssistantPrompt(URI);
    expect(out).toContain(URI);
    expect(out).not.toContain('searching-mlflow-docs');
    expect(out).toContain('Ask me');
  });

  it('build_create_prompt_assistant_prompt walks the 6 steps and does NOT reference local skills', () => {
    const out = buildCreatePromptAssistantPrompt(URI);
    expect(out).toContain(URI);
    expect(out).toMatch(/1\.\s/);
    expect(out).toMatch(/6\.\s/);
    expect(out).not.toContain('searching-mlflow-docs');
  });
});
