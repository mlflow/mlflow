import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { computeWorkflowLayout } from './GraphView.workflow';

/**
 * Helper to create a minimal ModelTraceSpanNode for testing.
 */
function makeSpan(
  overrides: Partial<ModelTraceSpanNode> & { title: string; type?: string },
  children?: ModelTraceSpanNode[],
): ModelTraceSpanNode {
  return {
    key: overrides.key ?? overrides.title,
    title: overrides.title,
    type: overrides.type,
    start: overrides.start ?? 0,
    end: overrides.end ?? 100,
    icon: null as any,
    assessments: [],
    traceId: 'test-trace',
    children: children ?? undefined,
    ...overrides,
  } as ModelTraceSpanNode;
}

describe('computeWorkflowLayout', () => {
  it('returns empty layout for null root', () => {
    const layout = computeWorkflowLayout(null);
    expect(layout.nodes).toHaveLength(0);
    expect(layout.edges).toHaveLength(0);
  });

  it('consolidates spans with same type and name under non-agent parents', () => {
    // Root (CHAIN) -> [tool1, tool2] both named "search"
    const root = makeSpan(
      { title: 'root', type: 'UNKNOWN', start: 0 },
      [
        makeSpan({ title: 'search', type: 'TOOL', key: 's1', start: 10 }),
        makeSpan({ title: 'search', type: 'TOOL', key: 's2', start: 20 }),
      ],
    );

    const layout = computeWorkflowLayout(root);
    // "search" spans should be consolidated into one node
    const searchNodes = layout.nodes.filter((n) => n.displayName === 'search');
    expect(searchNodes).toHaveLength(1);
    expect(searchNodes[0].count).toBe(2);
  });

  it('separates spans with same name under different AGENT parents', () => {
    // Root -> [AgentA, AgentB] each with a child "llm"
    const root = makeSpan(
      { title: 'orchestrator', type: 'UNKNOWN', start: 0 },
      [
        makeSpan(
          { title: 'agent_a', type: 'AGENT', key: 'a', start: 10 },
          [makeSpan({ title: 'llm', type: 'LLM', key: 'llm1', start: 15 })],
        ),
        makeSpan(
          { title: 'agent_b', type: 'AGENT', key: 'b', start: 20 },
          [makeSpan({ title: 'llm', type: 'LLM', key: 'llm2', start: 25 })],
        ),
      ],
    );

    const layout = computeWorkflowLayout(root);
    // "llm" under agent_a and agent_b should be separate nodes
    const llmNodes = layout.nodes.filter((n) => n.displayName === 'llm');
    expect(llmNodes).toHaveLength(2);
    expect(llmNodes[0].count).toBe(1);
    expect(llmNodes[1].count).toBe(1);
  });

  it('consolidates spans under same AGENT parent', () => {
    // Root -> Agent -> [llm1, llm2] both named "llm"
    const root = makeSpan(
      { title: 'root', type: 'UNKNOWN', start: 0 },
      [
        makeSpan(
          { title: 'my_agent', type: 'AGENT', key: 'agent', start: 10 },
          [
            makeSpan({ title: 'llm', type: 'LLM', key: 'llm1', start: 15 }),
            makeSpan({ title: 'llm', type: 'LLM', key: 'llm2', start: 20 }),
          ],
        ),
      ],
    );

    const layout = computeWorkflowLayout(root);
    const llmNodes = layout.nodes.filter((n) => n.displayName === 'llm');
    expect(llmNodes).toHaveLength(1);
    expect(llmNodes[0].count).toBe(2);
  });

  it('separates spans under different CHAIN parents', () => {
    // Root -> [ChainA, ChainB] each with "parse"
    const root = makeSpan(
      { title: 'root', type: 'UNKNOWN', start: 0 },
      [
        makeSpan(
          { title: 'chain_a', type: 'CHAIN', key: 'ca', start: 10 },
          [makeSpan({ title: 'parse', type: 'PARSER', key: 'p1', start: 15 })],
        ),
        makeSpan(
          { title: 'chain_b', type: 'CHAIN', key: 'cb', start: 20 },
          [makeSpan({ title: 'parse', type: 'PARSER', key: 'p2', start: 25 })],
        ),
      ],
    );

    const layout = computeWorkflowLayout(root);
    const parseNodes = layout.nodes.filter((n) => n.displayName === 'parse');
    expect(parseNodes).toHaveLength(2);
  });

  it('handles nested agents correctly', () => {
    // Root (AGENT) -> SubAgent (AGENT) -> llm
    // Root (AGENT) -> llm
    const root = makeSpan(
      { title: 'root_agent', type: 'AGENT', start: 0 },
      [
        makeSpan({ title: 'llm', type: 'LLM', key: 'llm_root', start: 5 }),
        makeSpan(
          { title: 'sub_agent', type: 'AGENT', key: 'sub', start: 10 },
          [makeSpan({ title: 'llm', type: 'LLM', key: 'llm_sub', start: 15 })],
        ),
      ],
    );

    const layout = computeWorkflowLayout(root);
    const llmNodes = layout.nodes.filter((n) => n.displayName === 'llm');
    // llm under root_agent and llm under sub_agent should be separate
    expect(llmNodes).toHaveLength(2);
  });
});
