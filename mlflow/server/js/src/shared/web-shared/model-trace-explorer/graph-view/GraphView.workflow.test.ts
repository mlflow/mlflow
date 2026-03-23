import { describe, test, expect } from '@jest/globals';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import type { GraphSchema } from './GraphView.types';
import { DEFAULT_WORKFLOW_LAYOUT_CONFIG } from './GraphView.types';
import { computeLogicalFlowLayout } from './GraphView.workflow';

function makeSpan(overrides: Partial<ModelTraceSpanNode> & { key: string }): ModelTraceSpanNode {
  return {
    key: overrides.key,
    title: overrides.title ?? overrides.key,
    start: overrides.start ?? 0,
    end: overrides.end ?? 100,
    type: overrides.type ?? 'CHAIN',
    traceId: 'trace-1',
    assessments: [],
    attributes: overrides.attributes ?? {},
    events: [],
    children: overrides.children ?? [],
  } as ModelTraceSpanNode;
}

describe('computeLogicalFlowLayout', () => {
  const schema: GraphSchema = {
    nodes: [
      { id: '__start__', type: 'start' },
      { id: 'agent', data: { name: 'agent' } },
      { id: 'tools', data: { name: 'tools' } },
      { id: '__end__', type: 'end' },
    ],
    edges: [
      { source: '__start__', target: 'agent' },
      { source: 'agent', target: 'tools', conditional: true, data: 'continue' },
      { source: 'agent', target: '__end__', conditional: true, data: 'end' },
      { source: 'tools', target: 'agent' },
    ],
  };

  test('produces nodes and edges from schema', () => {
    const result = computeLogicalFlowLayout(schema, null, DEFAULT_WORKFLOW_LAYOUT_CONFIG);

    expect(result.nodes).toHaveLength(4);
    expect(result.edges).toHaveLength(4);

    const nodeIds = result.nodes.map((n) => n.id);
    expect(nodeIds).toContain('__start__');
    expect(nodeIds).toContain('__end__');
    expect(nodeIds).toContain('agent');
    expect(nodeIds).toContain('tools');
  });

  test('marks structural nodes', () => {
    const result = computeLogicalFlowLayout(schema, null, DEFAULT_WORKFLOW_LAYOUT_CONFIG);

    const startNode = result.nodes.find((n) => n.id === '__start__');
    const endNode = result.nodes.find((n) => n.id === '__end__');
    const agentNode = result.nodes.find((n) => n.id === 'agent');

    expect(startNode?.isStructural).toBe(true);
    expect(startNode?.displayName).toBe('START');
    expect(endNode?.isStructural).toBe(true);
    expect(endNode?.displayName).toBe('END');
    expect(agentNode?.isStructural).toBeFalsy();
  });

  test('marks conditional edges', () => {
    const result = computeLogicalFlowLayout(schema, null, DEFAULT_WORKFLOW_LAYOUT_CONFIG);

    const conditionalEdges = result.edges.filter((e) => e.isConditional);
    expect(conditionalEdges).toHaveLength(2);
    expect(conditionalEdges.map((e) => e.condition).sort()).toEqual(['continue', 'end']);
  });

  test('maps spans to schema nodes and marks executed paths', () => {
    const agentSpan = makeSpan({
      key: 'span-agent',
      title: 'agent',
      start: 100,
      end: 200,
      attributes: { metadata: { langgraph_node: 'agent', langgraph_step: 1 } },
    });

    const toolsSpan = makeSpan({
      key: 'span-tools',
      title: 'tools',
      start: 200,
      end: 300,
      attributes: { metadata: { langgraph_node: 'tools', langgraph_step: 2 } },
    });

    const agentSpan2 = makeSpan({
      key: 'span-agent-2',
      title: 'agent',
      start: 300,
      end: 400,
      attributes: { metadata: { langgraph_node: 'agent', langgraph_step: 3 } },
    });

    const rootSpan = makeSpan({
      key: 'span-root',
      title: 'LangGraph',
      start: 0,
      end: 500,
      children: [agentSpan, toolsSpan, agentSpan2],
    });

    const result = computeLogicalFlowLayout(schema, rootSpan, DEFAULT_WORKFLOW_LAYOUT_CONFIG);

    // Agent node should have 2 spans
    const agentNode = result.nodes.find((n) => n.id === 'agent');
    expect(agentNode?.count).toBe(2);
    expect(agentNode?.isExecuted).toBe(true);
    expect(agentNode?.spans).toHaveLength(2);

    // Tools node should have 1 span
    const toolsNode = result.nodes.find((n) => n.id === 'tools');
    expect(toolsNode?.count).toBe(1);
    expect(toolsNode?.isExecuted).toBe(true);

    // Executed edges: __start__->agent, agent->tools, tools->agent, agent->__end__
    // (the last step's node transitions to __end__ since the graph completed)
    const executedEdges = result.edges.filter((e) => e.isExecuted);
    const executedPairs = executedEdges.map((e) => `${e.sourceId}->${e.targetId}`);
    expect(executedPairs).toContain('__start__->agent');
    expect(executedPairs).toContain('agent->tools');
    expect(executedPairs).toContain('tools->agent');
    expect(executedPairs).toContain('agent->__end__');

    // __start__->agent is a non-conditional executed edge
    const startEdge = result.edges.find((e) => e.sourceId === '__start__' && e.targetId === 'agent');
    expect(startEdge?.isExecuted).toBe(true);
    expect(startEdge?.isConditional).toBe(false);
  });

  test('unexecuted nodes when no matching spans', () => {
    const result = computeLogicalFlowLayout(schema, null, DEFAULT_WORKFLOW_LAYOUT_CONFIG);

    const agentNode = result.nodes.find((n) => n.id === 'agent');
    expect(agentNode?.isExecuted).toBe(false);
    expect(agentNode?.count).toBe(0);
  });

  test('returns empty layout for empty schema', () => {
    const result = computeLogicalFlowLayout({ nodes: [], edges: [] }, null, DEFAULT_WORKFLOW_LAYOUT_CONFIG);
    expect(result.nodes).toHaveLength(0);
    expect(result.edges).toHaveLength(0);
  });

  test('all nodes have valid positions after layout', () => {
    const result = computeLogicalFlowLayout(schema, null, DEFAULT_WORKFLOW_LAYOUT_CONFIG);

    expect(result.width).toBeGreaterThan(0);
    expect(result.height).toBeGreaterThan(0);
    for (const node of result.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(0);
      expect(node.y).toBeGreaterThanOrEqual(0);
    }
  });

  test('react agent layout: nodes layered correctly and no back-edges', () => {
    const reactSchema: GraphSchema = {
      nodes: [
        { id: '__start__', type: 'start' },
        { id: 'react_agent', data: { name: 'react_agent' } },
        { id: 'think', data: { name: 'think' } },
        { id: 'web_search', data: { name: 'web_search' } },
        { id: 'doc_lookup', data: { name: 'doc_lookup' } },
        { id: 'generate_final_answer', data: { name: 'generate_final_answer' } },
        { id: '__end__', type: 'end' },
      ],
      edges: [
        { source: '__start__', target: 'react_agent' },
        { source: 'react_agent', target: 'think', conditional: true },
        { source: 'react_agent', target: 'web_search', conditional: true },
        { source: 'react_agent', target: 'doc_lookup', conditional: true },
        { source: 'react_agent', target: 'generate_final_answer', conditional: true },
        { source: 'think', target: 'react_agent' },
        { source: 'web_search', target: 'react_agent' },
        { source: 'doc_lookup', target: 'react_agent' },
        { source: 'generate_final_answer', target: '__end__' },
      ],
    };

    const result = computeLogicalFlowLayout(reactSchema, null, DEFAULT_WORKFLOW_LAYOUT_CONFIG);

    // No edges should be marked as back-edges in logical flow
    expect(result.edges.every((e) => !e.isBackEdge)).toBe(true);

    // react_agent should be above tools (lower y)
    const agentY = result.nodes.find((n) => n.id === 'react_agent')!.y;
    const thinkY = result.nodes.find((n) => n.id === 'think')!.y;
    expect(agentY).toBeLessThan(thinkY);

    // generate_final_answer should be at the same layer as tools (or below agent)
    const genY = result.nodes.find((n) => n.id === 'generate_final_answer')!.y;
    expect(genY).toBeGreaterThan(agentY);

    // __end__ should be the lowest
    const endY = result.nodes.find((n) => n.id === '__end__')!.y;
    expect(endY).toBeGreaterThan(genY);
  });
});
