import { describe, expect, it } from '@jest/globals';

import type { A2uiMessage } from '@a2ui/web_core/v0_9';

import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { ModelSpanType } from '../ModelTrace.types';
import type { CustomViewData } from './customViewBuilders';
import { resolveTemplate } from './resolveTemplate';

const makeNode = (node: Partial<ModelTraceSpanNode> & { key: string; start: number }): ModelTraceSpanNode =>
  node as unknown as ModelTraceSpanNode;

const nodeMap = {
  root: makeNode({ key: 'root', start: 0, title: 'agent', type: ModelSpanType.AGENT }),
  tool0: makeNode({ key: 'tool0', start: 10, parentId: 'root', title: 'run_sql_query', type: ModelSpanType.TOOL }),
} satisfies Record<string, ModelTraceSpanNode>;

const viewData: CustomViewData = {
  metrics: { status: 'OK', latency: '275ms', totalTokens: '1,024', assessments: '3' } as CustomViewData['metrics'],
  assessmentItems: [],
};

const ctx = { viewData, nodeMap };

// Reads the resolved components array out of the (single) updateComponents msg.
const componentsOf = (messages: A2uiMessage[]): Record<string, unknown>[] => {
  const msg = messages.find((m) => 'updateComponents' in m) as unknown as Record<string, unknown> | undefined;
  const payload = msg?.updateComponents as Record<string, unknown> | undefined;
  return (payload?.components ?? []) as Record<string, unknown>[];
};

const updateComponents = (components: Record<string, unknown>[]): A2uiMessage =>
  ({ version: 'v0.9', updateComponents: { surfaceId: 'main', components } }) as unknown as A2uiMessage;

describe('resolveTemplate', () => {
  it('resolves scalar source markers against the current trace', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          { id: 'root', component: 'StatCard', value: { $source: 'metrics.latency' }, label: 'Latency' },
        ]),
      ],
      ctx,
    );
    const [stat] = componentsOf(resolved);
    expect(stat.value).toBe('275ms');
    expect(stat.label).toBe('Latency');
  });

  it('materializes an assessments structural marker into AssessmentCard children', () => {
    const withAssessments: CustomViewData = {
      ...viewData,
      assessmentItems: [
        { name: 'correctness', value: 'yes', sentiment: 'positive' },
        { name: 'relevance', value: 'no', sentiment: 'negative' },
      ] as CustomViewData['assessmentItems'],
    };
    const resolved = resolveTemplate(
      [updateComponents([{ id: 'root', component: 'AssessmentBoard', children: { $source: 'assessments' } }])],
      { viewData: withAssessments, nodeMap },
    );
    const components = componentsOf(resolved);
    const board = components.find((c) => c.id === 'root')!;
    // The marker becomes concrete child ids and the cards are appended as siblings.
    expect((board.children as string[]).length).toBe(2);
    const cards = components.filter((c) => c.component === 'AssessmentCard');
    expect(cards.map((c) => c.name)).toEqual(['correctness', 'relevance']);
  });

  // Component type is irrelevant to resolveTemplate (it resolves markers generically,
  // regardless of catalog membership) — a synthetic name keeps these two cases from
  // implying any specific real catalog component carries a "spanId" prop.
  it('resolves a $spanRef prop to a concrete span id', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          {
            id: 'root',
            component: 'SpanBoundTestComponent',
            name: 'Helpful',
            spanId: { $spanRef: { type: 'TOOL' } },
          },
        ]),
      ],
      ctx,
    );
    expect(componentsOf(resolved)[0].spanId).toBe('tool0');
  });

  it('drops an unresolved $spanRef prop instead of pointing at a missing span', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          {
            id: 'root',
            component: 'SpanBoundTestComponent',
            name: 'Helpful',
            spanId: { $spanRef: { type: 'TOOL', nth: 9 } },
          },
        ]),
      ],
      ctx,
    );
    expect('spanId' in componentsOf(resolved)[0]).toBe(false);
  });

  it('prunes a renderIfSpan subtree when the guard resolves to no span', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          { id: 'root', component: 'Column', children: ['card', 'keep'] },
          { id: 'card', component: 'Card', renderIfSpan: { type: 'TOOL', nth: 9 }, child: 'inner' },
          { id: 'inner', component: 'Text', text: 'hidden' },
          { id: 'keep', component: 'Text', text: 'shown' },
        ]),
      ],
      ctx,
    );
    const components = componentsOf(resolved);
    const ids = components.map((c) => c.id);
    expect(ids).toEqual(['root', 'keep']);
    expect((components.find((c) => c.id === 'root')?.children as string[]) ?? []).toEqual(['keep']);
  });

  it('collapses the whole Card when the guard sits on its required child, not the Card', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          { id: 'root', component: 'Column', children: ['card', 'keep'] },
          // No guard on the Card itself — it sits on the Card's single child.
          { id: 'card', component: 'Card', child: 'inner' },
          { id: 'inner', component: 'Text', renderIfSpan: { type: 'TOOL', nth: 9 }, text: 'hidden' },
          { id: 'keep', component: 'Text', text: 'shown' },
        ]),
      ],
      ctx,
    );
    const components = componentsOf(resolved);
    // The Card is pruned along with its child rather than left childless (which
    // would fail strict Card validation before rendering).
    expect(components.map((c) => c.id)).toEqual(['root', 'keep']);
    expect((components.find((c) => c.id === 'root')?.children as string[]) ?? []).toEqual(['keep']);
  });

  it('cascades the collapse to a parent Card when its child Card collapses', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          { id: 'root', component: 'Column', children: ['outer', 'keep'] },
          { id: 'outer', component: 'Card', child: 'inner' },
          { id: 'inner', component: 'Card', child: 'leaf' },
          { id: 'leaf', component: 'Text', renderIfSpan: { type: 'TOOL', nth: 9 }, text: 'hidden' },
          { id: 'keep', component: 'Text', text: 'shown' },
        ]),
      ],
      ctx,
    );
    const components = componentsOf(resolved);
    // leaf -> inner Card -> outer Card all collapse; only the sibling survives.
    expect(components.map((c) => c.id)).toEqual(['root', 'keep']);
    expect((components.find((c) => c.id === 'root')?.children as string[]) ?? []).toEqual(['keep']);
  });

  it('keeps a renderIfSpan subtree when the guard resolves, and strips the guard prop', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          { id: 'root', component: 'Card', renderIfSpan: { type: 'TOOL' }, child: 'inner' },
          { id: 'inner', component: 'Text', text: 'shown' },
        ]),
      ],
      ctx,
    );
    const components = componentsOf(resolved);
    expect(components.map((c) => c.id)).toEqual(['root', 'inner']);
    expect('renderIfSpan' in components[0]).toBe(false);
  });

  it('passes a template with no markers through unchanged', () => {
    const template = [updateComponents([{ id: 'root', component: 'Text', text: 'literal' }])];
    expect(resolveTemplate(template, ctx)).toEqual(template);
  });
});
