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
  tool0: makeNode({
    key: 'tool0',
    start: 10,
    parentId: 'root',
    title: 'run_sql_query',
    type: ModelSpanType.TOOL,
    outputs: { choices: [{ message: { content: 'The answer is 42.' } }] },
  }),
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

  // Markdown is bindable like any other component: the resolver walks every prop
  // generically, so a spanField marker on its "text" renders per-trace prose. Only
  // hand-written Markdown has to stay trace-agnostic.
  it('resolves a spanField marker on a Markdown "text" prop, leaving static props literal', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          {
            id: 'root',
            component: 'Markdown',
            title: 'Model answer',
            text: {
              $source: 'spanField',
              spanRef: { type: 'TOOL', nth: 0 },
              field: 'outputs',
              path: ['choices', 0, 'message', 'content'],
            },
          },
        ]),
      ],
      ctx,
    );
    const [markdown] = componentsOf(resolved);
    expect(markdown.text).toBe('The answer is 42.');
    expect(markdown.title).toBe('Model answer');
  });

  // Backs the catalog guidance to point a Markdown binding at a SCALAR leaf: a
  // pathless whole-object field still resolves, but only to a raw JSON string.
  it('resolves a pathless spanField binding to a JSON string rather than an object', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          {
            id: 'root',
            component: 'Markdown',
            text: { $source: 'spanField', spanRef: { type: 'TOOL', nth: 0 }, field: 'outputs' },
          },
        ]),
      ],
      ctx,
    );
    expect(componentsOf(resolved)[0].text).toBe(JSON.stringify(nodeMap.tool0.outputs));
  });

  it('resolves a $spanRef prop to a concrete span id', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          {
            id: 'root',
            component: 'FeedbackThumbsUpDownButtons',
            name: 'Helpful',
            spanId: { $spanRef: { type: 'TOOL' } },
          },
        ]),
      ],
      ctx,
    );
    expect(componentsOf(resolved)[0].spanId).toBe('tool0');
  });

  it('prunes feedback whose $spanRef target is missing instead of logging it at trace level', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          {
            id: 'root',
            component: 'FeedbackThumbsUpDownButtons',
            name: 'Helpful',
            spanId: { $spanRef: { type: 'TOOL', nth: 9 } },
          },
        ]),
      ],
      ctx,
    );
    expect(componentsOf(resolved)).toEqual([{ id: 'root', component: 'Column', children: [] }]);
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

  // A rootless stream is rejected wholesale by `validateAndPrepareMessages`, which
  // would surface a render error instead of hiding the guarded content — so a prune
  // that reaches the root collapses the view to an empty root instead.
  it('collapses to an empty root when the guard sits on the root itself', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          { id: 'root', component: 'Column', renderIfSpan: { type: 'TOOL', nth: 9 }, children: ['card'] },
          { id: 'card', component: 'Text', text: 'hidden' },
        ]),
      ],
      ctx,
    );
    expect(componentsOf(resolved)).toEqual([{ id: 'root', component: 'Column', children: [] }]);
  });

  it('collapses to an empty root when the child cascade reaches a root Card', () => {
    const resolved = resolveTemplate(
      [
        updateComponents([
          // No guard on the root Card — it sits on the Card's single child, and the
          // collapse cascades up to the root.
          { id: 'root', component: 'Card', child: 'inner' },
          { id: 'inner', component: 'Text', renderIfSpan: { type: 'TOOL', nth: 9 }, text: 'hidden' },
        ]),
      ],
      ctx,
    );
    expect(componentsOf(resolved)).toEqual([{ id: 'root', component: 'Column', children: [] }]);
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
