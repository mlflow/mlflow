import { describe, test, expect } from '@jest/globals';

import {
  formatExecutionDuration,
  getSpanNodeParentIds,
  getTimelineTreeExpandedRows,
  getTimelineTreeNodesMap,
} from './TimelineTree.utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { MOCK_TRACE } from '../../ModelTraceExplorer.test-utils';
import { parseModelTraceToTree } from '../ModelTraceExplorer.utils';

describe('TimelineTree.utils', () => {
  test('getSpanNodeParentIds', () => {
    const rootNode = parseModelTraceToTree(MOCK_TRACE) as ModelTraceSpanNode;
    const nodeMap = getTimelineTreeNodesMap([rootNode]);

    // root node has no parent, so it should be an empty set
    const rootParentIds = getSpanNodeParentIds(rootNode, nodeMap);
    expect(rootParentIds).toEqual(new Set());

    // expect that a nested child has its parents constructed correctly
    const generateResponse = (rootNode?.children ?? [])[0] as ModelTraceSpanNode;
    const rephraseChat = (generateResponse?.children ?? [])[0] as ModelTraceSpanNode;
    const childParentIds = getSpanNodeParentIds(rephraseChat, nodeMap);
    expect(childParentIds).toEqual(new Set(['document-qa-chain', '_generate_response']));
  });

  test('flattens only expanded rows and retains hierarchy state', () => {
    const leaf = (key: string): ModelTraceSpanNode => ({
      key,
      title: key,
      start: 0,
      end: 1,
      traceId: 'trace',
      assessments: [],
    });
    const first: ModelTraceSpanNode = {
      ...leaf('first'),
      children: [leaf('first-child'), leaf('first-sibling')],
    };
    const second: ModelTraceSpanNode = {
      ...leaf('second'),
      children: [leaf('second-child'), leaf('selected')],
    };
    const collapsed: ModelTraceSpanNode = { ...leaf('collapsed'), children: [leaf('collapsed-child')] };
    const rows = getTimelineTreeExpandedRows([first, second, collapsed], new Set(['first', 'second']), 'selected');

    expect(rows.map(({ node }) => node.key)).toEqual([
      'first',
      'first-child',
      'first-sibling',
      'second',
      'second-child',
      'selected',
      'collapsed',
    ]);
    expect(rows[1].linesToRender).toEqual([{ shouldRender: true, isActive: false }]);
    expect(rows[4].linesToRender).toEqual([{ shouldRender: true, isActive: true }]);
    expect(rows[5].linesToRender).toEqual([{ shouldRender: false, isActive: false }]);
  });
});

describe('formatExecutionDuration', () => {
  test.each([
    ['2.201498101s', '2.20s'],
    ['0.071573452s', '71.57ms'],
    ['350ms', '0.35s'],
    ['2.5m', '2.50m'],
    ['900us', '0.90ms'],
    ['12.3h', '12.30h'],
    ['not-a-duration', 'not-a-duration'],
  ])('formats %s as %s', (value, expected) => {
    expect(formatExecutionDuration(value)).toBe(expected);
  });
});
