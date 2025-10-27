import { getSpanNodeParentIds, getTimelineTreeNodesMap } from './TimelineTree.utils';
import type { ModelTraceSpanNode } from '../ModelTrace.types';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';
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
});
