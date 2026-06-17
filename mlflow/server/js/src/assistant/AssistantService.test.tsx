import { describe, test, expect, jest } from '@jest/globals';
import { processContentBlocks } from './AssistantService';
import type { ToolUseInfo, ToolResultInfo } from './types';

describe('processContentBlocks', () => {
  test('emits text blocks via onMessage', () => {
    const onMessage = jest.fn();
    processContentBlocks([{ text: 'hello' }], onMessage);
    expect(onMessage).toHaveBeenCalledWith('hello');
  });

  test('emits tool_use blocks via onToolUse', () => {
    const onToolUse = jest.fn<(tools: ToolUseInfo[]) => void>();
    processContentBlocks(
      [{ id: 'tu1', name: 'Bash', input: { command: 'ls', description: 'list' } }],
      jest.fn(),
      onToolUse,
    );
    expect(onToolUse).toHaveBeenCalledWith([
      { id: 'tu1', name: 'Bash', description: 'list', input: { command: 'ls', description: 'list' } },
    ]);
  });

  test('emits a tool_result block via onToolResult with string content', () => {
    const onToolResult = jest.fn<(r: ToolResultInfo) => void>();
    processContentBlocks(
      [{ tool_use_id: 'tu1', content: 'done', is_error: false }],
      jest.fn(),
      jest.fn(),
      onToolResult,
    );
    expect(onToolResult).toHaveBeenCalledWith({ toolUseId: 'tu1', content: 'done', isError: false });
  });

  test('JSON-stringifies array tool_result content', () => {
    const onToolResult = jest.fn<(r: ToolResultInfo) => void>();
    processContentBlocks(
      [{ tool_use_id: 'tu1', content: [{ a: 1 }], is_error: true }],
      jest.fn(),
      jest.fn(),
      onToolResult,
    );
    expect(onToolResult).toHaveBeenCalledWith({
      toolUseId: 'tu1',
      content: JSON.stringify([{ a: 1 }], null, 2),
      isError: true,
    });
  });

  test('normalizes null tool_result content to an empty string', () => {
    const onToolResult = jest.fn<(r: ToolResultInfo) => void>();
    processContentBlocks([{ tool_use_id: 'tu1', content: null }], jest.fn(), jest.fn(), onToolResult);
    expect(onToolResult).toHaveBeenCalledWith({ toolUseId: 'tu1', content: '', isError: false });
  });

  test('routes a tool_result block to onToolResult, not onToolUse', () => {
    const onToolUse = jest.fn();
    const onToolResult = jest.fn();
    processContentBlocks(
      [{ tool_use_id: 'tu1', name: 'Bash', input: {}, content: 'x' }],
      jest.fn(),
      onToolUse,
      onToolResult,
    );
    expect(onToolResult).toHaveBeenCalledTimes(1);
    expect(onToolUse).not.toHaveBeenCalled();
  });
});
