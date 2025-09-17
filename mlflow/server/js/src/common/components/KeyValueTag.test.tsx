import { screen, renderWithIntl } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import type { KeyValueEntity } from '../types';
import { KeyValueTag, getKeyAndValueComplexTruncation } from './KeyValueTag';

describe('KeyValueTag', () => {
  const handleTagClose = jest.fn();
  function renderTestComponent(tag: KeyValueEntity, isClosable = true) {
    renderWithIntl(<KeyValueTag tag={tag} isClosable={isClosable} onClose={isClosable ? handleTagClose : undefined} />);
  }

  function createTestTag(key: string, value: string): KeyValueEntity {
    return {
      key,
      value,
    } as KeyValueEntity;
  }

  test('it should render only tag key if tag value is empty', () => {
    renderTestComponent(createTestTag('tagKey', ''));
    expect(screen.getByText('tagKey')).toBeInTheDocument();
    expect(screen.queryByText(':')).not.toBeInTheDocument();
  });

  test('it should render tag key and value if tag value is not empty', () => {
    renderTestComponent(createTestTag('tagKey', 'tagValue'));
    expect(screen.getByText('tagKey')).toBeInTheDocument();
    expect(screen.getByText(': tagValue')).toBeInTheDocument();
  });

  test('it should call handleTagClose when click on close button', () => {
    renderTestComponent(createTestTag('tagKey', 'tagValue'));
    expect(handleTagClose).not.toHaveBeenCalled();
    screen.getByRole('button').click();
    expect(handleTagClose).toHaveBeenCalled();
  });

  test('it should not render handleTagClose when onClose is not provided', () => {
    renderTestComponent(createTestTag('tagKey', 'tagValue'), false);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  describe('getKeyAndValueComplexTruncation', () => {
    test('it should not truncate tag key and value if they are short', () => {
      const result = getKeyAndValueComplexTruncation(createTestTag('tagKey', 'tagValue'));
      expect(result.shouldTruncateKey).toBe(false);
      expect(result.shouldTruncateValue).toBe(false);
    });

    test('it should truncate tag key if it is too long', () => {
      const longKey = '123'.repeat(100);
      const result = getKeyAndValueComplexTruncation(createTestTag(longKey, 'value'));
      expect(result.shouldTruncateKey).toBe(true);
      expect(result.shouldTruncateValue).toBe(false);
    });

    test('it should truncate tag value if it is too long', () => {
      const longValue = '123'.repeat(100);
      const result = getKeyAndValueComplexTruncation(createTestTag('key', longValue));
      expect(result.shouldTruncateKey).toBe(false);
      expect(result.shouldTruncateValue).toBe(true);
    });

    test('it should truncate tag key and value if they are too long', () => {
      const longKey = '123'.repeat(100);
      const longValue = 'abc'.repeat(100);
      const result = getKeyAndValueComplexTruncation(createTestTag(longKey, longValue));
      expect(result.shouldTruncateKey).toBe(true);
      expect(result.shouldTruncateValue).toBe(true);
    });

    test('it should accept custom charsLength', () => {
      const result = getKeyAndValueComplexTruncation(createTestTag('tagKey', 'tagValue'), 5);
      expect(result.shouldTruncateKey).toBe(true);
      expect(result.shouldTruncateValue).toBe(true);
    });
  });
});
