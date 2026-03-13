import { describe, test, expect } from '@jest/globals';
import { getDatasetSourceUrl } from './DatasetUtils';
import { DatasetSourceTypes } from '../types';
import type { RunDatasetWithTags } from '../types';

const createDatasetWithTags = (sourceType: string, source: string): RunDatasetWithTags =>
  ({
    dataset: {
      name: 'test-dataset',
      digest: 'abc123',
      sourceType,
      source,
    },
    tags: [],
  }) as any;

describe('getDatasetSourceUrl', () => {
  test('returns url for HTTP source type', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.HTTP, JSON.stringify({ url: 'https://example.com/data' }));
    expect(getDatasetSourceUrl(dataset)).toBe('https://example.com/data');
  });

  test('returns url for EXTERNAL source type', () => {
    const dataset = createDatasetWithTags(
      DatasetSourceTypes.EXTERNAL,
      JSON.stringify({ url: 'https://external.example.com/dataset' }),
    );
    expect(getDatasetSourceUrl(dataset)).toBe('https://external.example.com/dataset');
  });

  test('returns uri for S3 source type', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.S3, JSON.stringify({ uri: 's3://bucket/path' }));
    expect(getDatasetSourceUrl(dataset)).toBe('s3://bucket/path');
  });

  test('returns constructed URL for HUGGING_FACE source type', () => {
    const dataset = createDatasetWithTags(
      DatasetSourceTypes.HUGGING_FACE,
      JSON.stringify({ path: 'org/dataset-name' }),
    );
    expect(getDatasetSourceUrl(dataset)).toBe('https://huggingface.co/datasets/org/dataset-name');
  });

  test('returns null for unknown source type', () => {
    const dataset = createDatasetWithTags('unknown', JSON.stringify({ url: 'https://example.com' }));
    expect(getDatasetSourceUrl(dataset)).toBeNull();
  });

  test('returns null when source JSON is invalid', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.HTTP, 'not-json');
    expect(getDatasetSourceUrl(dataset)).toBeNull();
  });

  test('returns null for EXTERNAL source with no url field', () => {
    const dataset = createDatasetWithTags(DatasetSourceTypes.EXTERNAL, JSON.stringify({ other: 'value' }));
    expect(getDatasetSourceUrl(dataset)).toBeNull();
  });
});
