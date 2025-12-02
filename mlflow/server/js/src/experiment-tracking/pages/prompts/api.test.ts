import { describe, it, expect } from '@jest/globals';
import { setupServer } from '../../../common/utils/setup-msw';
import { NotFoundError } from '@databricks/web-shared/errors';
import { rest } from 'msw';
import { RegisteredPromptsApi } from './api';
import { buildSearchFilterClause } from './utils';

describe('PromptsPage', () => {
  const server = setupServer();

  it('should properly return error when API responds with bare status', async () => {
    server.use(rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) => res(ctx.status(404))));

    const expectedMessage = new NotFoundError({}).message;

    await expect(RegisteredPromptsApi.listRegisteredPrompts()).rejects.toThrow(expectedMessage);
  });

  it('should properly return error with message extracted from API', async () => {
    server.use(
      rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) =>
        res(
          ctx.status(404),
          ctx.json({
            code: 'NOT_FOUND',
            message: 'Custom message: models not found',
          }),
        ),
      ),
    );

    await expect(RegisteredPromptsApi.listRegisteredPrompts()).rejects.toThrow('Custom message: models not found');
  });
});

describe('buildSearchFilterClause', () => {
  it('should return empty string when search filter is undefined', () => {
    expect(buildSearchFilterClause(undefined)).toBe('');
  });

  it('should return empty string when search filter is empty string', () => {
    expect(buildSearchFilterClause('')).toBe('');
  });

  it('should wrap simple text search in ILIKE pattern', () => {
    expect(buildSearchFilterClause('my-prompt')).toBe("name ILIKE '%my-prompt%'");
  });

  it('should treat filter with ILIKE as raw SQL filter', () => {
    expect(buildSearchFilterClause('name ILIKE "%test%"')).toBe('name ILIKE "%test%"');
  });

  it('should treat filter with LIKE as raw SQL filter', () => {
    expect(buildSearchFilterClause('name LIKE "test%"')).toBe('name LIKE "test%"');
  });

  it('should treat filter with equals sign as raw SQL filter', () => {
    expect(buildSearchFilterClause('tags.environment = "production"')).toBe('tags.environment = "production"');
  });

  it('should treat filter with not equals as raw SQL filter', () => {
    expect(buildSearchFilterClause('tags.status != "archived"')).toBe('tags.status != "archived"');
  });

  it('should be case insensitive for SQL keywords', () => {
    expect(buildSearchFilterClause('name ilike "%test%"')).toBe('name ilike "%test%"');
    expect(buildSearchFilterClause('name like "test%"')).toBe('name like "test%"');
  });

  it('should require whitespace before ILIKE/LIKE to avoid false positives', () => {
    expect(buildSearchFilterClause('prompt-ILIKE-test')).toBe("name ILIKE '%prompt-ILIKE-test%'");
    expect(buildSearchFilterClause('prompt-LIKE-test')).toBe("name ILIKE '%prompt-LIKE-test%'");
  });

  it('should handle complex SQL filters', () => {
    const complexFilter = 'name ILIKE "%prompt%" AND tags.env = "prod"';
    expect(buildSearchFilterClause(complexFilter)).toBe(complexFilter);
  });
});
