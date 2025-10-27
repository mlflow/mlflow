import { rest } from 'msw';
import { REGISTERED_PROMPT_CONTENT_TAG_KEY, REGISTERED_PROMPT_SOURCE_RUN_ID } from './utils';
import type { ModelAliasMap } from '../../types';
import type { KeyValueEntity } from '../../../common/types';

export const getMockedRegisteredPromptSetTagsResponse = (spyFn = jest.fn()) =>
  rest.post('/ajax-api/2.0/mlflow/registered-models/set-tag', (req, res, ctx) => {
    spyFn(req.body);
    return res(ctx.json({}));
  });

export const getMockedRegisteredPromptDeleteResponse = (spyFn = jest.fn()) =>
  rest.delete('/ajax-api/2.0/mlflow/registered-models/delete', (req, res, ctx) => {
    spyFn(req.body);
    return res(ctx.json({}));
  });

export const getMockedRegisteredPromptVersionSetTagsResponse = (spyFn = jest.fn()) =>
  rest.post('/ajax-api/2.0/mlflow/model-versions/set-tag', (req, res, ctx) => {
    spyFn(req.body);
    return res(ctx.json({}));
  });

export const getMockedRegisteredPromptCreateResponse = (spyFn = jest.fn()) =>
  rest.post('/ajax-api/2.0/mlflow/registered-models/create', (req, res, ctx) => {
    spyFn(req.body);
    return res(ctx.json({}));
  });

export const getMockedRegisteredPromptCreateVersionResponse = (spyFn = jest.fn()) =>
  rest.post('/ajax-api/2.0/mlflow/model-versions/create', (req, res, ctx) => {
    spyFn(req.body);
    return res(ctx.json({ model_version: { version: '1' } }));
  });

export const getMockedRegisteredPromptSourceRunResponse = (spyFn = jest.fn()) =>
  rest.get('/ajax-api/2.0/mlflow/runs/get', (req, res, ctx) => {
    const run_id = req.url.searchParams.get('run_id');
    return res(
      ctx.json({
        run: {
          data: {},
          info: {
            run_uuid: run_id,
            run_id: run_id,
            experiment_id: 'test_experiment_id',
            run_name: `${run_id}_name`,
          },
        },
      }),
    );
  });

export const getMockedRegisteredPromptsResponse = (n = 3) =>
  rest.get('/ajax-api/2.0/mlflow/registered-models/search', (req, res, ctx) => {
    return res(
      ctx.json({
        registered_models: [
          {
            name: 'prompt1',
            last_updated_timestamp: 1620000000000,
            tags: [{ key: 'some_tag', value: 'abc' }],
            latest_versions: [{ version: 3 }],
          },
          {
            name: 'prompt2',
            last_updated_timestamp: 1620000000000,
            tags: [{ key: 'another_tag', value: 'xyz' }],
            latest_versions: [{ version: 5 }],
          },
          {
            name: 'prompt3',
            last_updated_timestamp: 1620000000000,
            tags: [{ key: 'another_tag', value: 'xyz' }],
            latest_versions: [{ version: 7 }],
          },
        ].slice(0, n),
      }),
    );
  });

export const getFailedRegisteredPromptDetailsResponse = (status = 404) =>
  rest.get('/ajax-api/2.0/mlflow/registered-models/get', (req, res, ctx) => res(ctx.status(status)));

export const getMockedRegisteredPromptDetailsResponse = (name = 'prompt1', tags: KeyValueEntity[] = []) =>
  rest.get('/ajax-api/2.0/mlflow/registered-models/get', (req, res, ctx) => {
    const aliases: ModelAliasMap = [
      {
        alias: 'alias1',
        version: '1',
      },
      {
        alias: 'alias2',
        version: '2',
      },
      {
        alias: 'alias2',
        version: '3',
      },
    ];

    return res(
      ctx.json({
        registered_model: {
          name: 'prompt1',
          creation_timestamp: 1620000000000,
          last_updated_timestamp: 1620000000000,
          tags: [{ key: 'some_tag', value: 'abc' }, ...tags],
          aliases,
        },
      }),
    );
  });
export const getMockedRegisteredPromptVersionsResponse = (name = 'prompt1', n = 3, tags: KeyValueEntity[] = []) =>
  rest.get('/ajax-api/2.0/mlflow/model-versions/search', (req, res, ctx) => {
    return res(
      ctx.json({
        model_versions: [
          {
            name,
            version: 1,
            creation_timestamp: 1620000000000,
            last_updated_timestamp: 1620000000000,
            description: 'some commit message for version 1',
            tags: [
              { key: 'some_version_tag', value: 'abc' },
              { key: REGISTERED_PROMPT_CONTENT_TAG_KEY, value: 'content of prompt version 1' },
              ...tags,
            ],
          },
          {
            name,
            version: 2,
            creation_timestamp: 1620000000000,
            last_updated_timestamp: 1620000000000,
            description: 'some commit message for version 2',
            tags: [
              { key: 'another_version_tag', value: 'xyz' },
              { key: REGISTERED_PROMPT_CONTENT_TAG_KEY, value: 'content for prompt version 2' },
              ...tags,
            ],
          },
          {
            name,
            version: 3,
            creation_timestamp: 1620000000000,
            last_updated_timestamp: 1620000000000,
            description: 'some commit message for version 3',
            tags: [
              { key: 'another_version_tag', value: 'xyz' },
              { key: REGISTERED_PROMPT_CONTENT_TAG_KEY, value: 'text of prompt version 3' },
              ...tags,
            ],
          },
        ].slice(0, n),
      }),
    );
  });
