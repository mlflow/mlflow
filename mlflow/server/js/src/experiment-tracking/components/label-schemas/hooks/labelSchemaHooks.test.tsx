import { describe, jest, it, expect } from '@jest/globals';
import { act, renderHook, waitFor } from '@testing-library/react';
import { rest } from 'msw';
import React from 'react';

import { setupServer } from '../../../../common/utils/setup-msw';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClientProvider, QueryClient } from '../../../../shared/web-shared/query-client/queryClient';

import { useCreateLabelSchemaMutation } from './useCreateLabelSchemaMutation';
import { useDeleteLabelSchemaMutation } from './useDeleteLabelSchemaMutation';
import { useGetLabelSchemaByNameQuery } from './useGetLabelSchemaByNameQuery';
import { useGetLabelSchemaQuery } from './useGetLabelSchemaQuery';
import { useListLabelSchemasQuery } from './useListLabelSchemasQuery';
import { useUpdateLabelSchemaMutation } from './useUpdateLabelSchemaMutation';
import { useUpsertLabelSchemaMutation } from './useUpsertLabelSchemaMutation';
import type { LabelSchema } from '../types';

const mockSchema: LabelSchema = {
  schema_id: 'ls-test-1',
  experiment_id: '1',
  name: 'correctness',
  type: 'feedback',
  title: 'Is the answer correct?',
  enable_comment: true,
  input: { pass_fail: { positive_label: 'Correct', negative_label: 'Incorrect' } },
  created_at: 1000,
  last_updated_at: 1000,
};

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>
      <QueryClientProvider client={new QueryClient()}>{children}</QueryClientProvider>
    </DesignSystemProvider>
  </IntlProvider>
);

describe('label-schema hooks', () => {
  const server = setupServer();

  describe('useGetLabelSchemaQuery', () => {
    it('fetches a schema by id', async () => {
      server.use(
        rest.get('*/ajax-api/3.0/mlflow/label-schemas/get', (req, res, ctx) => {
          expect(req.url.searchParams.get('schema_id')).toEqual('ls-test-1');
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );

      const { result } = renderHook(() => useGetLabelSchemaQuery({ schemaId: 'ls-test-1' }), { wrapper });
      await waitFor(() => expect(result.current.isLoading).toEqual(false));
      expect(result.current.labelSchema?.schema_id).toEqual('ls-test-1');
    });

    it('does not fire when schemaId is empty', () => {
      const queryFn = jest.fn();
      server.use(
        rest.get('*/ajax-api/3.0/mlflow/label-schemas/get', (req, res, ctx) => {
          queryFn();
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );

      renderHook(() => useGetLabelSchemaQuery({ schemaId: '' }), { wrapper });
      expect(queryFn).not.toHaveBeenCalled();
    });
  });

  describe('useGetLabelSchemaByNameQuery', () => {
    it('passes experiment_id + name as query params', async () => {
      server.use(
        rest.get('*/ajax-api/3.0/mlflow/label-schemas/get-by-name', (req, res, ctx) => {
          expect(req.url.searchParams.get('experiment_id')).toEqual('1');
          expect(req.url.searchParams.get('name')).toEqual('correctness');
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );

      const { result } = renderHook(() => useGetLabelSchemaByNameQuery({ experimentId: '1', name: 'correctness' }), {
        wrapper,
      });
      await waitFor(() => expect(result.current.isLoading).toEqual(false));
      expect(result.current.labelSchema?.name).toEqual('correctness');
    });
  });

  describe('useListLabelSchemasQuery', () => {
    it('paginates through max_results + page_token', async () => {
      server.use(
        rest.get('*/ajax-api/3.0/mlflow/label-schemas/list', (req, res, ctx) => {
          expect(req.url.searchParams.get('experiment_id')).toEqual('1');
          expect(req.url.searchParams.get('max_results')).toEqual('2');
          expect(req.url.searchParams.get('page_token')).toEqual('pt');
          return res(ctx.json({ label_schemas: [mockSchema], next_page_token: 'nextpt' }));
        }),
      );

      const { result } = renderHook(
        () => useListLabelSchemasQuery({ experimentId: '1', maxResults: 2, pageToken: 'pt' }),
        { wrapper },
      );
      await waitFor(() => expect(result.current.isLoading).toEqual(false));
      expect(result.current.labelSchemas).toHaveLength(1);
      expect(result.current.nextPageToken).toEqual('nextpt');
    });

    it('omits max_results + page_token when undefined', async () => {
      server.use(
        rest.get('*/ajax-api/3.0/mlflow/label-schemas/list', (req, res, ctx) => {
          expect(req.url.searchParams.has('max_results')).toEqual(false);
          expect(req.url.searchParams.has('page_token')).toEqual(false);
          return res(ctx.json({ label_schemas: [] }));
        }),
      );
      const { result } = renderHook(() => useListLabelSchemasQuery({ experimentId: '1' }), { wrapper });
      await waitFor(() => expect(result.current.isLoading).toEqual(false));
      expect(result.current.labelSchemas).toHaveLength(0);
    });
  });

  describe('useCreateLabelSchemaMutation', () => {
    it('POSTs to /create with the params body', async () => {
      const requestBody = jest.fn();
      server.use(
        rest.post('*/ajax-api/3.0/mlflow/label-schemas/create', async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );

      const { result } = renderHook(() => useCreateLabelSchemaMutation(), { wrapper });
      await act(async () => {
        await result.current.createLabelSchemaAsync({
          experiment_id: '1',
          name: 'correctness',
          type: 'feedback',
          title: 'Is the answer correct?',
          input: { pass_fail: { positive_label: 'Correct', negative_label: 'Incorrect' } },
          enable_comment: true,
        });
      });
      // Asserts the full payload made it to the wire (not just a subset), so
      // a regression that drops `input` or mangles the oneof would be caught.
      // `instruction: undefined` is omitted by JSON.stringify so it doesn't
      // appear in the deserialized body.
      expect(requestBody).toHaveBeenCalledWith({
        experiment_id: '1',
        name: 'correctness',
        type: 'feedback',
        title: 'Is the answer correct?',
        input: { pass_fail: { positive_label: 'Correct', negative_label: 'Incorrect' } },
        enable_comment: true,
      });
    });
  });

  describe('useUpdateLabelSchemaMutation', () => {
    it('strips undefined sparse-update keys before sending', async () => {
      const requestBody = jest.fn();
      server.use(
        rest.patch('*/ajax-api/3.0/mlflow/label-schemas/update', async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );

      const { result } = renderHook(() => useUpdateLabelSchemaMutation(), { wrapper });
      await act(async () => {
        await result.current.updateLabelSchemaAsync({
          schema_id: 'ls-test-1',
          title: 'Updated title',
          // name, instruction, enable_comment, input intentionally omitted
        });
      });
      const body = requestBody.mock.calls[0][0] as Record<string, unknown>;
      expect(body).toEqual({ schema_id: 'ls-test-1', title: 'Updated title' });
    });

    it('forwards empty-string title as a real value (replaces stored field with "")', async () => {
      // The proto contract treats empty strings as set values (HasField=true)
      // that overwrite the stored field, NOT as "no-op". This pins the
      // documented behavior so a future cleanup that switches to truthy
      // guards (e.g., `if (params.title)`) would fail this test.
      const requestBody = jest.fn();
      server.use(
        rest.patch('*/ajax-api/3.0/mlflow/label-schemas/update', async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );
      const { result } = renderHook(() => useUpdateLabelSchemaMutation(), { wrapper });
      await act(async () => {
        await result.current.updateLabelSchemaAsync({ schema_id: 'ls-test-1', title: '' });
      });
      expect(requestBody.mock.calls[0][0]).toEqual({ schema_id: 'ls-test-1', title: '' });
    });

    it('forwards explicit false for enable_comment', async () => {
      const requestBody = jest.fn();
      server.use(
        rest.patch('*/ajax-api/3.0/mlflow/label-schemas/update', async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );
      const { result } = renderHook(() => useUpdateLabelSchemaMutation(), { wrapper });
      await act(async () => {
        await result.current.updateLabelSchemaAsync({ schema_id: 'ls-test-1', enable_comment: false });
      });
      expect(requestBody.mock.calls[0][0]).toEqual({ schema_id: 'ls-test-1', enable_comment: false });
    });
  });

  describe('useUpsertLabelSchemaMutation', () => {
    it('omits enable_comment when undefined (preserves replace semantic)', async () => {
      const requestBody = jest.fn();
      server.use(
        rest.post('*/ajax-api/3.0/mlflow/label-schemas/upsert', async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json({ label_schema: mockSchema }));
        }),
      );
      const { result } = renderHook(() => useUpsertLabelSchemaMutation(), { wrapper });
      await act(async () => {
        await result.current.upsertLabelSchemaAsync({
          experiment_id: '1',
          name: 'correctness',
          type: 'feedback',
          title: 'Is the answer correct?',
          input: { pass_fail: { positive_label: 'Correct', negative_label: 'Incorrect' } },
        });
      });
      const body = requestBody.mock.calls[0][0] as Record<string, unknown>;
      expect(body).not.toHaveProperty('enable_comment');
      expect(body).not.toHaveProperty('instruction');
    });
  });

  describe('useDeleteLabelSchemaMutation', () => {
    it('sends DELETE with schema_id in body', async () => {
      const requestBody = jest.fn();
      server.use(
        rest.delete('*/ajax-api/3.0/mlflow/label-schemas/delete', async (req, res, ctx) => {
          requestBody(await req.json());
          return res(ctx.json({}));
        }),
      );
      const { result } = renderHook(() => useDeleteLabelSchemaMutation(), { wrapper });
      await act(async () => {
        await result.current.deleteLabelSchemaAsync({ schema_id: 'ls-test-1' });
      });
      expect(requestBody).toHaveBeenCalledWith({ schema_id: 'ls-test-1' });
    });
  });
});
