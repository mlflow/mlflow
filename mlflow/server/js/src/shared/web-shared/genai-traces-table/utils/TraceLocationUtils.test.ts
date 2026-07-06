import { describe, it, expect } from '@jest/globals';

import {
  createTraceLocationForUCTablePrefix,
  createTraceLocationForDestinationPath,
  isTablePrefixDestinationPath,
  doesTraceSupportV4API,
} from './TraceLocationUtils';

describe('TraceLocationUtils', () => {
  describe('createTraceLocationForUCTablePrefix', () => {
    it('parses a 3-part path into a UC_TABLE_PREFIX location', () => {
      const result = createTraceLocationForUCTablePrefix('my_catalog.my_schema.my_prefix');
      expect(result).toEqual({
        type: 'UC_TABLE_PREFIX',
        uc_table_prefix: {
          catalog_name: 'my_catalog',
          schema_name: 'my_schema',
          table_prefix: 'my_prefix',
        },
      });
    });
  });

  describe('createTraceLocationForDestinationPath', () => {
    it('returns UC_SCHEMA for a 2-part path', () => {
      const result = createTraceLocationForDestinationPath('catalog.schema');
      expect(result).toEqual({
        type: 'UC_SCHEMA',
        uc_schema: {
          catalog_name: 'catalog',
          schema_name: 'schema',
        },
      });
    });

    it('returns UC_TABLE_PREFIX for a 3-part path', () => {
      const result = createTraceLocationForDestinationPath('catalog.schema.prefix');
      expect(result).toEqual({
        type: 'UC_TABLE_PREFIX',
        uc_table_prefix: {
          catalog_name: 'catalog',
          schema_name: 'schema',
          table_prefix: 'prefix',
        },
      });
    });
  });

  describe('isTablePrefixDestinationPath', () => {
    it('returns true for a 3-part path', () => {
      expect(isTablePrefixDestinationPath('catalog.schema.prefix')).toBe(true);
    });

    it('returns false for a 2-part path', () => {
      expect(isTablePrefixDestinationPath('catalog.schema')).toBe(false);
    });

    it('returns false for a 1-part path', () => {
      expect(isTablePrefixDestinationPath('catalog')).toBe(false);
    });
  });

  describe('doesTraceSupportV4API', () => {
    it('returns true for UC_SCHEMA traces', () => {
      const traceInfo = {
        trace_id: 'test-trace-id',
        request_time: '2025-01-01T00:00:00Z',
        state: 'OK' as const,
        tags: {},
        trace_location: {
          type: 'UC_SCHEMA' as const,
          uc_schema: { catalog_name: 'cat', schema_name: 'sch' },
        },
      };
      expect(doesTraceSupportV4API(traceInfo)).toBe(true);
    });

    it('returns true for UC_TABLE_PREFIX traces', () => {
      const traceInfo = {
        trace_id: 'test-trace-id',
        request_time: '2025-01-01T00:00:00Z',
        state: 'OK' as const,
        tags: {},
        trace_location: {
          type: 'UC_TABLE_PREFIX' as const,
          uc_table_prefix: { catalog_name: 'cat', schema_name: 'sch', table_prefix: 'pfx' },
        },
      };
      expect(doesTraceSupportV4API(traceInfo)).toBe(true);
    });

    it('returns false for MLFLOW_EXPERIMENT traces', () => {
      const traceInfo = {
        trace_id: 'test-trace-id',
        request_time: '2025-01-01T00:00:00Z',
        state: 'OK' as const,
        tags: {},
        trace_location: {
          type: 'MLFLOW_EXPERIMENT' as const,
          mlflow_experiment: { experiment_id: '123' },
        },
      };
      expect(doesTraceSupportV4API(traceInfo)).toBe(false);
    });

    it('returns false for undefined traceInfo', () => {
      expect(doesTraceSupportV4API(undefined)).toBe(false);
    });
  });
});
