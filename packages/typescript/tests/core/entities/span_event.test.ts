import { SpanEvent } from '../../../src/core/entities/span_event';

describe('SpanEvent', () => {
  describe('constructor', () => {
    it('should create a span event with all parameters', () => {
      const timestamp = BigInt(Date.now()) * 1_000_000n; // nanoseconds
      const attributes = {
        key1: 'value1',
        key2: 42,
        key3: true,
        key4: ['a', 'b', 'c']
      };

      const event = new SpanEvent({
        name: 'test_event',
        timestamp,
        attributes
      });

      expect(event.name).toBe('test_event');
      expect(event.timestamp).toBe(timestamp);
      expect(event.attributes).toEqual(attributes);
    });
  });

  describe('fromException', () => {
    it('should create a span event from a basic error', () => {
      const error = new Error('Test error message');
      const event = SpanEvent.fromException(error);

      expect(event.name).toBe('exception');
      expect(event.attributes['exception.message']).toBe('Test error message');
      expect(event.attributes['exception.type']).toBe('Error');
      expect(event.attributes['exception.stacktrace']).toContain('Test error message');
    });
  });

  describe('toJson round-trip serialization', () => {
    it('should serialize and maintain all properties', () => {
      const originalEvent = new SpanEvent({
        name: 'test_event',
        timestamp: 1234567890000n,
        attributes: {
          string_attr: 'test_value',
          number_attr: 42,
          boolean_attr: true,
          string_array: ['a', 'b', 'c'],
          number_array: [1, 2, 3],
          boolean_array: [true, false, true]
        }
      });

      const json = originalEvent.toJson();

      // Verify JSON structure
      expect(json).toEqual({
        name: 'test_event',
        timestamp: 1234567890000n,
        attributes: {
          string_attr: 'test_value',
          number_attr: 42,
          boolean_attr: true,
          string_array: ['a', 'b', 'c'],
          number_array: [1, 2, 3],
          boolean_array: [true, false, true]
        }
      });

      // Create new event from JSON data
      const recreatedEvent = new SpanEvent({
        name: json.name as string,
        timestamp: json.timestamp as bigint,
        attributes: json.attributes as Record<string, any>
      });

      // Verify round-trip preservation
      expect(recreatedEvent.name).toBe(originalEvent.name);
      expect(recreatedEvent.timestamp).toBe(originalEvent.timestamp);
      expect(recreatedEvent.attributes).toEqual(originalEvent.attributes);
      expect(recreatedEvent.toJson()).toEqual(originalEvent.toJson());
    });

    it('should handle events with minimal properties', () => {
      const originalEvent = new SpanEvent({
        name: 'minimal_event'
      });

      const json = originalEvent.toJson();

      expect(json.name).toBe('minimal_event');
      expect(json.timestamp).toBeGreaterThan(0);
      expect(json.attributes).toEqual({});

      // Recreate and verify
      const recreatedEvent = new SpanEvent({
        name: json.name as string,
        timestamp: json.timestamp as bigint,
        attributes: json.attributes as Record<string, any>
      });

      expect(recreatedEvent.name).toBe(originalEvent.name);
      expect(recreatedEvent.timestamp).toBe(originalEvent.timestamp);
      expect(recreatedEvent.attributes).toEqual(originalEvent.attributes);
    });
  });
});
