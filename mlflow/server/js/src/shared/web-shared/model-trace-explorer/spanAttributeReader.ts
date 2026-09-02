import type { ModelTraceSpan } from './ModelTrace.types';

/**
 * Extract an attribute value from span attributes.
 * V3 traces have flat objects: { 'mlflow.spanInputs': '{"x": 10}' }
 * V4 traces have arrays: [{ key: 'mlflow.spanInputs', value: { string_value: '{"x":10}' } }]
 */
export const getSpanAttribute = (
  attributes: ModelTraceSpan['attributes'],
  attributeKey: string,
): string | undefined => {
  if (!attributes) {
    return undefined;
  }

  // V3: attributes is a flat object
  if (!Array.isArray(attributes)) {
    return attributes[attributeKey];
  }

  // V4: attributes is an array - find the matching key
  const attribute = attributes.find((attr) => attr.key === attributeKey);
  if (!attribute?.value) {
    return undefined;
  }

  // V4 values are typed - return whichever type is present
  return attribute.value.string_value ?? attribute.value.int_value ?? attribute.value.bool_value;
};
