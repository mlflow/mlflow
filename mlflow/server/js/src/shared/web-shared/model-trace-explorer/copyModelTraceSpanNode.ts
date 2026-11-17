/** Copy tree structure without reading deferred payload getters. */
export const copyModelTraceSpanNode = <T>(node: T, updates: Partial<T>): T =>
  Object.create(Object.getPrototypeOf(node), {
    ...Object.getOwnPropertyDescriptors(node),
    ...Object.getOwnPropertyDescriptors(updates),
  }) as T;
