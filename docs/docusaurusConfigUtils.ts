export function postProcessSidebar(items) {
  // Remove items with customProps.hide set to true
  return items.filter((item) => item.customProps?.hide !== true);
}

export function apiReferencePrefix(): string {
  let prefix = process.env.API_REFERENCE_PREFIX || 'https://mlflow.org/docs/latest/';
  if (!prefix.startsWith('http')) {
    throw new Error(`API reference prefix must start with http, got ${prefix}`);
  }

  if (!prefix.endsWith('/')) {
    prefix += '/';
  }
  return prefix;
}
