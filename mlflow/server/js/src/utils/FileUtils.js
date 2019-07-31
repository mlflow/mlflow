export const getBasename = (path) => {
  const parts = path.split('/');
  return parts[parts.length - 1];
};

export const getExtension = (path) => {
  const parts = path.split(/[./]/);
  return parts[parts.length - 1];
};

export const IMAGE_EXTENSIONS = new Set(['jpg', 'bmp', 'jpeg', 'png', 'gif', 'svg']);
export const TEXT_EXTENSIONS = new Set(
  ['txt', 'log', 'py', 'js', 'yaml', 'yml', 'json', 'csv', 'tsv',
    'md', 'rst', 'MLmodel', 'MLproject', 'jsonnet']);
