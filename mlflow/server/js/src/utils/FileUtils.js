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
  ['txt', 'py', 'js', 'yaml', 'json', 'csv', 'md', 'rst', 'MLmodel', 'MLproject']);
