import { Converter } from "showdown";

// Use Github-like Markdown (i.e. support for tasklists, strikethrough,
// simple line breaks, code blocks, emojis)
const DEFAULT_MARKDOWN_FLAVOR = 'github';

export const getConverter = () => {
  const converter = new Converter();
  converter.setFlavor(DEFAULT_MARKDOWN_FLAVOR);
  return converter;
};
