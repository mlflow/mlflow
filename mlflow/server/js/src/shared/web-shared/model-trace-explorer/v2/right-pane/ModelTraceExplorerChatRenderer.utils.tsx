export const CONTENT_TRUNCATION_LIMIT = 300;

// Matches markdown image syntax: ![alt text](url)
const MARKDOWN_IMAGE_REGEX = /!\[[^\]]*\]\([^)]+\)/g;

// Returns content length excluding markdown image blocks.
export const getDisplayLength = (content: string): number => {
  let displayLength = 0;
  let lastIndex = 0;

  const regex = new RegExp(MARKDOWN_IMAGE_REGEX.source, 'g');
  let match: RegExpExecArray | null;

  while ((match = regex.exec(content)) !== null) {
    displayLength += match.index - lastIndex;
    lastIndex = match.index + match[0].length;
  }

  displayLength += content.length - lastIndex;
  return displayLength;
};

// Truncates content without breaking markdown image syntax.
export const truncatePreservingImages = (content: string, limit: number): string => {
  let result = '';
  let displayLength = 0;
  let lastIndex = 0;

  const regex = new RegExp(MARKDOWN_IMAGE_REGEX.source, 'g');
  let match: RegExpExecArray | null;

  while ((match = regex.exec(content)) !== null) {
    const textBefore = content.slice(lastIndex, match.index);
    const remainingLimit = limit - displayLength;

    if (textBefore.length > remainingLimit) {
      result += textBefore.slice(0, remainingLimit) + '...';
      return result;
    }

    result += textBefore + match[0];
    displayLength += textBefore.length;
    lastIndex = match.index + match[0].length;
  }

  const remaining = content.slice(lastIndex);
  const remainingLimit = limit - displayLength;
  if (remaining.length > remainingLimit) {
    result += remaining.slice(0, remainingLimit) + '...';
  } else {
    result += remaining;
  }

  return result;
};
