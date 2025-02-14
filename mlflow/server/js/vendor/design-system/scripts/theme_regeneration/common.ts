import _ from 'lodash';

export type token = {
  $type: string;
  $value: string | number;
};

export type tokenJson = {
  [tokenName: string]: token | tokenJson;
};

function formatColorToken(token: string): string {
  // Force consistent styling for hex colors
  if (token.startsWith('#')) {
    return token.toUpperCase();
  }

  // Splices primitive path into token name; we only care about the first and last parts today. I.e. Primitives.Blue.Blue100 -> primitives.blue100
  if (token.startsWith('{') && token.endsWith('}')) {
    const parts = token.replace(/[{}]/g, '').split('/');
    return `${_.camelCase(parts[0])}.${_.camelCase(parts[parts.length - 1])}`;
  }
  return token;
}

export const flattenTokens = (tokens: tokenJson): { [key: string]: string } => {
  return Object.keys(tokens).reduce((acc, key) => {
    const token = tokens[key];

    const newKey = _.camelCase(key);

    // Skipping numbers for now.
    if (token.$type === 'number') {
      return acc;
    }

    if ('$value' in token) {
      acc[newKey] = formatColorToken(token.$value as string);
    } else {
      Object.assign(acc, flattenTokens(token));
    }

    return acc;
  }, {});
};

export const stringifyTokens = (tokens: { [key: string]: string }): string => {
  return `{
    ${Object.keys(tokens)
      .sort()
      .map((key) => {
        // Converts strings into references; will point to an ES import in the output.
        if (tokens[key].includes('primitives.')) {
          return `"${key}": ${tokens[key]}`;
        } else {
          return `"${key}": "${tokens[key]}"`;
        }
      })
      .join(',\n')}
  }`;
};
