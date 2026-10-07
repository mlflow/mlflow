function removeHtmlComments(text) {
  return text.replace(/<!--[\s\S]*?-->/g, "");
}

function removeControlCharacters(text) {
  return text.replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g, "");
}

function removeInvisibleCharacters(text) {
  return text.replace(
    /[\u00ad\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\u{e0000}-\u{e007f}]/gu,
    ""
  );
}

function isEscaped(text, index) {
  let backslashes = 0;
  while (index > backslashes && text[index - backslashes - 1] === "\\") {
    backslashes++;
  }
  return backslashes % 2 === 1;
}

function getLinkDestinationRanges(text) {
  const ranges = [];
  const links =
    /\[[^\]\n]*\]\(\s*(<[^>\s]*>|(?:\\.|[^()\s]|\([^()\s]*\))*)(?:\s+(?:"[^"]*"|'[^']*'|\([^)]*\)))?\s*\)/g;
  for (const match of text.matchAll(links)) {
    let start = match.index + match[0].indexOf("](") + 2;
    while (/[ \t\r\n]/.test(text[start] || "")) {
      start++;
    }
    ranges.push([start, start + match[1].length]);
  }
  return ranges;
}

function neutralizeProse(text) {
  const ranges = getLinkDestinationRanges(text);
  for (const match of text.matchAll(/(?:[A-Za-z][A-Za-z0-9+.-]*:\/\/|www\.)[^\s<>]+/g)) {
    ranges.push([match.index, match.index + match[0].length]);
  }
  ranges.sort((left, right) => left[0] - right[0]);
  const protectedRanges = [];
  for (const [start, end] of ranges) {
    const previous = protectedRanges[protectedRanges.length - 1];
    if (previous && start <= previous[1]) {
      previous[1] = Math.max(previous[1], end);
    } else {
      protectedRanges.push([start, end]);
    }
  }

  let rangeIndex = 0;
  return text.replace(
    /@[A-Za-z0-9][A-Za-z0-9_-]*(?:\/[A-Za-z0-9][A-Za-z0-9._-]*)?/g,
    (mention, offset) => {
      while (rangeIndex < protectedRanges.length && protectedRanges[rangeIndex][1] <= offset) {
        rangeIndex++;
      }
      const range = protectedRanges[rangeIndex];
      const previous = text[offset - 1];
      if (
        (previous && /[A-Za-z0-9_]/.test(previous)) ||
        (range && offset >= range[0] && offset < range[1])
      ) {
        return mention;
      }
      return `@ ${mention.slice(1)}`;
    }
  );
}

function neutralizeInlineCode(text) {
  return text
    .split(/(\r?\n[ \t]*\r?\n)/)
    .map((paragraph) => {
      const delimiters = Array.from(paragraph.matchAll(/`+/g))
        .filter((match) => !isEscaped(paragraph, match.index))
        .map((match) => ({
          start: match.index,
          end: match.index + match[0].length,
          length: match[0].length,
        }));
      const nextByDelimiter = [];
      const lastByDelimiter = new Map();
      for (let index = delimiters.length - 1; index >= 0; index--) {
        nextByDelimiter[index] = lastByDelimiter.get(delimiters[index].length);
        lastByDelimiter.set(delimiters[index].length, index);
      }

      let result = "";
      let proseStart = 0;
      for (let index = 0; index < delimiters.length; index++) {
        const closingIndex = nextByDelimiter[index];
        if (closingIndex === undefined) {
          continue;
        }
        const opening = delimiters[index];
        const closing = delimiters[closingIndex];
        result += neutralizeProse(paragraph.slice(proseStart, opening.start));
        result += paragraph.slice(opening.start, closing.end);
        proseStart = closing.end;
        index = closingIndex;
      }
      return result + neutralizeProse(paragraph.slice(proseStart));
    })
    .join("");
}

function neutralizeMentions(text) {
  const parts = [];
  let prose = "";
  let fence = null;
  const flushProse = () => {
    parts.push(neutralizeInlineCode(prose));
    prose = "";
  };

  for (const line of text.split(/(?<=\n)/)) {
    if (fence) {
      parts.push(line);
      const closing = line.match(
        /^(?: {0,3}>[ \t]?)*(?: {0,3}(?:[-+*]|\d+[.)])[ \t]+)? {0,3}(`+|~+)[ \t]*\r?(?:\n)?$/
      );
      if (closing && closing[1][0] === fence[0] && closing[1].length >= fence.length) {
        fence = null;
      }
      continue;
    }

    const opening = line.match(
      /^(?: {0,3}>[ \t]?)*(?: {0,3}(?:[-+*]|\d+[.)])[ \t]+)? {0,3}(`{3,}|~{3,})([^\r\n]*)(?:\r?\n)?$/
    );
    if (opening && (opening[1][0] !== "`" || !opening[2].includes("`"))) {
      flushProse();
      fence = opening[1];
      parts.push(line);
    } else {
      prose += line;
    }
  }
  flushProse();
  return parts.join("");
}

function sanitizeInput(text) {
  return removeHtmlComments(removeInvisibleCharacters(removeControlCharacters(text)));
}

function sanitizeOutput(text) {
  return neutralizeMentions(
    removeHtmlComments(removeInvisibleCharacters(removeControlCharacters(text)))
  );
}

module.exports = {
  removeHtmlComments,
  removeControlCharacters,
  removeInvisibleCharacters,
  neutralizeMentions,
  sanitizeInput,
  sanitizeOutput,
};

if (require.main === module) {
  const sanitizers = new Map([
    ["input", sanitizeInput],
    ["output", sanitizeOutput],
  ]);
  const sanitizer = sanitizers.get(process.argv[2]);
  if (process.argv.length !== 3 || !sanitizer) {
    console.error("Usage: node sanitize.js <input|output>");
    process.exit(1);
  }

  let input = "";
  process.stdin.setEncoding("utf8");
  process.stdin.on("data", (chunk) => {
    input += chunk;
  });
  process.stdin.on("end", () => {
    process.stdout.write(sanitizer(input));
  });
}
