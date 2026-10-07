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

function sanitizeInput(text) {
  return removeInvisibleCharacters(removeControlCharacters(removeHtmlComments(text)));
}

module.exports = {
  removeHtmlComments,
  removeControlCharacters,
  removeInvisibleCharacters,
  sanitizeInput,
};

if (require.main === module) {
  const sanitizers = new Map([["input", sanitizeInput]]);
  const sanitizer = sanitizers.get(process.argv[2]);
  if (process.argv.length !== 3 || !sanitizer) {
    console.error("Usage: node sanitize.js input");
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
