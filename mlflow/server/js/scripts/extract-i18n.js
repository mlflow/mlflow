const { extract } = require('@formatjs/cli');
const fs = require('fs');
const { sync: globSync } = require('fast-glob');
const { ArgumentParser } = require('argparse');

const OUT_FILE = './src/lang/default/en.json';
const FILE_PATTERN = 'src/**/*.(j|t)s?(x)';
const FILE_IGNORE_PATTERNS = ['**/*.d.ts', '**/*.(j|t)est.(j|t)s?(x)'];
const EXTRACT_OPTS = {
  idInterpolationPattern: '[sha512:contenthash:base64:6]',
  removeDefaultMessage: false,
  extractFromFormatMessageCall: true,
  ast: true,
};

const parser = new ArgumentParser({
  description: 'Databricks i18n Key Extractor',
  add_help: true,
});
parser.add_argument('-l', '--lint', {
  action: 'store_true',
  help: 'Only report if the extracted keys need to be updated without actually updating.',
});

async function main(args) {
  const files = globSync(FILE_PATTERN, { ignore: FILE_IGNORE_PATTERNS });

  const extractedMessages = JSON.parse(await extract(files, EXTRACT_OPTS));
  console.log(`Extracted ${Object.keys(extractedMessages).length} keys from ${files.length} files`);

  if (args.lint) {
    let existingMessages = {};
    if (fs.existsSync(OUT_FILE)) {
      existingMessages = JSON.parse(fs.readFileSync(OUT_FILE, 'utf8'));
      console.log(`There are ${Object.keys(existingMessages).length} existing keys in ${OUT_FILE}`);
    } else {
      console.log(`${OUT_FILE} does not exist`);
    }

    const extractedKeys = Object.keys(extractedMessages);
    const existingKeys = new Set(Object.keys(existingMessages));

    if (extractedKeys.length === existingKeys.size && extractedKeys.every((key) => existingKeys.has(key))) {
      console.log('Extracted keys are up-to-date.');
      process.exit(0);
    } else {
      console.log('Mismatch detected between extracted keys. Run without --lint to update.');
      process.exit(1);
    }
  } else {
    fs.writeFileSync(OUT_FILE, JSON.stringify(extractedMessages, null, 2));
  }
}

if (require.main === module) {
  main(parser.parse_args());
}
