const fs = require('fs');
const path = require('path');

const aliases = require('./../../../assets/icons/aliases.json');

const iconFolder = path.resolve(__dirname, '..');
const iconComponentsFolder = path.join(iconFolder, '__generated', 'icons');
const outputFolder = path.join(iconFolder, '__generated');

// eslint-disable-next-line no-console -- TODO(FEINF-3587)
console.log(`Generating icon story in ${iconComponentsFolder}...`);

const templateFileContent = fs.readFileSync(path.join(iconFolder, 'generator', 'IconsListTemplate.tsx.template'));

const files = fs.readdirSync(iconComponentsFolder);
const names = files
  .map((file) => path.parse(file).name)
  .filter((name) => !['index', 'stories', '.v1', '.v2'].some((excluded) => name.includes(excluded)));

function getIconData(name) {
  return `{
    Icon: Icons.${name},
    name: "${name}",
    aliases: ${JSON.stringify(aliases[name] ?? [])}
  }`;
}

const listComponentContent = templateFileContent
  .toString()
  .replace('/* CONTENT HERE */', names.map(getIconData).join(','));

const listComponentPath = path.join(outputFolder, 'stories', 'IconsList.tsx');
fs.writeFileSync(listComponentPath, listComponentContent);

// eslint-disable-next-line no-console -- TODO(FEINF-3587)
console.log(`Wrote list to: ${listComponentPath}`);
