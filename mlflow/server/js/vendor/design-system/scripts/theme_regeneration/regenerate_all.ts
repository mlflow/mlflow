import { execSync } from 'child_process';
import path from 'path';

const scriptsToRun = [
  'regenerate_primitive_color_list.ts',
  'regenerate_semantic_color_lists.ts',
  'regenerate_color_validity_enums.ts',
  'regenerate_border_radii_list.ts',
];

// eslint-disable-next-line no-console -- TODO(FEINF-3587)
console.log('Regenerating theme files...');

try {
  for (const script of scriptsToRun) {
    const scriptPath = path.join(__dirname, script);
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    console.log(`Running ${script}...`);
    execSync(`swc-node ${scriptPath}`, { stdio: 'inherit', env: { ...process.env } });
  }

  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('All theme regeneration scripts completed successfully.');
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('Running Prettier on generated files...');
  const generatedDir = process.env.THEME_GENERATED_DIR || './src/theme/_generated';
  execSync(`databricks-prettier ${generatedDir} --write`, { stdio: 'inherit' });
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.log('Theme regeneration process completed.');
} catch (error) {
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.error('An error occurred during theme regeneration:', error);
  process.exit(1);
}
