import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { OPENAI_MODEL_RATES } from '../src/openaiPricing';
import { catalogToRates } from '../src/modelCatalog';

const CATALOG_PATH = join(__dirname, '../../../../../mlflow/utils/model_catalog/openai.json');

describe('openaiPricing.ts (generated)', () => {
  it('matches mlflow/utils/model_catalog/openai.json (run `npm run sync:pricing` to fix)', () => {
    const catalog = JSON.parse(readFileSync(CATALOG_PATH, 'utf8'));
    expect(OPENAI_MODEL_RATES).toEqual(catalogToRates(catalog));
  });
});
