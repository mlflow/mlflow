import { readFileSync } from 'node:fs';
import { join } from 'node:path';

import { ANTHROPIC_MODEL_RATES } from '../src/anthropicPricing';
import { catalogToRates } from '../src/modelCatalog';

const CATALOG_PATH = join(__dirname, '../../../../../mlflow/utils/model_catalog/anthropic.json');

describe('anthropicPricing.ts (generated)', () => {
  it('matches mlflow/utils/model_catalog/anthropic.json (run `npm run sync:pricing` to fix)', () => {
    const catalog = JSON.parse(readFileSync(CATALOG_PATH, 'utf8'));
    expect(ANTHROPIC_MODEL_RATES).toEqual(catalogToRates(catalog));
  });
});
