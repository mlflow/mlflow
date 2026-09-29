import type { DecisionTranslator } from '../Decision.types';
import { typeSafeSystemOneTranslator } from './TypeSafeSystemOneTranslator';

export const decisionTranslators: readonly DecisionTranslator[] = [typeSafeSystemOneTranslator];
