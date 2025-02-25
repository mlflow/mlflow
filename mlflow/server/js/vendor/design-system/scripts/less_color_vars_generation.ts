import { writeFileSync } from 'fs';
import path from 'path';

import { primitiveColors } from '../src/theme/_generated/PrimitiveColors';
import { darkColorList } from '../src/theme/_generated/SemanticColors-Dark';
import { lightColorList } from '../src/theme/_generated/SemanticColors-Light';
import { deprecatedSemanticColorsDark, deprecatedSemanticColorsLight } from '../src/theme/colors';

const designSystemDirectory = path.resolve(__dirname, '..');

writeFileSync(`${designSystemDirectory}/dist/dubois-colors.less`, convertColorsToLess());

function convertColorsToLess() {
  const colorLists = {
    primitives: primitiveColors,
    light: { ...lightColorList, ...deprecatedSemanticColorsLight },
    dark: { ...darkColorList, ...deprecatedSemanticColorsDark },
  };
  let output = '// Auto-generated file\n\n';

  Object.entries(colorLists).forEach(([colorGroup, colorList]) => {
    Object.entries(colorList).forEach(([key, value]) => {
      const lessVariableName = createVariableName(
        key,
        colorGroup === 'light' || colorGroup === 'dark' ? colorGroup : undefined,
      );
      output += `${lessVariableName}: ${value};\n`;
    });
  });

  return output;
}

function createVariableName(colorName: string, mode?: 'light' | 'dark') {
  if (mode) {
    return `@dubois-${mode}-${colorName}`;
  }
  return `@dubois-${colorName}`;
}
