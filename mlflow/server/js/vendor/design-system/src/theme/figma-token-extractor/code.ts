console.clear();

function exportToJSON() {
  const collections: VariableCollection[] = figma.variables.getLocalVariableCollections();
  const files: any[] = [];
  collections.forEach((collection) => files.push(...processCollection(collection)));
  figma.ui.postMessage({ type: 'EXPORT_RESULT', files });
}

function processCollection({ name, modes, variableIds }: { name: string; modes: any[]; variableIds: string[] }) {
  const files: any[] = [];
  modes.forEach((mode) => {
    const snakeCase = (string: string) => {
      return string
        .replace(/\W+/g, ' ')
        .split(/ |\B(?=[A-Z])/)
        .map((word) => word.toLowerCase())
        .join('_');
    };

    const file = {
      fileName: `${snakeCase(name)}${mode.name === 'Mode' ? '' : `.${snakeCase(mode.name)}`}.json`,
      body: {},
    };
    variableIds.forEach((variableId) => {
      const variable = figma.variables.getVariableById(variableId);

      if (!variable) {
        return;
      }

      const { name, resolvedType, valuesByMode } = variable;
      const value = valuesByMode[mode.modeId];
      if (value !== undefined && ['COLOR', 'FLOAT'].includes(resolvedType)) {
        let obj: any = file.body;
        name.split('/').forEach((groupName: string) => {
          // * means "new/published" in figma but we don't care about it in code
          if (groupName.endsWith('*')) {
            groupName = groupName.slice(0, -1);
          }

          // Strip emojis from the groupName
          groupName = groupName.replace(
            /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F1E0}-\u{1F1FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu,
            '',
          );

          obj[groupName] = obj[groupName] || {};
          obj = obj[groupName];
        });
        obj.$type = resolvedType === 'COLOR' ? 'color' : 'number';
        if ((value as any).type === 'VARIABLE_ALIAS') {
          const aliasValue = figma.variables.getVariableById((value as any).id);

          if (aliasValue) {
            const collection = aliasValue?.variableCollectionId
              ? figma.variables.getVariableCollectionById(aliasValue.variableCollectionId)
              : null;

            // * means "new/published" in figma but we don't care about it in code
            const aliasName = aliasValue.name.endsWith('*') ? aliasValue.name.slice(0, -1) : aliasValue.name;

            obj.$value = `{${collection ? `${collection.name}/` : ''}${aliasName}}`;
          }
        } else {
          if (resolvedType === 'COLOR' && typeof value === 'object' && value !== null) {
            if ('r' in value && 'g' in value && 'b' in value) {
              obj.$value = rgbToHex({
                r: value.r,
                g: value.g,
                b: value.b,
                a: 'a' in value ? value.a : 1,
              });
            }
          } else {
            obj.$value = value;
          }
        }
      }
    });
    files.push(file);
  });
  return files;
}

figma.ui.onmessage = (e: any) => {
  console.log('code received message', e);
  if (e.type === 'EXPORT') {
    exportToJSON();
  }
};

if (figma.command === 'export') {
  figma.showUI(__uiFiles__['export'], {
    width: 500,
    height: 500,
    themeColors: true,
  });
}

function rgbToHex({ r, g, b, a }: { r: number; g: number; b: number; a: number }): string {
  if (a !== 1) {
    return `rgba(${[r, g, b].map((n) => Math.round(n * 255)).join(', ')}, ${a.toFixed(4)})`;
  }
  const toHex = (value: number): string => {
    const hex = Math.round(value * 255).toString(16);
    return hex.padStart(2, '0');
  };

  const hex = [toHex(r), toHex(g), toHex(b)].join('');
  return `#${hex}`;
}
