import type { Model } from './routeConstants';

export const groupModelsByFamily = (models: Model[]) => {
  const groups: Record<string, Model[]> = {};
  const ungrouped: Model[] = [];

  models.forEach((model) => {
    const modelId = model.id.toLowerCase();

    // OpenAI grouping patterns
    if (modelId.includes('gpt-5')) {
      if (!groups['GPT-5']) groups['GPT-5'] = [];
      groups['GPT-5'].push(model);
    } else if (modelId.includes('gpt-4o')) {
      if (!groups['GPT-4o']) groups['GPT-4o'] = [];
      groups['GPT-4o'].push(model);
    } else if (modelId.includes('gpt-4-turbo') || (modelId.includes('gpt-4') && modelId.includes('turbo'))) {
      if (!groups['GPT-4 Turbo']) groups['GPT-4 Turbo'] = [];
      groups['GPT-4 Turbo'].push(model);
    } else if (modelId.includes('gpt-4')) {
      if (!groups['GPT-4']) groups['GPT-4'] = [];
      groups['GPT-4'].push(model);
    } else if (modelId.includes('gpt-3.5')) {
      if (!groups['GPT-3.5']) groups['GPT-3.5'] = [];
      groups['GPT-3.5'].push(model);
    } else if (
      modelId.startsWith('o1') ||
      modelId.includes('-o1') ||
      modelId.includes('o1-') ||
      modelId.startsWith('o2') ||
      modelId.includes('-o2') ||
      modelId.includes('o2-') ||
      modelId.startsWith('o3') ||
      modelId.includes('-o3') ||
      modelId.includes('o3-') ||
      modelId.startsWith('o4') ||
      modelId.includes('-o4') ||
      modelId.includes('o4-')
    ) {
      if (!groups['Reasoning Models']) groups['Reasoning Models'] = [];
      groups['Reasoning Models'].push(model);
      // Anthropic grouping patterns
    } else if (modelId.includes('claude') && modelId.includes('sonnet')) {
      if (!groups['Claude Sonnet']) groups['Claude Sonnet'] = [];
      groups['Claude Sonnet'].push(model);
    } else if (modelId.includes('claude') && modelId.includes('opus')) {
      if (!groups['Claude Opus']) groups['Claude Opus'] = [];
      groups['Claude Opus'].push(model);
    } else if (modelId.includes('claude') && modelId.includes('haiku')) {
      if (!groups['Claude Haiku']) groups['Claude Haiku'] = [];
      groups['Claude Haiku'].push(model);
      // Google grouping patterns
    } else if (modelId.includes('gemini') && modelId.includes('pro')) {
      if (!groups['Gemini Pro']) groups['Gemini Pro'] = [];
      groups['Gemini Pro'].push(model);
    } else if (modelId.includes('gemini') && modelId.includes('flash')) {
      if (!groups['Gemini Flash']) groups['Gemini Flash'] = [];
      groups['Gemini Flash'].push(model);
    } else {
      ungrouped.push(model);
    }
  });

  // Define explicit group ordering
  const groupOrder = [
    'GPT-5',
    'GPT-4o',
    'GPT-4 Turbo',
    'GPT-4',
    'GPT-3.5',
    'Reasoning Models',
    'Claude Sonnet',
    'Claude Opus',
    'Claude Haiku',
    'Gemini Pro',
    'Gemini Flash',
  ];

  // Convert to array of groups with explicit ordering
  const result = groupOrder
    .filter((groupName) => groups[groupName])
    .map((groupName) => ({
      groupName,
      models: groups[groupName].sort((a, b) => {
        if (a.created && b.created) return b.created - a.created;
        return b.id.localeCompare(a.id);
      }),
    }));

  // Add ungrouped models
  if (ungrouped.length > 0) {
    result.push({
      groupName: 'Other',
      models: ungrouped.sort((a, b) => {
        if (a.created && b.created) return b.created - a.created;
        return a.id.localeCompare(b.id);
      }),
    });
  }

  return result;
};
