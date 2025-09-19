import React from 'react';
import { FormattedMessage } from 'react-intl';
import type { HomeNewsItemDefinition } from './types';

export const homeNewsItems: HomeNewsItemDefinition[] = [
  {
    id: 'auto-tune-llm-judge',
    title: (
      <FormattedMessage defaultMessage="Auto-tune LLM judge" description="Home page news card title one" />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Compare evaluations with updated heuristics and guardrails."
        description="Home page news card description one"
      />
    ),
    ctaLabel: <FormattedMessage defaultMessage="Read update" description="Home page news card CTA one" />,
    link: {
      type: 'external',
      href: 'https://mlflow.org/docs/latest/llms/llm-evaluate/index.html',
      target: '_blank',
      rel: 'noopener noreferrer',
    },
    componentId: 'mlflow.home.news.auto_tune_llm_judge',
    thumbnail: {
      label: (
        <FormattedMessage
          defaultMessage="Evaluations dashboard"
          description="Accessibility label for the evaluations news thumbnail"
        />
      ),
      gradient: 'linear-gradient(135deg, #E9F2FF 0%, #F4EBFF 100%)',
    },
  },
  {
    id: 'optimize-prompts',
    title: <FormattedMessage defaultMessage="Optimize prompts" description="Home page news card title two" />,
    description: (
      <FormattedMessage
        defaultMessage="Discover techniques to improve prompt performance in production."
        description="Home page news card description two"
      />
    ),
    ctaLabel: <FormattedMessage defaultMessage="See guide" description="Home page news card CTA two" />,
    link: {
      type: 'external',
      href: 'https://mlflow.org/docs/latest/llms/prompt-engineering/index.html',
      target: '_blank',
      rel: 'noopener noreferrer',
    },
    componentId: 'mlflow.home.news.optimize_prompts',
    thumbnail: {
      label: (
        <FormattedMessage
          defaultMessage="Prompt optimization walkthrough"
          description="Accessibility label for prompt news thumbnail"
        />
      ),
      gradient: 'linear-gradient(135deg, #E8F7F2 0%, #D5E8FF 100%)',
    },
  },
  {
    id: 'agents-as-a-judge',
    title: <FormattedMessage defaultMessage="Agents-as-a-judge" description="Home page news card title three" />,
    description: (
      <FormattedMessage
        defaultMessage="Leverage automated reviewers to scale evaluation coverage."
        description="Home page news card description three"
      />
    ),
    ctaLabel: (
      <FormattedMessage defaultMessage="View announcement" description="Home page news card CTA three" />
    ),
    link: {
      type: 'external',
      href: 'https://mlflow.org/docs/latest/llms/index.html',
      target: '_blank',
      rel: 'noopener noreferrer',
    },
    componentId: 'mlflow.home.news.agents_as_a_judge',
    thumbnail: {
      label: (
        <FormattedMessage
          defaultMessage="Agent workflow preview"
          description="Accessibility label for agents news thumbnail"
        />
      ),
      gradient: 'linear-gradient(135deg, #FFF5E1 0%, #FFE2F2 100%)',
    },
  },
  {
    id: 'dataset-tracking',
    title: <FormattedMessage defaultMessage="Dataset tracking" description="Home page news card title four" />,
    description: (
      <FormattedMessage
        defaultMessage="Track dataset lineage and versions alongside experiments."
        description="Home page news card description four"
      />
    ),
    ctaLabel: <FormattedMessage defaultMessage="Learn more" description="Home page news card CTA four" />,
    link: {
      type: 'external',
      href: 'https://mlflow.org/docs/latest/data/index.html',
      target: '_blank',
      rel: 'noopener noreferrer',
    },
    componentId: 'mlflow.home.news.dataset_tracking',
    thumbnail: {
      label: (
        <FormattedMessage
          defaultMessage="Dataset comparison snapshot"
          description="Accessibility label for dataset news thumbnail"
        />
      ),
      gradient: 'linear-gradient(135deg, #E6F3FF 0%, #E0F1F6 100%)',
    },
  },
];
