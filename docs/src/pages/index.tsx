import React, { useState, useEffect } from 'react';
import { Redirect } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './index.module.css';
import useBaseUrl from '@docusaurus/useBaseUrl';

interface PathSelectorProps {
  title: string;
  description: string;
  color: 'blue' | 'red';
  buttons: {
    text: string;
    link: string;
  }[];
}

function PathSelector({ title, description, color, buttons }: PathSelectorProps): JSX.Element {
  return (
    <div className={clsx(styles.glossyCard, styles[`glossyCard${color}`])}>
      <div className={styles.cardContent}>
        <div className={styles.cardHeader}>
          <div className={clsx(styles.colorBlock, styles[`colorBlock${color}`])}></div>
          <h2 className={styles.cardTitle}>{title}</h2>
        </div>
        <p className={styles.cardDescription}>{description}</p>
        <div className={styles.cardActions}>
          {buttons.map((button, index) => (
            <a
              key={index}
              href={button.link}
              className={styles.cardButton}
              onClick={(e) => {
                e.stopPropagation();
              }}
            >
              {button.text} <span className={styles.arrowIcon}>→</span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Home(): JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  const [redirect, setRedirect] = useState<string | null>(null);

  useEffect(() => {
    document.body.classList.add('homepage');
    return () => {
      document.body.classList.remove('homepage');
    };
  }, []);

  if (redirect) {
    return <Redirect to={redirect} />;
  }

  return (
    <Layout
      title={siteConfig.title}
      description="MLflow Documentation - Machine Learning and GenAI lifecycle management"
    >
      <main className={styles.homeContainer}>
        <div className={styles.contentGrid}>
          <div className={styles.textColumn}>
            <h1 className={styles.megaHeading}>Documentation</h1>
            <p className={styles.introText}>
              Welcome to the MLflow Documentation. Our documentation is organized into two sections to help you find
              exactly what you need. Choose Model Training for traditional ML workflows, or select GenAI Apps & Agents
              for generative AI applications, tracing, and evaluation tools.
            </p>
          </div>

          <div className={styles.cardsColumn}>
            <PathSelector
              title="Model Training"
              description="Access comprehensive guides for experiment tracking, model packaging, registry management,
              and deployment. Get started with MLflow's core functionality for traditional machine
              learning workflows, hyperparameter tuning, and model lifecycle management."
              color="blue"
              buttons={[
                {
                  text: 'Open Source',
                  link: useBaseUrl('/ml/'),
                },
                {
                  text: 'MLflow on Databricks',
                  link: 'https://docs.databricks.com/aws/en/mlflow/',
                },
              ]}
            />

            <PathSelector
              title="GenAI Apps & Agents"
              description="Explore tools for GenAI tracing, prompt management, foundation model deployment,
              and evaluation frameworks. Learn how to track, evaluate, and optimize your generative
              AI applications and agent workflows with MLflow."
              color="red"
              buttons={[
                {
                  text: 'Open Source',
                  link: useBaseUrl('/genai/'),
                },
                {
                  text: 'MLflow on Databricks',
                  link: 'https://docs.databricks.com/aws/en/mlflow3/genai/',
                },
              ]}
            />
          </div>
        </div>
      </main>
    </Layout>
  );
}
