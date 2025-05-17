import React, { useState, useEffect } from 'react';
import { Redirect } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import styles from './index.module.css';

interface PathSelectorProps {
  title: string;
  path: string;
  description: string;
  color: 'blue' | 'red';
  onClick: () => void;
}

function PathSelector({ 
  title, 
  path, 
  description, 
  color, 
  onClick 
}: PathSelectorProps): JSX.Element {
  return (
    <div 
      className={clsx(styles.glossyCard, styles[`glossyCard${color}`])}
      onClick={onClick}
    >
      <div className={styles.cardContent}>
        <div className={styles.cardHeader}>
          <div className={clsx(styles.colorBlock, styles[`colorBlock${color}`])}></div>
          <h2 className={styles.cardTitle}>{title}</h2>
        </div>
        <p className={styles.cardDescription}>{description}</p>
        <div className={styles.cardAction}>
          <span className={styles.cardButton}>
            View documentation <span className={styles.arrowIcon}>→</span>
          </span>
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
      description="MLflow Documentation - Machine Learning and GenAI lifecycle management">
      <main className={styles.homeContainer}>
        <div className={styles.contentGrid}>
          {/* Left Column - Text */}
          <div className={styles.textColumn}>
            <h1 className={styles.megaHeading}>Documentation</h1>
            <p className={styles.introText}>
              Welcome to the MLflow Documentation. Our documentation is organized into two sections 
              to help you find exactly what you need. Choose Model Training for traditional ML workflows, 
              or select GenAI Apps & Agents for generative AI applications, tracing, and evaluation tools.
            </p>
          </div>

          <div className={styles.cardsColumn}>
            <PathSelector
              title="Model Training"
              path="/ml"
              description="Access comprehensive guides for experiment tracking, model packaging, registry management, 
              and deployment. Get started with MLflow's core functionality for traditional machine 
              learning workflows, hyperparameter tuning, and model lifecycle management."
              color="blue"
              onClick={() => setRedirect('/ml')}
            />

            <PathSelector
              title="GenAI Apps & Agents"
              path="/genai"
              description="Explore tools for LLM tracing, prompt management, foundation model deployment, 
              and evaluation frameworks. Learn how to track, evaluate, and optimize your generative 
              AI applications and agent workflows with MLflow."
              color="red"
              onClick={() => setRedirect('/genai')}
            />
          </div>
        </div>
      </main>
    </Layout>
  );
}