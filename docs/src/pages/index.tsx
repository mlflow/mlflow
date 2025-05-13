import React, { useState, useEffect } from 'react';
import { Redirect } from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import styles from './index.module.css';
import Link from '@docusaurus/Link';
import clsx from 'clsx';

const classicMLImage = 'images/guides/introductory/hyperparameter-tuning-with-child-runs/no-child-first.gif';
const genAIImage = 'images/llms/tracing/tracing-top.gif';

interface PathSelectorProps {
  title: string;
  image: string;
  path: string;
  description: string;
  features: string[];
  color: 'blue' | 'orange';
  onClick: () => void;
}

function PathSelector({ 
  title, 
  image, 
  path, 
  description, 
  features, 
  color, 
  onClick 
}: PathSelectorProps): JSX.Element {
  return (
    <div 
      className={clsx(styles.pathCard, styles[`pathCard${color}`])}
      onClick={onClick}
    >
      <div className={styles.pathCardInner}>
        <h2 className={styles.cardTitle}>{title}</h2>
        <p className={styles.cardDescription}>{description}</p>
        
        <div className={styles.cardImageContainer}>
          <img src={image} alt={title} className={styles.cardImage} />
          <div className={clsx(styles.imageOverlay, styles[`imageOverlay${color}`])}></div>
        </div>
        
        <div className={styles.cardContent}>
          <div className={styles.featureList}>
            {features.map((feature, index) => (
              <div key={index} className={styles.featureItem}>
                <span className={clsx(styles.featureDot, styles[`featureDot${color}`])}></span>
                {feature}
              </div>
            ))}
          </div>
          
          <div className={styles.cardAction}>
            <Link
              to={path}
              className={clsx(styles.cardButton, styles[`cardButton${color}`])}
              onClick={(e) => e.preventDefault()}
            >
              Get Started
            </Link>
          </div>
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
        <div className={styles.headerSection}>
          <h1 className={styles.mainHeading}>MLflow Documentation</h1>
          <p className={styles.subtitle}>Choose your MLflow experience</p>
        </div>

        <div className={styles.selectionGrid}>
          <PathSelector
            title="MLflow for Classic ML"
            image={classicMLImage}
            path="/ml"
            description="Track experiments, package code, manage models"
            features={[
              "Experiment tracking & visualization",
              "Model packaging & reproducibility",
              "Model registry & versioning",
              "Model serving & deployment"
            ]}
            color="blue"
            onClick={() => setRedirect('/ml')}
          />

          <PathSelector
            title="MLflow for GenAI"
            image={genAIImage}
            path="/genai"
            description="Trace, evaluate, and deploy generative AI models"
            features={[
              "LLM tracing & conversation analytics",
              "Prompt management & versioning",
              "Foundation model deployment",
              "Evaluation frameworks & assessments"
            ]}
            color="orange"
            onClick={() => setRedirect('/genai')}
          />
        </div>
      </main>
    </Layout>
  );
}