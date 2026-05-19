# Security Policy

MLflow and its community take security bugs seriously. We appreciate efforts to improve the security of MLflow
and follow the [GitHub coordinated disclosure of security vulnerabilities](https://docs.github.com/en/code-security/security-advisories/about-coordinated-disclosure-of-security-vulnerabilities#about-reporting-and-disclosing-vulnerabilities-in-projects-on-github)
for responsible disclosure and prompt mitigation. We are committed to working with security researchers to
resolve the vulnerabilities they discover.

## Supported Versions

The latest version of MLflow has continued support. If a critical vulnerability is found in the current version
of MLflow, we may opt to backport patches to previous versions.

## Bounty Platform Policy

Due to an excessive volume of low-quality, AI-generated reports (including duplicates, low-effort submissions, non-reproducible claims, and spam), **we no longer accept vulnerability reports through bounty platforms such as [Huntr](https://huntr.com/)**. All reports submitted through such platforms will be automatically closed without review.

Please use the reporting process described below instead.

## Reporting a Vulnerability

When finding a security vulnerability in MLflow, please report it through [GitHub's private vulnerability reporting](https://github.com/mlflow/mlflow/security/advisories/new). Include detailed information about the security vulnerability, evidence that supports the relevance of the finding, and any reproducibility instructions for independent confirmation. Do not disclose vulnerability details publicly until coordinated disclosure has been completed.

> [!WARNING]
> Do not open regular GitHub issues for security vulnerabilities. Any security vulnerability reported as a normal issue will be automatically closed.

MLflow maintainers will review and validate your report within the associated [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories/creating-a-security-advisory), using it to coordinate private discussion of the vulnerability details, develop a fix, and publicly disclose the issue after the fix has been released.
