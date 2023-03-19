{{/*
Expand the name of the chart.
*/}}
{{- define "mlflow-quickstart.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "mlflow-quickstart.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mlflow-quickstart.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mlflow-quickstart.labels" -}}
helm.sh/chart: {{ include "mlflow-quickstart.chart" . }}
{{ include "mlflow-quickstart.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mlflow-quickstart.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mlflow-quickstart.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mlflow-quickstart.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mlflow-quickstart.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}


{{/* Create the Sqlalchemy connection string for postgres */}}
{{- define "mlflow-quickstart.backendStoreUri" -}}
postgresql+psycopg2://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "mlflow-quickstart.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- end }}

{{/* Create the Sqlalchemy connection string for postgres */}}
{{- define "mlflow-quickstart.artifactEndpointUrl" -}}
http://{{ include "mlflow-quickstart.fullname" . }}-minio:9000
{{- end }}