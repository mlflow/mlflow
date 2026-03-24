{{- define "mlflow.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "mlflow.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{ include "mlflow.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "mlflow.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "mlflow.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}

{{/*
Build mlflow server args from .Values.server.
Option names mirror MLflow CLI flags (underscores -> hyphens).
*/}}
{{- define "mlflow.serverArgs" -}}
- mlflow
- server
{{- range $key, $val := .Values.server }}
{{- $flag := $key | replace "_" "-" }}
{{- if kindIs "bool" $val }}
{{- if $val }}
- --{{ $flag }}
{{- else }}
- --no-{{ $flag }}
{{- end }}
{{- else if $val }}
- --{{ $flag }}={{ $val }}
{{- end }}
{{- end }}
{{- end }}
