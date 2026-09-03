{{- define "mlflow.fullname" -}}
{{- .Values.fullnameOverride | default (printf "%s-%s" .Release.Name .Chart.Name) | trunc 63 | trimSuffix "-" }}
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
{{- $tag := .Values.image.tag | default (printf "v%s-full" .Chart.AppVersion) }}
{{- printf "%s:%s" .Values.image.repository $tag }}
{{- end }}

{{- define "mlflow.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- .Values.serviceAccount.name | default (include "mlflow.fullname" .) }}
{{- else }}
{{- .Values.serviceAccount.name | default "default" }}
{{- end }}
{{- end }}

{{/*
Build mlflow server args from .Values.server.
  value_options: map of key/value pairs rendered as --key=value
    (empty value is not allowed, underscores in `key` converted to hyphens).
  flag_options:  list of flag names rendered as --flag (underscores converted to hyphens).
*/}}
{{- define "mlflow.serverArgs" -}}
- mlflow
- server
{{- range $key, $val := .Values.server.value_options }}
{{- $flag := $key | replace "_" "-" }}
- --{{ $flag }}={{ $val }}
{{- end }}
{{- range .Values.server.flag_options }}
- --{{ . | replace "_" "-" }}
{{- end }}
{{- end }}

{{/*
Render a liveness/readiness/startup probe body from a probes.* values entry.
The chart always owns the httpGet request (path, port, scheme); only the
timing/threshold fields are read from the probe values.
Expects a dict with:
  probe: the .Values.probes.<name> map (must have .enabled == true)
  path:  the health check path
  port:  the named container port
  tls:   whether to set scheme: HTTPS
*/}}
{{- define "mlflow.probe" -}}
httpGet:
  path: {{ .path }}
  port: {{ .port }}
  {{- if .tls }}
  scheme: HTTPS
  {{- end }}
{{- with .probe.initialDelaySeconds }}
initialDelaySeconds: {{ . }}
{{- end }}
{{- with .probe.periodSeconds }}
periodSeconds: {{ . }}
{{- end }}
{{- with .probe.timeoutSeconds }}
timeoutSeconds: {{ . }}
{{- end }}
{{- with .probe.failureThreshold }}
failureThreshold: {{ . }}
{{- end }}
{{- end }}
