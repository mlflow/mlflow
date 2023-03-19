{{/* mlflow enable tracking features */}}
{{- define "mlflow.enableTracking" -}}
{{- if not .Values.trackingServer.enabled -}}
--artifacts-only
{{- end -}}
{{- end -}}

{{/* mlflow server port */}}
{{- define "mlflow.serverPort" -}}
{{- if .Values.trackingServer.port -}}
--port={{ .Values.trackingServer.port }}
{{- end -}}
{{- end -}}

{{/* mlflow server worker */}}
{{- define "mlflow.serverWorkers" -}}
{{- if .Values.trackingServer.workers -}}
--workers={{ .Values.trackingServer.workers }}
{{- end -}}
{{- end -}}

{{/* mlflow server metrics */}}
{{- define "mlflow.serverMetrics" -}}
{{- if .Values.trackingServer.metrics -}}
--expose-prometheus=/metrics
{{- end -}}
{{- end -}}

{{/* mlflow server proxy artifacts */}}
{{- define "mlflow.proxyArtifacts" -}}
{{- if .Values.artifacts.serve -}}
--serve-artifacts
{{- else -}}
--no-serve-artifacts
{{- end -}}
{{- end -}}


{{/* mlflow deployment container command for launching the tracking server */}}
{{- define "mlflow.trackingServerArgs" -}}
{{- $args := list (include "mlflow.enableTracking" .)
                  (include "mlflow.serverPort" .)
                  (include "mlflow.serverWorkers" .)
                  (include "mlflow.serverMetrics" .)
                  (include "mlflow.proxyArtifacts" .) -}}
- mlflow
- server
- --host=0.0.0.0
- --backend-store-uri=$(BACKEND_STORE_URI)
- --default-artifact-root=$(DEFAULT_ARTIFACT_ROOT)
{{- range $args }}
{{- if . }}
- {{ . }}
{{- end }}
{{- end }}
{{- end -}}
