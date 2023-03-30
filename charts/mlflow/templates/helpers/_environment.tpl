{{/* Get the default artifact root */}}
{{- define "mlflow.defaultArtifactRoot" -}}
{{- if .Values.artifacts.s3.defaultArtifactRoot -}}
- name: DEFAULT_ARTIFACT_ROOT
  value: {{ .Values.artifacts.s3.defaultArtifactRoot }}
{{- else if .Values.artifacts.gcp.defaultArtifactRoot -}}
- name: DEFAULT_ARTIFACT_ROOT
  value: {{ .Values.artifacts.gcp.defaultArtifactRoot }}
{{- else if .Values.artifacts.azure.defaultArtifactRoot -}}
- name: DEFAULT_ARTIFACT_ROOT
  value: {{ .Values.artifacts.azure.defaultArtifactRoot }}
{{- else -}}
{{  fail "Could not resolve `default artifact root` from supplied values" }}
{{- end -}}
{{- end -}}


{{/* Get the backend store uri */}}
{{- define "mlflow.backendStoreUri" -}}
{{- if .Values.backendStore.existingSecret -}}
- name: BACKEND_STORE_URI
  valueFrom:
    secretKeyRef:
      name: {{ .Values.backendStore.existingSecret }}
      key: BACKEND_STORE_URI
      optional: false
{{- else if .Values.backendStore.createSecret.uri -}}
- name: BACKEND_STORE_URI
  valueFrom:
    secretKeyRef:
      name: mlflow-backend-store-credentials
      key: BACKEND_STORE_URI
      optional: false
{{- else -}}
{{  fail "Could not resolve `backend store uri` from supplied values" }}
{{- end -}}
{{- end -}}
