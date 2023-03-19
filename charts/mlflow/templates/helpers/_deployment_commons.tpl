{{/* mlflow deployment spec template metadata for use in all deployment manifests */}}
{{- define "mlflow.deploymentTemplateMetadataValues" -}}
metadata:
  {{- with .Values.podAnnotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  labels:
    {{- include "mlflow.selectorLabels" . | nindent 4 }}
{{- end -}}


{{/* mlflow deployment commen template spec configuration values */}}
{{- define "mlflow.deploymentCommonTemplateSpecValues" -}}
{{- with .Values.image.pullSecrets }}
imagePullSecrets:
  {{- toYaml . | nindent 2 }}
{{- end }}
serviceAccountName:
  {{- include "mlflow.serviceAccountName" . | nindent 2}}
securityContext:
  {{- toYaml .Values.podSecurityContext | nindent 2 }}
{{- with .Values.nodeSelector }}
nodeSelector:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.affinity }}
affinity:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- with .Values.tolerations }}
tolerations:
  {{- toYaml . | nindent 2 }}
{{- end }}
{{- end -}}


{{/* mlflow tracking server container common configuration values */}}
{{- define "mlflow.trackingServerContainerCommonValues" -}}
securityContext:
  {{- toYaml .Values.securityContext | nindent 2 }}
image: "{{ .Values.image.repository }}:{{ .Chart.AppVersion }}"
imagePullPolicy: {{ .Values.image.pullPolicy }}
command:
  {{- include "mlflow.trackingServerArgs" . | nindent 2 }}
ports:
  - name: http
    containerPort: {{ .Values.trackingServer.port }}
    protocol: TCP
livenessProbe:
  httpGet:
    path: /
    port: http
readinessProbe:
  httpGet:
    path: /
    port: http
resources:
  {{- toYaml .Values.resources | nindent 2 }}
{{- end -}}
