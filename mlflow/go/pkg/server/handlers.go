package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/mlflow/mlflow/mlflow/go/pkg/protos"
)

type HandlerMap = map[string]func(http.ResponseWriter, *http.Request)

var proxy *httputil.ReverseProxy

func proxyHandler(w http.ResponseWriter, r *http.Request) {
	proxy.ServeHTTP(w, r)
}

func InitProxy(randomPort int) {
	targetURL, _ := url.Parse(fmt.Sprintf("http://localhost:%d", randomPort))
	proxy = httputil.NewSingleHostReverseProxy(targetURL)

	proxy.Director = func(req *http.Request) {
		req.URL.Scheme = targetURL.Scheme
		req.URL.Host = targetURL.Host
		req.Host = targetURL.Host // Optionally set the Host header to the target host
		// Ensure headers are copied over
		req.Header = make(http.Header)
		for h, val := range req.Header {
			req.Header[h] = val
		}
	}

	// This is the catch all route in Go.
	http.HandleFunc("/", proxyHandler)
}

func handleWith[Input any, Output any](w http.ResponseWriter, r *http.Request, inner func(*Input) Output) {
	body, err := io.ReadAll(r.Body)
	defer r.Body.Close()
	if err != nil {
		http.Error(w, "Error reading request body", http.StatusInternalServerError)
		return
	}

	var input *Input
	err = json.Unmarshal(body, &input)
	if err != nil {
		http.Error(w, "Error parsing JSON body", http.StatusBadRequest)
		return
	}

	output := inner(input)
	jsonResponse, err := json.Marshal(output)
	if err != nil {
		http.Error(w, "Error serializing JSON response body", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(jsonResponse))
}

func logBatchHandler(w http.ResponseWriter, r *http.Request) {
	handleWith(w, r, func(input *protos.LogBatch) protos.LogBatch_Response {
		panic("Not implemented")
		return protos.LogBatch_Response{}
	})
}

var Handlers = HandlerMap{
	"GET_mlflow.MlflowService.getExperiment":                               proxyHandler,
	"POST_mlflow.MlflowService.setExperimentTag":                           proxyHandler,
	"POST_mlflow.MlflowService.logModel":                                   proxyHandler,
	"GET_mlflow.ModelRegistryService.getModelVersion":                      proxyHandler,
	"POST_mlflow.ModelRegistryService.setModelVersionTag":                  proxyHandler,
	"GET_mlflow.MlflowService.getExperimentByName":                         proxyHandler,
	"POST_mlflow.MlflowService.deleteExperiment":                           proxyHandler,
	"POST_mlflow.MlflowService.setTag":                                     proxyHandler,
	"GET_mlflow.ModelRegistryService.getModelVersionByAlias":               proxyHandler,
	"GET_mlflow.MlflowService.searchExperiments":                           proxyHandler,
	"POST_mlflow.MlflowService.deleteRun":                                  proxyHandler,
	"GET_mlflow.MlflowService.getRun":                                      proxyHandler,
	"PATCH_mlflow.ModelRegistryService.updateModelVersion":                 proxyHandler,
	"PUT_mlflow.artifacts.MlflowArtifactsService.uploadArtifact":           proxyHandler,
	"POST_mlflow.MlflowService.createExperiment":                           proxyHandler,
	"POST_mlflow.MlflowService.restoreExperiment":                          proxyHandler,
	"GET_mlflow.ModelRegistryService.getLatestVersions":                    proxyHandler,
	"POST_mlflow.ModelRegistryService.setRegisteredModelTag":               proxyHandler,
	"GET_mlflow.artifacts.MlflowArtifactsService.downloadArtifact":         proxyHandler,
	"POST_mlflow.artifacts.MlflowArtifactsService.completeMultipartUpload": proxyHandler,
	"GET_mlflow.MlflowService.getMetricHistory":                            proxyHandler,
	"GET_mlflow.MlflowService.getMetricHistoryBulkInterval":                proxyHandler,
	"GET_mlflow.ModelRegistryService.searchModelVersions":                  proxyHandler,
	"POST_mlflow.artifacts.MlflowArtifactsService.abortMultipartUpload":    proxyHandler,
	"POST_mlflow.MlflowService.createRun":                                  proxyHandler,
	"GET_mlflow.MlflowService.listArtifacts":                               proxyHandler,
	"POST_mlflow.ModelRegistryService.createRegisteredModel":               proxyHandler,
	"DELETE_mlflow.ModelRegistryService.deleteModelVersionTag":             proxyHandler,
	"POST_mlflow.MlflowService.updateExperiment":                           proxyHandler,
	"POST_mlflow.MlflowService.logMetric":                                  proxyHandler,
	"POST_mlflow.MlflowService.searchExperiments":                          proxyHandler,
	"POST_mlflow.ModelRegistryService.createModelVersion":                  proxyHandler,
	"POST_mlflow.ModelRegistryService.transitionModelVersionStage":         proxyHandler,
	"POST_mlflow.ModelRegistryService.renameRegisteredModel":               proxyHandler,
	"PATCH_mlflow.ModelRegistryService.updateRegisteredModel":              proxyHandler,
	"DELETE_mlflow.ModelRegistryService.deleteRegisteredModel":             proxyHandler,
	"POST_mlflow.ModelRegistryService.setRegisteredModelAlias":             proxyHandler,
	"DELETE_mlflow.artifacts.MlflowArtifactsService.deleteArtifact":        proxyHandler,
	"POST_mlflow.artifacts.MlflowArtifactsService.createMultipartUpload":   proxyHandler,
	"POST_mlflow.MlflowService.updateRun":                                  proxyHandler,
	"DELETE_mlflow.ModelRegistryService.deleteModelVersion":                proxyHandler,
	"GET_mlflow.ModelRegistryService.getModelVersionDownloadUri":           proxyHandler,
	"POST_mlflow.MlflowService.logParam":                                   proxyHandler,
	"POST_mlflow.MlflowService.searchRuns":                                 proxyHandler,
	"GET_mlflow.artifacts.MlflowArtifactsService.listArtifacts":            proxyHandler,
	"POST_mlflow.MlflowService.restoreRun":                                 proxyHandler,
	"POST_mlflow.MlflowService.deleteTag":                                  proxyHandler,
	"POST_mlflow.ModelRegistryService.getLatestVersions":                   proxyHandler,
	"POST_mlflow.MlflowService.logBatch":                                   logBatchHandler,
	"POST_mlflow.MlflowService.logInputs":                                  proxyHandler,
	"GET_mlflow.ModelRegistryService.getRegisteredModel":                   proxyHandler,
	"GET_mlflow.ModelRegistryService.searchRegisteredModels":               proxyHandler,
	"DELETE_mlflow.ModelRegistryService.deleteRegisteredModelTag":          proxyHandler,
	"DELETE_mlflow.ModelRegistryService.deleteRegisteredModelAlias":        proxyHandler,
}
