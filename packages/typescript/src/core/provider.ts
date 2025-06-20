import { trace, Tracer } from "@opentelemetry/api";
import { MlflowSpanExporter, MlflowSpanProcessor } from "../exporters/mlflow";
import { NodeSDK } from "@opentelemetry/sdk-node";


// TODO: Implement branching logic to actually set span processor and exporter
const exporter = new MlflowSpanExporter();
const processor = new MlflowSpanProcessor(exporter);
const sdk = new NodeSDK({spanProcessors: [processor]});
sdk.start();


export function getTracer(module_name: string): Tracer {
    return trace.getTracer(module_name);
}