/**
 * MLflow Tracing Plugin for Opencode
 *
 * This plugin listens for session.idle events and invokes the MLflow tracing
 * Python script to capture conversation traces.
 *
 * Usage:
 *   1. Run: mlflow autolog opencode -u http://localhost:5000
 *   2. Install: bun add mlflow-opencode
 *   3. Run opencode normally - tracing happens automatically
 */

import type { Plugin, PluginInput, Hooks } from "@opencode-ai/plugin";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

const processedTurns = new Map<string, number>();

// Silent plugin - no console output to avoid TUI interference
// Errors are logged to stderr only in debug mode
const DEBUG = process.env.MLFLOW_OPENCODE_DEBUG === "true";

interface SessionIdleEvent {
  type: string;
  properties: {
    sessionID?: string;
  };
}

interface MLflowConfig {
  enabled?: boolean;
  trackingUri?: string;
  experimentId?: string;
  experimentName?: string;
}

/**
 * Load MLflow config from .opencode/mlflow.json
 */
function loadMLflowConfig(directory: string): MLflowConfig {
  const configPath = join(directory, ".opencode", "mlflow.json");
  try {
    if (existsSync(configPath)) {
      const content = readFileSync(configPath, "utf-8");
      return JSON.parse(content) as MLflowConfig;
    }
  } catch (e) {
    if (DEBUG) console.error("[mlflow] Failed to load config:", e);
  }
  return {};
}

/**
 * MLflow tracing plugin for Opencode.
 * Automatically traces conversations to MLflow when sessions become idle.
 */
export const MLflowTracingPlugin: Plugin = async (
  input: PluginInput
): Promise<Hooks> => {
  const { client, directory } = input;

  return {
    event: async ({ event }) => {
      const typedEvent = event as SessionIdleEvent;

      // Only process session.idle events
      if (typedEvent.type !== "session.idle") {
        return;
      }

      const sessionID = typedEvent.properties?.sessionID;
      if (!sessionID) {
        return;
      }

      try {
        // Fetch session info and messages using the SDK client
        const sessionResult = await client.session.get({
          path: { id: sessionID },
        });
        if (!sessionResult.data) {
          if (DEBUG) console.error("[mlflow] Failed to fetch session:", sessionID);
          return;
        }

        const messagesResult = await client.session.messages({
          path: { id: sessionID },
          query: { limit: 1000 },
        });
        if (!messagesResult.data) {
          if (DEBUG) console.error("[mlflow] Failed to fetch messages:", sessionID);
          return;
        }

        // Check if we've already processed this exact turn (same message count)
        const allMessages = messagesResult.data;
        const messageCount = allMessages.length;
        const lastProcessedCount = processedTurns.get(sessionID) || 0;

        if (messageCount <= lastProcessedCount) {
          // Already processed this turn, skip
          return;
        }

        // Get only the NEW messages since last trace (current turn)
        const newMessages = allMessages.slice(lastProcessedCount);
        processedTurns.set(sessionID, messageCount);

        // Clean up old entries to prevent memory leak (keep last 50 sessions)
        if (processedTurns.size > 50) {
          const keys = Array.from(processedTurns.keys());
          for (let i = 0; i < keys.length - 50; i++) {
            processedTurns.delete(keys[i]);
          }
        }

        // Load MLflow config from .opencode/mlflow.json
        const mlflowConfig = loadMLflowConfig(directory);
        if (!mlflowConfig.enabled) {
          if (DEBUG) console.error("[mlflow] Tracing not enabled in config");
          return;
        }

        // Prepare session data for MLflow tracing - only the current turn
        const sessionData = {
          sessionID,
          session: sessionResult.data,
          messages: newMessages, // Only new messages for this turn
          turnNumber: Math.floor(messageCount / 2), // Approximate turn number
          directory,
        };

        // Build environment variables from config
        const envVars: Record<string, string> = {
          ...process.env as Record<string, string>,
          MLFLOW_OPENCODE_TRACING_ENABLED: "true",
        };

        if (mlflowConfig.trackingUri) {
          envVars.MLFLOW_TRACKING_URI = mlflowConfig.trackingUri;
        }
        if (mlflowConfig.experimentId) {
          envVars.MLFLOW_EXPERIMENT_ID = mlflowConfig.experimentId;
        } else if (mlflowConfig.experimentName) {
          envVars.MLFLOW_EXPERIMENT_NAME = mlflowConfig.experimentName;
        }

        // Invoke Python MLflow tracing script via subprocess
        const proc = Bun.spawn(
          ["python", "-m", "mlflow.opencode.hooks", "session_completed"],
          {
            stdin: "pipe",
            stdout: "pipe",
            stderr: "pipe",
            env: envVars,
          }
        );

        // Send session data as JSON to stdin using Bun's FileSink API
        const jsonData = JSON.stringify(sessionData);
        proc.stdin.write(jsonData);
        proc.stdin.end();

        // Wait for process to complete
        const exitCode = await proc.exited;

        if (exitCode !== 0 && DEBUG) {
          const stderr = await new Response(proc.stderr).text();
          console.error("[mlflow] Tracing failed:", stderr);
        }
        // Success - silent
      } catch (error) {
        if (DEBUG) console.error("[mlflow] Error processing session:", error);
      }
    },
  };
};

export default MLflowTracingPlugin;
