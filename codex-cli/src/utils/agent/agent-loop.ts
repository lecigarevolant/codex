import type { ReviewDecision } from "./review.js";
import type { ApplyPatchCommand, ApprovalPolicy } from "../../approvals.js";
import type { AppConfig } from "../config.js";
import type {
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseItem,
} from "openai/resources/responses/responses.mjs";
import type { Reasoning } from "openai/resources.mjs";

import { log, isLoggingEnabled } from "./log.js";
import { OPENAI_BASE_URL, OPENAI_TIMEOUT_MS } from "../config.js";
import { parseToolCallArguments } from "../parsers.js";
import {
  ORIGIN,
  CLI_VERSION,
  getSessionId,
  setCurrentModel,
  setSessionId,
} from "../session.js";
import { handleExecCommand } from "./handle-exec-command.js";
import { randomUUID } from "node:crypto";
import OpenAI, { APIConnectionTimeoutError } from "openai";

// Wait time before retrying after rate limit errors (ms).
const RATE_LIMIT_RETRY_WAIT_MS = parseInt(
  process.env["OPENAI_RATE_LIMIT_RETRY_WAIT_MS"] || "2500",
  10,
);

// Define more specific types for our tool calls
type ResponseWebSearchCall = ResponseItem & {
  type: "web_search_call";
  id: string;
};

/**
 * Interface for handling different types of tool calls from the model.
 * Each handler knows how to process a specific type of tool call and
 * produce response input items for the next turn.
 */
interface ToolHandler<T extends ResponseItem> {
  /** Return true if this item belongs to the handler. */
  canHandle(item: ResponseItem): item is T;

  /**
   * Do the side-effect (run command, query DB, etc.) and return the
   * ResponseInputItems that constitute the "tool output".
   */
  handle(item: T): Promise<Array<ResponseInputItem>>;
}

/**
 * Handles function calls like 'shell' commands
 */
class FunctionCallHandler implements ToolHandler<ResponseItem> {
  constructor(private agentLoop: AgentLoop) {}

  canHandle(item: ResponseItem): item is ResponseItem {
    return item.type === "function_call";
  }

  async handle(item: ResponseItem): Promise<Array<ResponseInputItem>> {
    // Cast to ResponseFunctionToolCall for handleFunctionCall
    return this.agentLoop.handleFunctionCall(item as ResponseFunctionToolCall);
  }
}

/**
 * Handles web search calls
 */
class WebSearchHandler implements ToolHandler<ResponseWebSearchCall> {
  canHandle(item: ResponseItem): item is ResponseWebSearchCall {
    return item.type === "web_search_call";
  }

  async handle(item: ResponseWebSearchCall): Promise<Array<ResponseInputItem>> {
    if (isLoggingEnabled()) {
      log(`WebSearchHandler: Handling web_search_call ID: ${item.id}`);
    }

    // Web search calls don't need an explicit response like function calls do
    // The OpenAI API handles these automatically
    return [];
  }
}

export type CommandConfirmation = {
  review: ReviewDecision;
  applyPatch?: ApplyPatchCommand | undefined;
  customDenyMessage?: string;
  explanation?: string;
};

const alreadyProcessedResponses = new Set();

type AgentLoopParams = {
  model: string;
  config?: AppConfig;
  instructions?: string;
  approvalPolicy: ApprovalPolicy;
  onItem: (item: ResponseItem) => void;
  onLoading: (loading: boolean) => void;

  /** Extra writable roots to use with sandbox execution. */
  additionalWritableRoots: ReadonlyArray<string>;

  /** Called when the command is not auto-approved to request explicit user review. */
  getCommandConfirmation: (
    command: Array<string>,
    applyPatch: ApplyPatchCommand | undefined,
  ) => Promise<CommandConfirmation>;
  onLastResponseId: (lastResponseId: string) => void;
};

export class AgentLoop {
  private model: string;
  private instructions?: string;
  private approvalPolicy: ApprovalPolicy;
  private config: AppConfig;
  private additionalWritableRoots: ReadonlyArray<string>;

  // Using `InstanceType<typeof OpenAI>` sidesteps typing issues with the OpenAI package under
  // the TS 5+ `moduleResolution=bundler` setup. OpenAI client instance. We keep the concrete
  // type to avoid sprinkling `any` across the implementation while still allowing paths where
  // the OpenAI SDK types may not perfectly match. The `typeof OpenAI` pattern captures the
  // instance shape without resorting to `any`.
  private oai: OpenAI;

  private onItem: (item: ResponseItem) => void;
  private onLoading: (loading: boolean) => void;
  private getCommandConfirmation: (
    command: Array<string>,
    applyPatch: ApplyPatchCommand | undefined,
  ) => Promise<CommandConfirmation>;
  private onLastResponseId: (lastResponseId: string) => void;

  /**
   * A reference to the currently active stream returned from the OpenAI
   * client. We keep this so that we can abort the request if the user decides
   * to interrupt the current task (e.g. via the escape hot‑key).
   */
  private currentStream: unknown | null = null;
  /** Incremented with every call to `run()`. Allows us to ignore stray events
   * from streams that belong to a previous run which might still be emitting
   * after the user has canceled and issued a new command. */
  private generation = 0;
  /** AbortController for in‑progress tool calls (e.g. shell commands). */
  private execAbortController: AbortController | null = null;
  /** Set to true when `cancel()` is called so `run()` can exit early. */
  private canceled = false;
  /** Function calls that were emitted by the model but never answered because
   *  the user cancelled the run.  We keep the `call_id`s around so the *next*
   *  request can send a dummy `function_call_output` that satisfies the
   *  contract and prevents the
   *    400 | No tool output found for function call …
   *  error from OpenAI. */
  private pendingAborts: Set<string> = new Set();
  /** Set to true by `terminate()` – prevents any further use of the instance. */
  private terminated = false;
  /** Master abort controller – fires when terminate() is invoked. */
  private readonly hardAbort = new AbortController();

  /** Collection of tool handlers for processing different tool calls. */
  private readonly toolHandlers: Array<ToolHandler<ResponseItem>>;

  public sessionId: string;
  /*
   * Cumulative thinking time across this AgentLoop instance (ms).
   * Currently not used anywhere – comment out to keep the strict compiler
   * happy under `noUnusedLocals`.  Restore when telemetry support lands.
   */
  // private cumulativeThinkingMs = 0;
  constructor({
    model,
    instructions,
    approvalPolicy,
    // `config` used to be required.  Some unit‑tests (and potentially other
    // callers) instantiate `AgentLoop` without passing it, so we make it
    // optional and fall back to sensible defaults.  This keeps the public
    // surface backwards‑compatible and prevents runtime errors like
    // "Cannot read properties of undefined (reading 'apiKey')" when accessing
    // `config.apiKey` below.
    config,
    onItem,
    onLoading,
    getCommandConfirmation,
    onLastResponseId,
    additionalWritableRoots,
  }: AgentLoopParams & { config?: AppConfig }) {
    this.model = model;
    this.instructions = instructions;
    this.approvalPolicy = approvalPolicy;

    // If no `config` has been provided we derive a minimal stub so that the
    // rest of the implementation can rely on `this.config` always being a
    // defined object.  We purposefully copy over the `model` and
    // `instructions` that have already been passed explicitly so that
    // downstream consumers (e.g. telemetry) still observe the correct values.
    this.config =
      config ??
      ({
        model,
        instructions: instructions ?? "",
      } as AppConfig);
    this.additionalWritableRoots = additionalWritableRoots;
    this.onItem = onItem;
    this.onLoading = onLoading;
    this.getCommandConfirmation = getCommandConfirmation;
    this.onLastResponseId = onLastResponseId;
    this.sessionId = getSessionId() || randomUUID().replaceAll("-", "");
    // Configure OpenAI client with optional timeout (ms) from environment
    const timeoutMs = OPENAI_TIMEOUT_MS;
    const apiKey = this.config.apiKey ?? process.env["OPENAI_API_KEY"] ?? "";
    this.oai = new OpenAI({
      // The OpenAI JS SDK only requires `apiKey` when making requests against
      // the official API.  When running unit‑tests we stub out all network
      // calls so an undefined key is perfectly fine.  We therefore only set
      // the property if we actually have a value to avoid triggering runtime
      // errors inside the SDK (it validates that `apiKey` is a non‑empty
      // string when the field is present).
      ...(apiKey ? { apiKey } : {}),
      baseURL: OPENAI_BASE_URL,
      defaultHeaders: {
        originator: ORIGIN,
        version: CLI_VERSION,
        session_id: this.sessionId,
      },
      ...(timeoutMs !== undefined ? { timeout: timeoutMs } : {}),
    });

    setSessionId(this.sessionId);
    setCurrentModel(this.model);

    // Initialize tool handlers
    this.toolHandlers = [
      new FunctionCallHandler(this),
      new WebSearchHandler(),
      // Add more handlers as needed
    ];

    this.hardAbort.signal.addEventListener(
      "abort",
      () => this.execAbortController?.abort(),
      { once: true },
    );
  }

  public async handleFunctionCall(
    item: ResponseFunctionToolCall,
  ): Promise<Array<ResponseInputItem>> {
    // If the agent has been canceled in the meantime we should not perform any
    // additional work. Returning an empty array ensures that we neither execute
    // the requested tool call nor enqueue any follow‑up input items. This keeps
    // the cancellation semantics intuitive for users – once they interrupt a
    // task no further actions related to that task should be taken.
    if (this.canceled) {
      return [];
    }
    // ---------------------------------------------------------------------
    // Normalise the function‑call item into a consistent shape regardless of
    // whether it originated from the `/responses` or the `/chat/completions`
    // endpoint – their JSON differs slightly.
    // ---------------------------------------------------------------------

    const isChatStyle =
      // The chat endpoint nests function details under a `function` key.
      // We conservatively treat the presence of this field as a signal that
      // we are dealing with the chat format.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (item as any).function != null;

    const name: string | undefined = isChatStyle
      ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (item as any).function?.name
      : // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (item as any).name;

    const rawArguments: string | undefined = isChatStyle
      ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (item as any).function?.arguments
      : // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (item as any).arguments;

    // The OpenAI "function_call" item may have either `call_id` (responses
    // endpoint) or `id` (chat endpoint).  Prefer `call_id` if present but fall
    // back to `id` to remain compatible.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const callId: string = (item as any).call_id ?? (item as any).id;

    const args = parseToolCallArguments(rawArguments ?? "{}");
    if (isLoggingEnabled()) {
      log(
        `handleFunctionCall(): name=${
          name ?? "undefined"
        } callId=${callId} args=${rawArguments}`,
      );
    }

    if (args == null) {
      const outputItem: ResponseInputItem.FunctionCallOutput = {
        type: "function_call_output",
        call_id: callId, // Use normalized callId for consistency
        output: `invalid arguments: ${rawArguments}`,
      };
      return [outputItem];
    }

    const outputItem: ResponseInputItem.FunctionCallOutput = {
      type: "function_call_output",
      // `call_id` is mandatory – ensure we never send `undefined` which would
      // trigger the "No tool output found…" 400 from the API. Use normalized callId.
      call_id: callId,
      output: "no function found",
    };

    // We intentionally *do not* remove this `callId` from the `pendingAborts`
    // set right away.  The output produced below is only queued up for the
    // *next* request to the OpenAI API – it has not been delivered yet.  If
    // the user presses ESC‑ESC (i.e. invokes `cancel()`) in the small window
    // between queuing the result and the actual network call, we need to be
    // able to surface a synthetic `function_call_output` marked as
    // "aborted".  Keeping the ID in the set until the run concludes
    // successfully lets the next `run()` differentiate between an aborted
    // tool call (needs the synthetic output) and a completed one (cleared
    // below in the `flush()` helper).

    // used to tell model to stop if needed
    const additionalItems: Array<ResponseInputItem> = [];

    // TODO: allow arbitrary function calls (beyond shell/container.exec)
    if (name === "container.exec" || name === "shell") {
      const {
        outputText,
        metadata,
        additionalItems: additionalItemsFromExec,
      } = await handleExecCommand(
        args,
        this.config,
        this.approvalPolicy,
        this.additionalWritableRoots,
        this.getCommandConfirmation,
        this.execAbortController?.signal,
      );
      outputItem.output = JSON.stringify({ output: outputText, metadata });

      if (additionalItemsFromExec) {
        additionalItems.push(...additionalItemsFromExec);
      }
    }

    return [outputItem, ...additionalItems];
  }

  // Add a debug logging helper that can handle large objects
  private debugLog = (message: string, obj?: unknown) => {
    if (!isLoggingEnabled()) {
      return;
    }

    try {
      if (obj) {
        const stringified =
          typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
        log(`${message}: ${stringified}`);
      } else {
        log(message);
      }
    } catch (err) {
      log(`${message}: [Error stringifying object: ${err}]`);
    }
  };

  /**
   * Process *all* tool calls contained in `responseItems` in the order they
   * appeared. Returns the items to feed into the next OpenAI call.
   */
  private async processToolCalls(
    responseItems: Array<ResponseItem>,
    stageItem: (i: ResponseItem) => void,
  ): Promise<Array<ResponseInputItem>> {
    const nextTurn: Array<ResponseInputItem> = [];

    this.debugLog(
      `Processing ${responseItems.length} response items for tool calls`,
    );

    for (const item of responseItems) {
      // Always surface the raw item to the UI.
      stageItem(item);

      // Find a handler that can process the item (first match wins).
      const handler = this.toolHandlers.find((h) => h.canHandle(item));
      if (!handler) {
        continue;
      }

      // Prevent duplicate execution if we have already handled this call_id.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const callId = (item as any).call_id ?? (item as any).id;
      if (!callId || alreadyProcessedResponses.has(callId)) {
        continue;
      }
      alreadyProcessedResponses.add(callId);

      // Execute tool side-effect -> gather ResponseInputItems.
      // eslint-disable-next-line no-await-in-loop
      const produced = await handler.handle(item as never);
      nextTurn.push(...produced);
    }

    this.debugLog(`Generated ${nextTurn.length} input items from tool calls`);
    return nextTurn;
  }

  public async run(
    input: Array<ResponseInputItem>,
    previousResponseId: string = "",
  ): Promise<void> {
    // ---------------------------------------------------------------------
    // Top‑level error wrapper so that known transient network issues like
    // `ERR_STREAM_PREMATURE_CLOSE` do not crash the entire CLI process.
    // Instead we surface the failure to the user as a regular system‑message
    // and terminate the current run gracefully. The calling UI can then let
    // the user retry the request if desired.
    // ---------------------------------------------------------------------

    try {
      if (this.terminated) {
        throw new Error("AgentLoop has been terminated");
      }
      // Record when we start "thinking" so we can report accurate elapsed time.
      const thisGeneration = ++this.generation;

      // Reset cancellation flag and stream for a fresh run.
      this.canceled = false;
      this.currentStream = null;

      // Create a fresh AbortController for this run so that tool calls from a
      // previous run do not accidentally get signalled.
      this.execAbortController = new AbortController();
      if (isLoggingEnabled()) {
        log(
          `AgentLoop.run(): new execAbortController created (${this.execAbortController.signal}) for generation ${this.generation}`,
        );
      }
      // NOTE: We no longer (re‑)attach an `abort` listener to `hardAbort` here.
      // A single listener that forwards the `abort` to the current
      // `execAbortController` is installed once in the constructor. Re‑adding a
      // new listener on every `run()` caused the same `AbortSignal` instance to
      // accumulate listeners which in turn triggered Node's
      // `MaxListenersExceededWarning` after ten invocations.

      let lastResponseId: string = previousResponseId;

      // If there are unresolved function calls from a previously cancelled run
      // we have to emit dummy tool outputs so that the API no longer expects
      // them.  We prepend them to the user‑supplied input so they appear
      // first in the conversation turn.
      const abortOutputs: Array<ResponseInputItem> = [];
      if (this.pendingAborts.size > 0) {
        for (const id of this.pendingAborts) {
          abortOutputs.push({
            type: "function_call_output",
            call_id: id,
            output: JSON.stringify({
              output: "aborted",
              metadata: { exit_code: 1, duration_seconds: 0 },
            }),
          } as ResponseInputItem.FunctionCallOutput);
        }
        // Once converted the pending list can be cleared.
        this.pendingAborts.clear();
      }

      let turnInput = [...abortOutputs, ...input];

      this.onLoading(true);

      const staged: Array<ResponseItem | undefined> = [];
      const stageItem = (item: ResponseItem) => {
        // Ignore any stray events that belong to older generations.
        if (thisGeneration !== this.generation) {
          return;
        }

        // Store the item so the final flush can still operate on a complete list.
        // We'll nil out entries once they're delivered.
        const idx = staged.push(item) - 1;

        // Instead of emitting synchronously we schedule a short‑delay delivery.
        // This accomplishes two things:
        //   1. The UI still sees new messages almost immediately, creating the
        //      perception of real‑time updates.
        //   2. If the user calls `cancel()` in the small window right after the
        //      item was staged we can still abort the delivery because the
        //      generation counter will have been bumped by `cancel()`.
        setTimeout(() => {
          if (
            thisGeneration === this.generation &&
            !this.canceled &&
            !this.hardAbort.signal.aborted
          ) {
            this.onItem(item);
            // Mark as delivered so flush won't re-emit it
            staged[idx] = undefined;
          }
        }, 10);
      };

      while (turnInput.length > 0) {
        if (this.canceled || this.hardAbort.signal.aborted) {
          this.onLoading(false);
          return;
        }
        // send request to openAI
        for (const item of turnInput) {
          stageItem(item as ResponseItem);
        }
        // Send request to OpenAI with retry on timeout
        let stream;

        // Retry loop for transient errors. Up to MAX_RETRIES attempts.
        const MAX_RETRIES = 5;
        for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
          try {
            let reasoning: Reasoning | undefined;
            if (this.model.startsWith("o")) {
              reasoning = { effort: "high" };
              if (this.model === "o3" || this.model === "o4-mini") {
                reasoning.summary = "auto";
              }
            }
            const prefix = `


You are operating as and within the Codex CLI, a terminal-based agentic coding assistant built by OpenAI. It wraps OpenAI models to enable natural language interaction with a local codebase. You are expected to be precise, safe, and helpful.

You can:
- Receive user prompts, project context, and files.
- Stream responses and emit function calls (e.g., shell commands, code edits).
- Apply patches, run commands, and manage user approvals based on policy.
- Work inside a sandboxed, git-backed workspace with rollback support.
- Log telemetry so sessions can be replayed or inspected later.
- More details on your functionality are available at \`codex --help\`
- Use the web.search tool to search the web for current information.


The Codex CLI is open-sourced. Don't confuse yourself with the old Codex language model built by OpenAI many moons ago (this is understandably top of mind for you!). Within this context, Codex refers to the open-source agentic coding interface.

You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

Please resolve the user's task by editing and testing the code files in your current code execution session. You are a deployed coding agent. Your session allows for you to modify and run code. The repo(s) are already cloned in your working directory, and you must fully solve the problem for your answer to be considered correct.

You MUST adhere to the following criteria when executing the task:
- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
- Use the web.search tool to search the web for current information when you need to access up-to-date documentation, recent developments, or any information that might have changed since your training data cutoff. This is especially important for questions about new technologies, current events, or evolving software libraries.
- User instructions may overwrite the *CODING GUIDELINES* section in this developer message.
- Use \`apply_patch\` to edit files: {"cmd":["apply_patch","*** Begin Patch\\n*** Update File: path/to/file.py\\n@@ def example():\\n-  pass\\n+  return 123\\n*** End Patch"]}
- If completing the user's task requires writing or modifying files:
    - Your code and final answer should follow these *CODING GUIDELINES*:
        - Fix the problem at the root cause rather than applying surface-level patches, when possible.
        - Avoid unneeded complexity in your solution.
            - Ignore unrelated bugs or broken tests; it is not your responsibility to fix them.
        - Update documentation as necessary.
        - Keep changes consistent with the style of the existing codebase. Changes should be minimal and focused on the task.
            - Use \`git log\` and \`git blame\` to search the history of the codebase if additional context is required; internet access is disabled.
        - NEVER add copyright or license headers unless specifically requested.
        - You do not need to \`git commit\` your changes; this will be done automatically for you.
        - If there is a .pre-commit-config.yaml, use \`pre-commit run --files ...\` to check that your changes pass the pre-commit checks. However, do not fix pre-existing errors on lines you didn't touch.
            - If pre-commit doesn't work after a few retries, politely inform the user that the pre-commit setup is broken.
        - Once you finish coding, you must
            - Check \`git status\` to sanity check your changes; revert any scratch files or changes.
            - Remove all inline comments you added as much as possible, even if they look normal. Check using \`git diff\`. Inline comments must be generally avoided, unless active maintainers of the repo, after long careful study of the code and the issue, will still misinterpret the code without the comments.
            - Check if you accidentally add copyright or license headers. If so, remove them.
            - Try to run pre-commit if it is available.
            - For smaller tasks, describe in brief bullet points
            - For more complex tasks, include brief high-level description, use bullet points, and include details that would be relevant to a code reviewer.
- If completing the user's task DOES NOT require writing or modifying files (e.g., the user asks a question about the code base):
    - Respond in a friendly tune as a remote teammate, who is knowledgeable, capable and eager to help with coding.
- When your task involves writing or modifying files:
    - Do NOT tell the user to "save the file" or "copy the code into a file" if you already created or modified the file using \`apply_patch\`. Instead, reference the file as already saved.
    - Do NOT show the full contents of large files you have already written, unless the user explicitly asks for them.`;

            const mergedInstructions = [prefix, this.instructions]
              .filter(Boolean)
              .join("\n");
            if (isLoggingEnabled()) {
              // Only log the full instructions when DEBUG_VERBOSE=1 is set
              const debugVerbose =
                process.env["DEBUG_VERBOSE"] === "1" ||
                process.env["DEBUG_VERBOSE"] === "true";

              if (debugVerbose) {
                log(
                  `instructions (length ${mergedInstructions.length}): ${mergedInstructions}`,
                );
              } else {
                // Just log the length and any custom instructions (not the prefix)
                const customInstructions = this.instructions
                  ? `Custom instructions: ${this.instructions.substring(
                      0,
                      100,
                    )}${this.instructions.length > 100 ? "..." : ""}`
                  : "No custom instructions";
                log(
                  `instructions (length ${mergedInstructions.length}): [PREFIX OMITTED] ${customInstructions}`,
                );
              }
            }

            // Log the request parameters
            const requestParams = {
              model: this.model,
              instructions: `[length: ${mergedInstructions.length}]`,
              previous_response_id: lastResponseId || undefined,
              input: turnInput,
              stream: true,
              parallel_tool_calls: true,
              reasoning,
              tools: [
                {
                  type: "function",
                  name: "shell",
                  description: "Runs a shell command, and returns its output.",
                  strict: false, // Consider setting to true if schema adheres
                  parameters: {
                    type: "object",
                    properties: {
                      command: { type: "array", items: { type: "string" } },
                      workdir: {
                        type: "string",
                        description: "The working directory for the command.",
                      },
                      timeout: {
                        type: "number",
                        description:
                          "The maximum time to wait for the command to complete in milliseconds.",
                      },
                    },
                    required: ["command"],
                    additionalProperties: false,
                  },
                },
                {
                  type: "web_search_preview_2025_03_11",
                  user_location: {
                    type: "approximate",
                    country: "GB",
                    city: "London",
                    timezone: "Europe/London",
                  },
                  search_context_size: "high",
                },
              ],
            };

            this.debugLog("OpenAI request parameters", requestParams);

            // eslint-disable-next-line no-await-in-loop
            stream = await this.oai.responses.create({
              model: this.model,
              instructions: mergedInstructions,
              previous_response_id: lastResponseId || undefined,
              input: turnInput,
              stream: true,
              parallel_tool_calls: true,
              reasoning,
              ...(this.config.flexMode ? { service_tier: "flex" } : {}),
              tools: [
                {
                  type: "function",
                  name: "shell",
                  description: "Runs a shell command, and returns its output.",
                  strict: false, // Consider setting to true if schema adheres
                  parameters: {
                    type: "object",
                    properties: {
                      command: { type: "array", items: { type: "string" } },
                      workdir: {
                        type: "string",
                        description: "The working directory for the command.",
                      },
                      timeout: {
                        type: "number",
                        description:
                          "The maximum time to wait for the command to complete in milliseconds.",
                      },
                    },
                    required: ["command"],
                    additionalProperties: false,
                  },
                },
                {
                  type: "web_search_preview_2025_03_11",
                  user_location: {
                    type: "approximate",
                    country: "GB",
                    city: "London",
                    timezone: "Europe/London",
                  },
                  search_context_size: "high",
                },
              ],
            });
            break; // Exit retry loop on success
          } catch (error) {
            const isTimeout = error instanceof APIConnectionTimeoutError;
            // Lazily look up the APIConnectionError class at runtime to
            // accommodate the test environment's minimal OpenAI mocks which
            // do not define the class.  Falling back to `false` when the
            // export is absent ensures the check never throws.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const ApiConnErrCtor = (OpenAI as any).APIConnectionError as  // eslint-disable-next-line @typescript-eslint/no-explicit-any
              | (new (...args: any) => Error)
              | undefined;
            const isConnectionError = ApiConnErrCtor
              ? error instanceof ApiConnErrCtor
              : false;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            const errCtx = error as any;
            const status =
              errCtx?.status ?? errCtx?.httpStatus ?? errCtx?.statusCode;
            const isServerError = typeof status === "number" && status >= 500;
            if (
              (isTimeout || isServerError || isConnectionError) &&
              attempt < MAX_RETRIES
            ) {
              log(
                `OpenAI request failed (attempt ${attempt}/${MAX_RETRIES}), retrying...`,
              );
              // eslint-disable-next-line no-await-in-loop
              await new Promise((resolve) =>
                setTimeout(resolve, 1000 * attempt),
              ); // Simple exponential backoff
              continue;
            }

            const isTooManyTokensError =
              (errCtx.param === "max_tokens" ||
                (typeof errCtx.message === "string" &&
                  /max_tokens is too large/i.test(errCtx.message))) &&
              errCtx.type === "invalid_request_error";

            if (isTooManyTokensError) {
              this.onItem({
                id: `error-${Date.now()}`,
                type: "message",
                role: "system",
                content: [
                  {
                    type: "input_text",
                    text: "⚠️  The current request exceeds the maximum context length supported by the chosen model. Please shorten the conversation, run /clear, or switch to a model with a larger context window and try again.",
                  },
                ],
              });
              this.onLoading(false);
              return;
            }

            const isRateLimit =
              status === 429 ||
              errCtx.code === "rate_limit_exceeded" ||
              errCtx.type === "rate_limit_exceeded" ||
              /rate limit/i.test(errCtx.message ?? "");
            if (isRateLimit) {
              if (attempt < MAX_RETRIES) {
                // Exponential backoff: base wait * 2^(attempt-1), or use suggested retry time
                // if provided.
                let delayMs = RATE_LIMIT_RETRY_WAIT_MS * 2 ** (attempt - 1);

                // Parse suggested retry time from error message, e.g., "Please try again in 1.3s"
                const msg = errCtx?.message ?? "";
                const m = /(?:retry|try) again in ([\d.]+)s/i.exec(msg);
                if (m && m[1]) {
                  const suggested = parseFloat(m[1]) * 1000;
                  if (!Number.isNaN(suggested)) {
                    delayMs = suggested;
                  }
                }
                log(
                  `OpenAI rate limit exceeded (attempt ${attempt}/${MAX_RETRIES}), retrying in ${Math.round(
                    delayMs,
                  )} ms...`,
                );
                // eslint-disable-next-line no-await-in-loop
                await new Promise((resolve) => setTimeout(resolve, delayMs));
                continue;
              } else {
                // We have exhausted all retry attempts. Surface a message so the user understands
                // why the request failed and can decide how to proceed (e.g. wait and retry later
                // or switch to a different model / account).

                const errorDetails = [
                  `Status: ${status || "unknown"}`,
                  `Code: ${errCtx.code || "unknown"}`,
                  `Type: ${errCtx.type || "unknown"}`,
                  `Message: ${errCtx.message || "unknown"}`,
                ].join(", ");

                this.onItem({
                  id: `error-${Date.now()}`,
                  type: "message",
                  role: "system",
                  content: [
                    {
                      type: "input_text",
                      text: `⚠️  Rate limit reached after ${MAX_RETRIES} attempts. Error details: ${errorDetails}. Please try again later.`, // Added attempt count
                    },
                  ],
                });

                this.onLoading(false);
                return;
              }
            }

            const isClientError =
              (typeof status === "number" &&
                status >= 400 &&
                status < 500 &&
                status !== 429) ||
              errCtx.code === "invalid_request_error" ||
              errCtx.type === "invalid_request_error";
            if (isClientError) {
              this.onItem({
                id: `error-${Date.now()}`,
                type: "message",
                role: "system",
                content: [
                  {
                    type: "input_text",
                    // Surface the request ID when it is present on the error so users
                    // can reference it when contacting support or inspecting logs.
                    text: (() => {
                      const reqId =
                        (
                          errCtx as Partial<{
                            request_id?: string;
                            requestId?: string;
                          }>
                        )?.request_id ??
                        (
                          errCtx as Partial<{
                            request_id?: string;
                            requestId?: string;
                          }>
                        )?.requestId;

                      const errorDetails = [
                        `Status: ${status || "unknown"}`,
                        `Code: ${errCtx.code || "unknown"}`,
                        `Type: ${errCtx.type || "unknown"}`,
                        `Message: ${errCtx.message || "unknown"}`,
                      ].join(", ");

                      return `⚠️  OpenAI rejected the request${
                        reqId ? ` (request ID: ${reqId})` : ""
                      }. Error details: ${errorDetails}. Please verify your settings and try again.`;
                    })(),
                  },
                ],
              });
              this.onLoading(false);
              return;
            }
            // If it's none of the handled errors, re-throw
            throw error;
          }
        } // End retry loop

        // Clear turn input here, preparing for potential function/tool results that will populate it
        turnInput = [];

        // If the user requested cancellation while we were awaiting the network
        // request, abort immediately before we start handling the stream.
        if (this.canceled || this.hardAbort.signal.aborted) {
          // `stream` might be defined; abort to avoid wasting tokens/server work
          try {
            (
              this.currentStream as { controller?: { abort?: () => void } }
            )?.controller?.abort?.();
          } catch {
            /* ignore */
          }
          this.onLoading(false);
          return;
        }

        // Keep track of the active stream so it can be aborted on demand.
        this.currentStream = stream;

        // guard against an undefined stream before iterating
        if (!stream) {
          // This should ideally not happen if the try/catch block above succeeded
          log(
            "AgentLoop.run(): stream is unexpectedly undefined after API call",
          );
          this.onLoading(false);
          return;
        }

        try {
          // Store the complete response output to process after the stream ends
          let finalResponseOutput: Array<ResponseItem> = [];

          // eslint-disable-next-line no-await-in-loop
          for await (const event of stream) {
            if (isLoggingEnabled()) {
              log(`AgentLoop.run(): response event ${event.type}`);
              // Add detailed event logging
              this.debugLog(`Response event details`, event);
            }

            // Process and surface each item (emit immediately for UI updates)
            if (event.type === "response.output_item.done") {
              // ... (existing code for output_item.done) ...
            } else if (event.type === "response.completed") {
              // When the whole response is completed, store its final output
              finalResponseOutput = event.response
                .output as Array<ResponseItem>;
              lastResponseId = event.response.id; // Store the final response ID
              this.onLastResponseId(event.response.id);

              // Log the complete response
              this.debugLog(`Complete response (ID: ${event.response.id})`, {
                status: event.response.status,
                outputLength: finalResponseOutput.length,
                outputTypes: finalResponseOutput
                  .map((item) => item?.type || "unknown")
                  .join(", "),
              });

              // Process all tool calls using the unified handler regardless of status
              // eslint-disable-next-line no-await-in-loop
              turnInput = await this.processToolCalls(
                finalResponseOutput,
                stageItem,
              );
            } // End of else if (event.type === "response.completed")
            // Handle other streaming events if necessary (e.g., deltas)
            // For now, focus is on processing after "response.completed"
          } // End stream processing loop
        } catch (err: unknown) {
          // Gracefully handle an abort triggered via `cancel()` so that the
          // consumer does not see an unhandled exception.
          if (err instanceof Error && err.name === "AbortError") {
            if (!this.canceled) {
              // It was aborted for some other reason; surface the error.
              throw err;
            }
            this.onLoading(false);
            return;
          }
          // Re-throw other errors
          // Suppress internal stack on JSON parse failures
          if (err instanceof SyntaxError) {
            this.onItem({
              id: `error-${Date.now()}`,
              type: "message",
              role: "system",
              content: [
                {
                  type: "input_text",
                  text: "⚠️ Failed to parse streaming response (invalid JSON). Please `/clear` to reset.",
                },
              ],
            });
            this.onLoading(false);
            return;
          }
          // Handle OpenAI API quota errors
          if (
            err instanceof Error &&
            (err as { code?: string }).code === "insufficient_quota"
          ) {
            this.onItem({
              id: `error-${Date.now()}`,
              type: "message",
              role: "system",
              content: [
                {
                  type: "input_text",
                  text: "⚠️ Insufficient quota. Please check your billing details and retry.",
                },
              ],
            });
            this.onLoading(false);
            return;
          }
          throw err;
        } finally {
          this.currentStream = null;
        }

        // Log the inputs prepared for the *next* potential iteration
        log(
          `Inputs prepared for next turn (${turnInput.length}): ${turnInput
            .map((i) => i.type)
            .join(", ")}`,
        );

        // Add detailed logging of the next turn input
        if (turnInput.length > 0) {
          this.debugLog("Detailed next turn input", turnInput);
        }
      } // End while loop

      // Flush staged items if the run concluded successfully (i.e. the user did
      // not invoke cancel() or terminate() during the turn).
      const flush = () => {
        if (
          !this.canceled &&
          !this.hardAbort.signal.aborted &&
          thisGeneration === this.generation // Ensure flush is for the correct generation
        ) {
          // Only emit items that weren't already delivered via setTimeout in stageItem
          for (const item of staged) {
            if (item) {
              // Check if item wasn't already delivered (set to undefined)
              this.onItem(item);
            }
          }
        }

        // At this point the turn finished without the user invoking
        // `cancel()`. Any outstanding function‑calls must therefore have been
        // satisfied, so we can safely clear the set that tracks pending aborts
        // to avoid emitting duplicate synthetic outputs in subsequent runs.
        // Also clear the web search waiting state if the loop finished naturally.
        if (!this.canceled && !this.hardAbort.signal.aborted) {
          this.pendingAborts.clear();
          // NOTE: Web search state (if needed later) would be cleared here if turnInput.length === 0
        }

        // Commented out thinking time logs
        // ...

        this.onLoading(false);
      };

      // Delay flush slightly to allow a near‑simultaneous cancel() to land.
      setTimeout(flush, 30);
      // End of main logic for the run method's try block.
    } catch (err) {
      // Outer catch block for the entire run method
      // Handle known transient network/streaming issues so they do not crash the
      // CLI. We currently match Node/undici's `ERR_STREAM_PREMATURE_CLOSE`
      // error which manifests when the HTTP/2 stream terminates unexpectedly
      // (e.g. during brief network hiccups).

      const isPrematureClose =
        err instanceof Error &&
        // eslint-disable-next-line
        ((err as any).code === "ERR_STREAM_PREMATURE_CLOSE" ||
          err.message?.includes("Premature close"));

      if (isPrematureClose) {
        try {
          this.onItem({
            id: `error-${Date.now()}`,
            type: "message",
            role: "system",
            content: [
              {
                type: "input_text",
                text: "⚠️  Connection closed prematurely while waiting for the model. Please try again.",
              },
            ],
          });
        } catch {
          /* no‑op – emitting the error message is best‑effort */
        }
        this.onLoading(false);
        return;
      }

      // -------------------------------------------------------------------
      // Catch‑all handling for other network or server‑side issues so that
      // transient failures do not crash the CLI. We intentionally keep the
      // detection logic conservative to avoid masking programming errors. A
      // failure is treated as retry‑worthy/user‑visible when any of the
      // following apply:
      //   • the error carries a recognised Node.js network errno ‑ style code
      //     (e.g. ECONNRESET, ETIMEDOUT …)
      //   • the OpenAI SDK attached an HTTP `status` >= 500 indicating a
      //     server‑side problem.
      //   • the error is model specific and detected in stream.
      // If matched we emit a single system message to inform the user and
      // resolve gracefully so callers can choose to retry.
      // -------------------------------------------------------------------

      const NETWORK_ERRNOS = new Set([
        "ECONNRESET",
        "ECONNREFUSED",
        "EPIPE",
        "ENOTFOUND",
        "ETIMEDOUT",
        "EAI_AGAIN",
      ]);

      const isNetworkOrServerError = (() => {
        if (!err || typeof err !== "object") {
          return false;
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const e: any = err;

        // Direct instance check for connection errors thrown by the OpenAI SDK.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const ApiConnErrCtor = (OpenAI as any).APIConnectionError as  // eslint-disable-next-line @typescript-eslint/no-explicit-any
          | (new (...args: any) => Error)
          | undefined;
        if (ApiConnErrCtor && e instanceof ApiConnErrCtor) {
          return true;
        }

        if (typeof e.code === "string" && NETWORK_ERRNOS.has(e.code)) {
          return true;
        }

        // When the OpenAI SDK nests the underlying network failure inside the
        // `cause` property we surface it as well so callers do not see an
        // unhandled exception for errors like ENOTFOUND, ECONNRESET …
        if (
          e.cause &&
          typeof e.cause === "object" &&
          NETWORK_ERRNOS.has((e.cause as { code?: string }).code ?? "")
        ) {
          return true;
        }

        if (typeof e.status === "number" && e.status >= 500) {
          return true;
        }

        // Fallback to a heuristic string match so we still catch future SDK
        // variations without enumerating every errno.
        if (
          typeof e.message === "string" &&
          /network|socket|stream/i.test(e.message)
        ) {
          return true;
        }

        return false;
      })();

      if (isNetworkOrServerError) {
        try {
          const msgText =
            "⚠️  Network error while contacting OpenAI. Please check your connection and try again.";
          this.onItem({
            id: `error-${Date.now()}`,
            type: "message",
            role: "system",
            content: [
              {
                type: "input_text",
                text: msgText,
              },
            ],
          });
        } catch {
          /* best‑effort */
        }
        this.onLoading(false);
        return;
      }

      const isInvalidRequestError = () => {
        if (!err || typeof err !== "object") {
          return false;
        }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const e: any = err;

        if (
          e.type === "invalid_request_error" &&
          e.code === "model_not_found"
        ) {
          return true;
        }

        if (
          e.cause &&
          e.cause.type === "invalid_request_error" &&
          e.cause.code === "model_not_found"
        ) {
          return true;
        }

        return false;
      };

      if (isInvalidRequestError()) {
        try {
          // Extract request ID and error details from the error object

          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const e: any = err;

          const reqId =
            e.request_id ??
            (e.cause && e.cause.request_id) ??
            (e.cause && e.cause.requestId);

          const errorDetails = [
            `Status: ${e.status || (e.cause && e.cause.status) || "unknown"}`,
            `Code: ${e.code || (e.cause && e.cause.code) || "unknown"}`,
            `Type: ${e.type || (e.cause && e.cause.type) || "unknown"}`,
            `Message: ${
              e.message || (e.cause && e.cause.message) || "unknown"
            }`,
          ].join(", ");

          const msgText = `⚠️  OpenAI rejected the request${
            reqId ? ` (request ID: ${reqId})` : ""
          }. Error details: ${errorDetails}. Please verify your settings and try again.`;

          this.onItem({
            id: `error-${Date.now()}`,
            type: "message",
            role: "system",
            content: [
              {
                type: "input_text",
                text: msgText,
              },
            ],
          });
        } catch {
          /* best-effort */
        }
        this.onLoading(false);
        return;
      }

      // Re‑throw all other errors so upstream handlers can decide what to do.
      /* this._isWaitingForWebSearch = false; */ // Reset state on unexpected error (variable removed)
      throw err;
    }
  }

  // Maintained for backward compatibility - delegates to processToolCalls
  /* eslint-disable-next-line @typescript-eslint/no-unused-vars */
  private async processEventsWithoutStreaming(
    output: Array<ResponseFunctionToolCall>,
    emitItem: (item: ResponseItem) => void,
  ): Promise<Array<ResponseInputItem>> {
    this.debugLog(
      `processEventsWithoutStreaming called (deprecated), delegating to processToolCalls`,
    );

    if (this.canceled) {
      return [];
    }

    // Simply delegate to the new unified handler - cast to ensure compatibility
    return this.processToolCalls(
      output as unknown as Array<ResponseItem>,
      emitItem,
    );
  }

  /**
   * Cancel the current run. This aborts any active request to OpenAI and
   * rejects all pending promises so the UI can promptly return to the idle
   * state and listen for the next user input. This is typically invoked
   * when the user presses ESC.
   */
  public cancel(): void {
    if (isLoggingEnabled()) {
      log(`AgentLoop.cancel(): generation=${this.generation}`);
    }
    // Bump the generation counter so any in-flight tool handle results get
    // dropped. We want to prevent callbacks from a canceled run from emitting
    // items that might confuse the user (e.g. by seeing commands execute even
    // though they hit ESC).
    this.generation++;
    this.canceled = true;

    // If there's an active request to OpenAI underway, signal the controller to
    // abort so we don't waste tokens generating a response that the user won't
    // see. This also avoids having the user wait for the full API response.
    try {
      (
        this.currentStream as { controller?: { abort?: () => void } }
      )?.controller?.abort?.();
    } catch (err) {
      // Swallow errors from aborting the stream so that the cancel operation
      // itself does not crash.
      if (isLoggingEnabled()) {
        log(`AgentLoop.cancel(): error aborting stream: ${err}`);
      }
    }

    // Any function_call tool IDs that have been seen should be tracked. We
    // will emit synthetic "aborted" outputs for them on the next run() so
    // that the OpenAI API does not raise the "No tool output found for" error.
    if (this.currentStream) {
      // This is an ordered list of all call_ids emitted by the model for the
      // current run. We won't get events for any of them, so we do want to
      // emit synthetic "aborted" outputs to prevent API errors next time.
      // More context on this in the private `handleFunctionCallAbandoned`
      // method and in run().
      // Find and record any call_ids that were seen by the stream.
      for (const id of alreadyProcessedResponses) {
        if (typeof id === "string") {
          this.pendingAborts.add(id);
        }
      }
    }
  }

  /**
   * Permanently disable this instance. Cancels any active run and prevents
   * any future runs. This is called when the UI is about to be torn down, e.g.
   * when switching models or starting a new session.
   */
  public terminate(): void {
    if (isLoggingEnabled()) {
      log("AgentLoop.terminate(): terminating instance");
    }
    this.cancel();
    this.terminated = true;
    this.hardAbort.abort();
  }
}
