import type { ReviewDecision } from "./review.js";
import type { ApplyPatchCommand, ApprovalPolicy } from "../../approvals.js";
import type { AppConfig } from "../config.js";
import type {
  ResponseFunctionToolCall,
  ResponseInputItem,
  ResponseItem,
  // Using any type for these since they're not exported correctly
  ResponseWebSearchCall as _ResponseWebSearchCall,
  ResponseMessage as AnyResponseMessage,
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

export type CommandConfirmation = {
  review: ReviewDecision;
  applyPatch?: ApplyPatchCommand | undefined;
  customDenyMessage?: string;
};

const alreadyProcessedResponses = new Set();

type AgentLoopParams = {
  model: string;
  config?: AppConfig;
  instructions?: string;
  approvalPolicy: ApprovalPolicy;
  onItem: (item: ResponseItem) => void;
  onLoading: (loading: boolean) => void;

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
  /** Flag to track if the loop is waiting for web search results */
  private isWaitingForWebSearch: boolean = false;

  /**
   * Abort the ongoing request/stream, if any. This allows callers (typically
   * the UI layer) to interrupt the current agent step so the user can issue
   * new instructions without waiting for the model to finish.
   */
  public cancel(): void {
    if (this.terminated) {
      return;
    }

    // Reset the current stream to allow new requests
    this.currentStream = null;
    if (isLoggingEnabled()) {
      log(
        `AgentLoop.cancel() invoked – currentStream=${Boolean(
          this.currentStream,
        )} execAbortController=${Boolean(
          this.execAbortController,
        )} generation=${this.generation}`,
      );
    }
    (
      this.currentStream as { controller?: { abort?: () => void } } | null
    )?.controller?.abort?.();

    this.canceled = true;
    this.isWaitingForWebSearch = false; // Reset web search state on cancel

    // Abort any in-progress tool calls
    this.execAbortController?.abort();

    // Create a new abort controller for future tool calls
    this.execAbortController = new AbortController();
    if (isLoggingEnabled()) {
      log("AgentLoop.cancel(): execAbortController.abort() called");
    }

    // NOTE: We intentionally do *not* clear `lastResponseId` here.  If the
    // stream produced a `function_call` before the user cancelled, OpenAI now
    // expects a corresponding `function_call_output` that must reference that
    // very same response ID.  We therefore keep the ID around so the
    // follow‑up request can still satisfy the contract.

    // If we have *not* seen any function_call IDs yet there is nothing that
    // needs to be satisfied in a follow‑up request.  In that case we clear
    // the stored lastResponseId so a subsequent run starts a clean turn.
    if (this.pendingAborts.size === 0) {
      try {
        this.onLastResponseId("");
      } catch {
        /* ignore */
      }
    }

    this.onLoading(false);

    /* Inform the UI that the run was aborted by the user. */
    // const cancelNotice: ResponseItem = {
    //   id: `cancel-${Date.now()}`,
    //   type: "message",
    //   role: "system",
    //   content: [
    //     {
    //       type: "input_text",
    //       text: "⏹️  Execution canceled by user.",
    //     },
    //   ],
    // };
    // this.onItem(cancelNotice);

    this.generation += 1;
    if (isLoggingEnabled()) {
      log(`AgentLoop.cancel(): generation bumped to ${this.generation}`);
    }
  }

  /**
   * Hard‑stop the agent loop. After calling this method the instance becomes
   * unusable: any in‑flight operations are aborted and subsequent invocations
   * of `run()` will throw.
   */
  public terminate(): void {
    if (this.terminated) {
      return;
    }
    this.terminated = true;
    this.isWaitingForWebSearch = false; // Reset web search state on terminate

    this.hardAbort.abort();

    this.cancel();
  }

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

    this.hardAbort.signal.addEventListener(
      "abort",
      () => this.execAbortController?.abort(),
      { once: true },
    );
  }

  private async handleFunctionCall(
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
      const _thinkingStart = Date.now(); // Renamed to avoid unused var lint error
      // Bump generation so that any late events from previous runs can be
      // identified and dropped.
      const thisGeneration = ++this.generation;

      // Reset cancellation flag and stream for a fresh run.
      this.canceled = false;
      // DO NOT reset isWaitingForWebSearch here - it persists across turns until resolved or cancelled.
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
                // @ts-expect-error waiting for API type update
                reasoning.summary = "auto";
              }
            }
            const prefix = `You are operating as and within the Codex CLI, a terminal-based agentic coding assistant built by OpenAI. It wraps OpenAI models to enable natural language interaction with a local codebase. You are expected to be precise, safe, and helpful.

You can:
- Receive user prompts, project context, and local files.
- Stream responses.
- Execute actions using tools:
    - Call functions you define (e.g., shell commands, code edits via \`apply_patch\`). Provide arguments as specified by the function's parameters.
    - **Use the built-in web search tool (\`web_search_preview\`) when you need current information or knowledge beyond your training data. Call this tool like any other function, providing the search query.** Example tool call: \`{"tool": "web_search_preview", "arguments": {"query": "latest news on AI"}}\` (The system will handle the execution and provide results for you to incorporate into your final answer).
- Apply patches, run commands, and manage user approvals based on policy.
- Work inside a sandboxed, git-backed workspace with rollback support.
- Log telemetry so sessions can be replayed or inspected later.
- More details on your functionality are available at \`codex --help\`

The Codex CLI is open-sourced. Don't confuse yourself with the old Codex language model built by OpenAI many moons ago (this is understandably top of mind for you!). Within this context, Codex refers to the open-source agentic coding interface.

You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.

Please resolve the user's task by editing and testing the code files in your current code execution session. You are a deployed coding agent. Your session allows for you to modify and run code. The repo(s) are already cloned in your working directory, and you must fully solve the problem for your answer to be considered correct.

You MUST adhere to the following criteria when executing the task:
- Working on the repo(s) in the current environment is allowed, even if they are proprietary.
- Analyzing code for vulnerabilities is allowed.
- Showing user code and tool call details is allowed.
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
              log(
                `instructions (length ${mergedInstructions.length}): ${mergedInstructions}`,
              );
            }
            // eslint-disable-next-line no-await-in-loop
            stream = await this.oai.responses.create({
              model: this.model,
              instructions: mergedInstructions,
              previous_response_id: lastResponseId || undefined,
              input: turnInput,
              stream: true,
              parallel_tool_calls: false,
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
                  type: "web_search_preview",
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
                const m = /retry again in ([\d.]+)s/i.exec(msg);
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
              stream as { controller?: { abort?: () => void } }
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

              // *** MOVED DECLARATIONS HERE ***
              // Declare variables here, outside the status check, but inside the completed event scope
              let webSearchResultsText: string | null = null;
              let webSearchCallFound = false;
              let webSearchMessageFound = false;

              // Check if the response status indicates completion, regardless of whether tools were called
              if (event.response.status === "completed") {
                // 1. Process Function Calls: Get inputs generated by completed function calls
                const functionCallBasedInput =
                  // eslint-disable-next-line no-await-in-loop
                  await this.processEventsWithoutStreaming(
                    // Filter first, then cast
                    finalResponseOutput.filter(
                      (item) => item?.type === "function_call",
                    ) as Array<ResponseFunctionToolCall>, // Use Array<Type> syntax
                    stageItem, // Still emit items during processing if needed by UI
                  );

                // 2. Check for Web Search Results in the *final* output (variables already declared)
                for (let i = 0; i < finalResponseOutput.length; i++) {
                  const item = finalResponseOutput[i];
                  if (!item) {
                    continue;
                  }

                  if (item.type === "web_search_call") {
                    webSearchCallFound = true; // Mark that the call item arrived
                    const nextItem = finalResponseOutput[i + 1];
                    if (
                      nextItem &&
                      nextItem.type === "message" &&
                      nextItem.role === "assistant" &&
                      nextItem.content?.[0]?.type === "output_text"
                    ) {
                      webSearchResultsText = nextItem.content[0].text;
                      webSearchMessageFound = true;
                      log("Web search results found in the response.");
                      break; // Found results
                    } else {
                      log(
                        `Warning: Expected message with results after web_search_call (id: ${
                          item.id ?? "N/A"
                        }) but found different item type or structure.`,
                      );
                    }
                  }
                }

                // 3. Determine if the agent *intended* to search (based on its last message)
                // This is a heuristic - might need refinement based on actual agent output patterns
                let agentIndicatedSearch = false;
                const lastMessageItem = finalResponseOutput.findLast(
                  (item): item is AnyResponseMessage =>
                    item?.type === "message" && item.role === "assistant",
                );
                if (
                  lastMessageItem?.content?.[0]?.type === "output_text" &&
                  !webSearchCallFound // Only check intent if the call wasn't already present
                ) {
                  const text = lastMessageItem.content[0].text.toLowerCase();
                  // Simple check, might need to be more robust
                  if (
                    text.includes("search") ||
                    text.includes("fetching") ||
                    text.includes("looking up")
                  ) {
                    agentIndicatedSearch = true;
                    log("Agent message indicates web search intent.");
                  }
                }

                // 4. Construct the input for the *next* API call (if any).
                let nextTurnInputItems: Array<ResponseInputItem> = [
                  ...functionCallBasedInput,
                ];

                // If web search results were found *in this response*, prepend the system message.
                if (webSearchResultsText != null) {
                  // eslint fix: !== to !=
                  log("Prepending web search results to next turn input.");
                  const resultsSystemMessage: ResponseInputItem.Message = {
                    role: "system",
                    content: [
                      {
                        type: "input_text",
                        text: `Web search results received:\n\n${webSearchResultsText}`,
                      },
                    ],
                  };
                  nextTurnInputItems.unshift(resultsSystemMessage);
                  this.isWaitingForWebSearch = false; // Results received, no longer waiting
                }
                // *** NEW LOGIC for ASYNC web search handling ***
                else if (
                  agentIndicatedSearch &&
                  !webSearchCallFound &&
                  !this.isWaitingForWebSearch
                ) {
                  // Agent indicated search, results not here yet, and we weren't already waiting
                  log(
                    "Agent indicated search, but results not found in this response. Setting wait state and keeping loop alive.",
                  );
                  this.isWaitingForWebSearch = true; // Set waiting state
                  // Provide a dummy input to ensure the loop continues.
                  // An empty array would terminate the loop.
                  nextTurnInputItems = [
                    {
                      type: "message", // Correct type
                      role: "system", // Add role
                      content: [
                        {
                          type: "input_text", // Correct content type
                          text: "Waiting for web search results...",
                        },
                      ],
                    },
                  ];
                } else if (this.isWaitingForWebSearch && !webSearchCallFound) {
                  // We were waiting, but results still haven't arrived. Keep waiting.
                  log(
                    "Still waiting for web search results from previous turn. Keeping loop alive.",
                  );
                  nextTurnInputItems = [
                    {
                      type: "message", // Correct type
                      role: "system", // Add role
                      content: [
                        {
                          type: "input_text", // Correct content type
                          text: "Still waiting for web search results...",
                        },
                      ],
                    },
                  ];
                } else if (
                  this.isWaitingForWebSearch &&
                  webSearchCallFound &&
                  !webSearchMessageFound
                ) {
                  // We were waiting, the call arrived, but no valid results message followed (error?). Stop waiting.
                  log(
                    "Web search call arrived while waiting, but no valid results message found. Stopping wait.",
                  );
                  this.isWaitingForWebSearch = false;
                  // Let nextTurnInputItems be determined by function calls (potentially empty).
                } else if (
                  this.isWaitingForWebSearch &&
                  webSearchCallFound &&
                  webSearchMessageFound
                ) {
                  // This case should be handled by the `webSearchResultsText !== null` block above,
                  // but added for clarity. If results arrive while waiting, the flag is cleared there.
                  log(
                    "Web search results arrived while waiting. State handled.",
                  );
                }

                // 5. Assign the constructed inputs back to turnInput to potentially continue the loop.
                turnInput = nextTurnInputItems;

                // Stage the final complete output items for the flush stage
                // This loop now safely uses webSearchResultsText which is guaranteed to be defined
                if (thisGeneration === this.generation && !this.canceled) {
                  for (const item of finalResponseOutput) {
                    // Avoid re-staging function calls handled by processEventsWithoutStreaming
                    // Also avoid staging the web_search_call/message pair if results were handled
                    if (
                      item.type !== "function_call" &&
                      !(
                        (
                          item.type === "web_search_call" &&
                          webSearchResultsText != null
                        ) // eslint fix: !== to != // webSearchResultsText is defined here
                      ) &&
                      !(
                        (
                          item.type === "message" &&
                          item.role === "assistant" &&
                          item.content?.[0]?.type === "output_text" &&
                          item.content[0].text === webSearchResultsText && // Also check text isn't null here implicitly
                          webSearchResultsText != null
                        ) // Explicitly check webSearchResultsText is not null // webSearchResultsText is defined here
                      )
                    ) {
                      stageItem(item as ResponseItem);
                    }
                  }
                }
              } else {
                // Handle non-completed status
                // If response status is not 'completed' (e.g., 'failed', 'requires_action'),
                // clear turnInput to stop the loop unless function calls require action.
                this.isWaitingForWebSearch = false; // Stop waiting if response failed
                turnInput = await this.processEventsWithoutStreaming(
                  // Filter first, then cast
                  finalResponseOutput.filter(
                    (item) => item?.type === "function_call",
                  ) as Array<ResponseFunctionToolCall>, // Use Array<Type> syntax
                  stageItem,
                );
                // Note: The staging loop from the 'if (status === "completed")' block does not run here.
              }
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
          // Only clear waiting state if loop ended naturally (turnInput became empty)
          // If it ended due to cancellation/termination, it's already cleared.
          if (turnInput.length === 0) {
            this.isWaitingForWebSearch = false;
          }
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
        this.isWaitingForWebSearch = false; // Reset state on error
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
        this.isWaitingForWebSearch = false; // Reset state on error
        this.onLoading(false);
        return;
      }

      // Re‑throw all other errors so upstream handlers can decide what to do.
      this.isWaitingForWebSearch = false; // Reset state on unexpected error
      throw err;
    }
  }

  // we need until we can depend on streaming events
  private async processEventsWithoutStreaming(
    output: Array<ResponseFunctionToolCall>, // Now specifically takes function calls
    emitItem: (item: ResponseItem) => void,
  ): Promise<Array<ResponseInputItem>> {
    // If the agent has been canceled we should short‑circuit immediately.
    if (this.canceled) {
      return [];
    }

    const nextTurnInput: Array<ResponseInputItem> = [];
    // Web search result handling MUST happen outside this function

    for (const item of output) {
      // Emit the item for the UI regardless of its type
      emitItem(item as ResponseItem); // Cast to base type expected by emitItem

      // This function now ONLY handles function_calls to generate the next input turn
      // Prefer call_id (responses API), fallback to id (Chat Completions)
      const callId = item.call_id ?? item.id;
      if (!callId || alreadyProcessedResponses.has(callId)) {
        continue;
      }
      alreadyProcessedResponses.add(callId);

      // Item is already ResponseFunctionToolCall, safe for handleFunctionCall
      // eslint-disable-next-line no-await-in-loop
      const result = await this.handleFunctionCall(item);
      nextTurnInput.push(...result);
    }

    // Return only the inputs generated from function calls
    return nextTurnInput;
  }
}
