import * as fsSync from "fs";
import * as fs from "fs/promises";
import * as os from "os";
import * as path from "path";

interface Logger {
  /** Checking this can be used to avoid constructing a large log message. */
  isLoggingEnabled(): boolean;

  log(message: string): void;
}

class AsyncLogger implements Logger {
  private queue: Array<string> = [];
  private isWriting: boolean = false;
  private localFilePath: string | null = null;

  constructor(private filePath: string, useLocalLogs: boolean = false) {
    this.filePath = filePath;

    // Set up local directory logging if enabled
    if (useLocalLogs) {
      try {
        // Use current working directory for local logs
        const cwd = process.cwd();
        this.localFilePath = path.join(cwd, `codex-log-${now()}.log`);
        // Create an empty file to start
        fsSync.writeFileSync(this.localFilePath, "");
        // Use process.stderr directly instead of console
        process.stderr.write(
          `Local logging enabled. Logs will be written to: ${this.localFilePath}\n`,
        );
      } catch (err) {
        // Use process.stderr directly instead of console
        process.stderr.write(`Failed to set up local logging: ${err}\n`);
        this.localFilePath = null;
      }
    }
  }

  isLoggingEnabled(): boolean {
    return true;
  }

  log(message: string): void {
    const entry = `[${now()}] ${message}\n`;
    this.queue.push(entry);
    this.maybeWrite();
  }

  private async maybeWrite(): Promise<void> {
    if (this.isWriting || this.queue.length === 0) {
      return;
    }

    this.isWriting = true;
    const messages = this.queue.join("");
    this.queue = [];

    try {
      // Write to the standard log location
      await fs.appendFile(this.filePath, messages);

      // Also write to local file if enabled
      if (this.localFilePath) {
        await fs.appendFile(this.localFilePath, messages);
      }
    } catch (err) {
      // Log error to stderr directly instead of using console
      process.stderr.write(`Error writing logs: ${err}\n`);
    } finally {
      this.isWriting = false;
    }

    this.maybeWrite();
  }
}

class EmptyLogger implements Logger {
  isLoggingEnabled(): boolean {
    return false;
  }

  log(_message: string): void {
    // No-op
  }
}

function now() {
  const date = new Date();
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  const hours = String(date.getHours()).padStart(2, "0");
  const minutes = String(date.getMinutes()).padStart(2, "0");
  const seconds = String(date.getSeconds()).padStart(2, "0");
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}`;
}

let logger: Logger;

/**
 * Creates a .log file for this session, but also symlinks codex-cli-latest.log
 * to the current log file so you can reliably run:
 *
 * - Mac/Windows: `tail -F "$TMPDIR/oai-codex/codex-cli-latest.log"`
 * - Linux: `tail -F ~/.local/oai-codex/codex-cli-latest.log`
 *
 * When LOCAL_LOGS=1 is set in the environment, it will also create logs in the
 * current working directory.
 */
export function initLogger(): Logger {
  if (logger) {
    return logger;
  } else if (!process.env["DEBUG"]) {
    logger = new EmptyLogger();
    return logger;
  }

  // Check if local logging is enabled
  const useLocalLogs =
    process.env["LOCAL_LOGS"] === "1" || process.env["LOCAL_LOGS"] === "true";

  const isMac = process.platform === "darwin";
  const isWin = process.platform === "win32";

  // On Mac and Windows, os.tmpdir() returns a user-specifc folder, so prefer
  // it there. On Linux, use ~/.local/oai-codex so logs are not world-readable.
  const logDir =
    isMac || isWin
      ? path.join(os.tmpdir(), "oai-codex")
      : path.join(os.homedir(), ".local", "oai-codex");
  fsSync.mkdirSync(logDir, { recursive: true });
  const logFile = path.join(logDir, `codex-cli-${now()}.log`);
  // Write the empty string so the file exists and can be tail'd.
  fsSync.writeFileSync(logFile, "");

  // Symlink to codex-cli-latest.log on UNIX because Windows is funny about
  // symlinks.
  if (!isWin) {
    const latestLink = path.join(logDir, "codex-cli-latest.log");
    try {
      fsSync.symlinkSync(logFile, latestLink, "file");
    } catch (err: unknown) {
      const error = err as NodeJS.ErrnoException;
      if (error.code === "EEXIST") {
        fsSync.unlinkSync(latestLink);
        fsSync.symlinkSync(logFile, latestLink, "file");
      } else {
        throw err;
      }
    }
  }

  logger = new AsyncLogger(logFile, useLocalLogs);
  return logger;
}

export function log(message: string): void {
  (logger ?? initLogger()).log(message);
}

export function isLoggingEnabled(): boolean {
  return (logger ?? initLogger()).isLoggingEnabled();
}
