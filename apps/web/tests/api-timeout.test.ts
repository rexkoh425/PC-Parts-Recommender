import { afterEach, describe, expect, it, vi } from "vitest";
import { getFreshness } from "../lib/api";

function fetchUntilAborted(): typeof fetch {
  return vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
    new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal;
      if (signal?.aborted) {
        reject(signal.reason);
        return;
      }
      signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
    })) as typeof fetch;
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("real API request lifecycle", () => {
  it("aborts a request at its configured deadline and reports a bounded timeout", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", fetchUntilAborted());

    const request = getFreshness({ timeoutMs: 25 });
    const assertion = expect(request).rejects.toEqual(
      expect.objectContaining({
        name: "ApiError",
        status: 408,
        message: "The recommendation service did not respond before the request deadline.",
      }),
    );

    await vi.advanceTimersByTimeAsync(25);
    await assertion;
  });

  it("keeps the deadline active while the response body is being read", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        const signal = init?.signal;
        return {
          ok: true,
          json: () =>
            new Promise((_resolve, reject) => {
              if (signal?.aborted) {
                reject(signal.reason);
                return;
              }
              signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
            }),
        } as Response;
      }),
    );

    const request = getFreshness({ timeoutMs: 25 });
    const assertion = expect(request).rejects.toMatchObject({ name: "ApiError", status: 408 });

    await vi.advanceTimersByTimeAsync(25);
    await assertion;
  });

  it("forwards caller cancellation to the underlying fetch", async () => {
    vi.stubGlobal("fetch", fetchUntilAborted());
    const controller = new AbortController();
    const cancellation = new DOMException("View changed", "AbortError");

    const request = getFreshness({ signal: controller.signal, timeoutMs: 5_000 });
    controller.abort(cancellation);

    await expect(request).rejects.toBe(cancellation);
  });

  it("preserves caller cancellation while an error response body is being read", async () => {
    const controller = new AbortController();
    const cancellation = new DOMException("View changed", "AbortError");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
        const signal = init?.signal;
        return {
          ok: false,
          status: 503,
          json: () =>
            new Promise((_resolve, reject) => {
              if (signal?.aborted) {
                reject(signal.reason);
                return;
              }
              signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
            }),
        } as Response;
      }),
    );

    const request = getFreshness({ signal: controller.signal, timeoutMs: 5_000 });
    controller.abort(cancellation);

    await expect(request).rejects.toBe(cancellation);
  });

  it("clears the deadline after a successful response", async () => {
    vi.useFakeTimers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(JSON.stringify({ status: "fresh" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        })),
    );

    await expect(getFreshness({ timeoutMs: 5_000 })).resolves.toEqual({ status: "fresh" });
    expect(vi.getTimerCount()).toBe(0);
  });
});
