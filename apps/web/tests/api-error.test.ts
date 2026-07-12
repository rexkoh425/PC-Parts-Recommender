import { describe, expect, it } from "vitest";
import { ApiError, apiErrorEvidence, apiErrorRequestId } from "../lib/api";

describe("API error evidence", () => {
  it("surfaces compatibility and infeasibility reasons from a failed replacement", () => {
    const error = new ApiError("Replacement failed", 409, {
      compatibility_checks: [
        {
          rule_id: "gpu_case_length_v1",
          status: "fail",
          message: "GPU length exceeds the case clearance by 18 mm.",
        },
      ],
      infeasibility: {
        reasons: [
          {
            code: "psu_connector_missing",
            message: "The retained PSU does not have the required GPU connector.",
          },
        ],
      },
    });

    expect(apiErrorEvidence(error)).toEqual([
      "The retained PSU does not have the required GPU connector.",
      "GPU length exceeds the case clearance by 18 mm.",
    ]);
  });

  it("does not invent evidence for an ordinary network error", () => {
    expect(apiErrorEvidence(new Error("offline"))).toEqual([]);
  });

  it("extracts checks and request context from the backend error envelope", () => {
    const error = new ApiError("Replacement rejected", 409, {
      message: "The replacement was rejected by one or more hard compatibility rules.",
      error: {
        code: "incompatible_replacement",
        message: "The replacement was rejected by one or more hard compatibility rules.",
        request_id: "replace-request-42",
        details: {
          checks: [
            {
              rule_id: "gpu_case_length_v1",
              status: "fail",
              message: "GPU length exceeds the case clearance by 18 mm.",
              affected_categories: ["gpu", "case"],
            },
            {
              rule_id: "gpu_psu_connector_v1",
              status: "unknown",
              message: "The required GPU power connector could not be verified.",
              affected_categories: ["gpu", "psu"],
            },
          ],
        },
      },
    });

    expect(apiErrorEvidence(error)).toEqual([
      "GPU length exceeds the case clearance by 18 mm.",
      "The required GPU power connector could not be verified.",
    ]);
    expect(apiErrorRequestId(error)).toBe("replace-request-42");
  });
});
