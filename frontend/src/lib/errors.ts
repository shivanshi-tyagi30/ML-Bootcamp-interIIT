import type { ErrorCode, JobError } from "./types";

// Plan section 5. The backend's user_message wins; these are the fallbacks
// and the messages for checks we also run in the browser before uploading.
export const ERROR_MESSAGES: Record<ErrorCode, string> = {
  E_UNSUPPORTED_FORMAT:
    "This file type isn't supported. Upload MP3, WAV, M4A, OGG, FLAC or WEBM.",
  E_EMPTY_FILE: "The file is empty.",
  E_TOO_LARGE: "The file is too large to process.",
  E_UNREADABLE: "The file is corrupt or has no audio track.",
  E_NO_SPEECH: "No speech was detected in this recording.",
  E_STT_FAILED: "Transcription failed. Please try again.",
  E_LM1_FAILED: "Transcript refinement failed. The raw transcript is still available.",
  E_LM2_FAILED: "Writing the meeting record failed. Both transcripts are still available.",
  E_INTERNAL: "Unexpected error; see logs.",
  E_NETWORK: "Couldn't reach the server. Check that the backend is running.",
};

export const ACCEPTED_EXTENSIONS = ["mp3", "wav", "m4a", "ogg", "flac", "webm"];
export const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB ?? 500);

export function makeError(code: ErrorCode, message?: string): JobError {
  return { code, user_message: message ?? ERROR_MESSAGES[code] };
}

/** Same checks the backend runs first, so obvious mistakes fail instantly. */
export function precheckFile(file: File): JobError | null {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ACCEPTED_EXTENSIONS.includes(ext)) return makeError("E_UNSUPPORTED_FORMAT");
  if (file.size === 0) return makeError("E_EMPTY_FILE");
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024)
    return makeError("E_TOO_LARGE", `The file is too large to process (limit ${MAX_UPLOAD_MB} MB).`);
  return null;
}
