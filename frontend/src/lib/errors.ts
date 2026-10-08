import type { ErrorCode, JobError } from "./types";

// Plan section 5. The backend's user_message wins; these are the fallbacks
// and the messages for checks we also run in the browser before uploading.
export const ERROR_MESSAGES: Record<ErrorCode, string> = {
  E_UNSUPPORTED_FORMAT:
    "This file type isn't supported. Please upload an audio file (MP3, WAV, M4A, OGG, FLAC, WEBM, AAC).",
  E_EMPTY_FILE: "The file is empty. Please choose a recording that contains audio.",
  E_TOO_LARGE: "The file is too large.",
  E_TOO_LONG: "The recording is too long.",
  E_UNREADABLE: "The file could not be read. It may be corrupt or have no audio track.",
  E_NO_SPEECH: "No speech was detected in this recording.",
  E_STT_FAILED: "Transcription failed. Please try again.",
  E_LM1_FAILED: "Transcript refinement failed. The raw transcript is still available.",
  E_LM2_FAILED: "Writing the meeting record failed. Both transcripts are still available.",
  E_RENDER_FAILED: "The record was created but the downloads could not be generated.",
  E_NEEDS_KEY:
    "This website writes the minutes with Gemini. Paste your free Gemini API key (aistudio.google.com → Get API key) in the API key box, then upload again.",
  E_SERVER_SETUP: "This website isn't fully set up yet (a speech service key is missing on the server).",
  E_BUSY: "The server is busy with other uploads. Please try again in a moment.",
  E_NOT_FOUND: "This meeting could not be found.",
  E_JOB_RUNNING: "This meeting is still being processed. Try again when it has finished.",
  E_CANCELLED: "Processing was cancelled.",
  E_FFMPEG_MISSING:
    "ffmpeg is not installed on the server, so the audio can't be converted. Install ffmpeg and restart the backend.",
  E_INTERNAL: "Something unexpected went wrong. Please try again.",
  E_NETWORK: "Couldn't reach the server. Check that the backend is running.",
};

// Keep in step with the backend's ALLOWED_EXT and MAX_FILE_MB.
export const ACCEPTED_EXTENSIONS = ["mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4", "aac"];
export const MAX_UPLOAD_MB = Number(import.meta.env.VITE_MAX_UPLOAD_MB ?? 200);

export function makeError(code: ErrorCode, message?: string): JobError {
  return { code, user_message: message ?? ERROR_MESSAGES[code] };
}

/** Same checks the backend runs first, so obvious mistakes fail instantly. */
export function precheckFile(file: File): JobError | null {
  const ext = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (!ACCEPTED_EXTENSIONS.includes(ext)) return makeError("E_UNSUPPORTED_FORMAT");
  if (file.size === 0) return makeError("E_EMPTY_FILE");
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024)
    return makeError("E_TOO_LARGE", `The file is too large. The maximum size is ${MAX_UPLOAD_MB} MB.`);
  return null;
}
