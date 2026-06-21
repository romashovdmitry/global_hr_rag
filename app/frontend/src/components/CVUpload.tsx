import React, { useRef, useState } from "react";
import { uploadCV, clearCV } from "../api/client";

interface CVUploadProps {
  onUploaded: (charCount: number) => void;
  onCleared: () => void;
  cvLoaded: boolean;
}

export function CVUpload({ onUploaded, onCleared, cvLoaded }: CVUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setStatus("Uploading…");
    try {
      const result = await uploadCV(file);
      setStatus(`CV loaded (${result.text_length} chars)`);
      onUploaded(result.text_length);
    } catch (err: unknown) {
      setStatus(`Upload failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  async function handleClear() {
    setLoading(true);
    try {
      await clearCV();
      setStatus("");
      onCleared();
    } catch (err: unknown) {
      setStatus(`Failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="cv-upload">
      {cvLoaded ? (
        <button
          className="btn btn--secondary"
          disabled={loading}
          onClick={handleClear}
        >
          {loading ? "Removing…" : "Remove CV"}
        </button>
      ) : (
        <button
          className="btn btn--secondary"
          disabled={loading}
          onClick={() => inputRef.current?.click()}
        >
          {loading ? "Processing…" : "Attach CV (PDF)"}
        </button>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="application/pdf"
        style={{ display: "none" }}
        onChange={handleFile}
      />
      {status && <span className="cv-upload__status">{status}</span>}
    </div>
  );
}
