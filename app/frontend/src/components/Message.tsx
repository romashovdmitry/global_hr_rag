import React from "react";
import type { MessageData } from "../api/client";

interface MessageProps {
  message: MessageData;
  unanswered?: boolean;
}

export function Message({ message, unanswered = false }: MessageProps) {
  const isUser = message.role === "user";
  return (
    <div className={`message message--${message.role}`}>
      <span className="message__label">{isUser ? "You" : "Assistant"}</span>
      <div className="message__content">{message.content}</div>
      {unanswered && (
        <div className="message__unanswered">
          <span className="message__unanswered-icon">⚠</span>
          No response received — try sending again
        </div>
      )}
    </div>
  );
}
