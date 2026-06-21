import React, { useEffect, useRef, useState } from "react";
import type { MessageData } from "../api/client";
import { getMessages } from "../api/client";
import { Message } from "./Message";
import { useWebSocket } from "../hooks/useWebSocket";

export function Chat() {
  const [messages, setMessages] = useState<MessageData[]>([]);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  const handleAnswer = (text: string) => {
    const msg: MessageData = {
      id: crypto.randomUUID(),
      session_id: "",
      role: "assistant",
      content: text,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, msg]);
  };

  const { isConnected, isProcessing, statusText, sendMessage } = useWebSocket(handleAnswer);

  useEffect(() => {
    getMessages().then(setMessages).catch(console.error);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, statusText]);

  function handleSend() {
    const content = input.trim();
    if (!content || isProcessing) return;

    const userMsg: MessageData = {
      id: crypto.randomUUID(),
      session_id: "",
      role: "user",
      content,
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    sendMessage(content);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="chat">
      <div className="chat__status-bar">
        <span className={`chat__dot chat__dot--${isConnected ? "online" : "offline"}`} />
        {isConnected ? "Connected" : "Connecting…"}
      </div>

      <div className="chat__messages">
        {messages.length === 0 && (
          <p className="chat__empty">
            Ask me about IT vacancies. Upload your CV for personalised matching.
          </p>
        )}
        {messages.map((m, idx) => {
          // A user message is "unanswered" when the next message is also from
          // the user (or it's the last message) and we're not currently
          // processing a response for it.
          const nextMsg = messages[idx + 1];
          const isLastMsg = idx === messages.length - 1;
          const unanswered =
            m.role === "user" &&
            !isProcessing &&
            (isLastMsg || nextMsg?.role === "user");
          return <Message key={m.id} message={m} unanswered={unanswered} />;
        })}
        {(isProcessing || statusText) && (
          <div className={`chat__thinking${statusText.startsWith("Error:") ? " chat__thinking--error" : ""}`}>
            {isProcessing && <span className="chat__spinner" />}
            {statusText || "Processing…"}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat__input-row">
        <textarea
          className="chat__textarea"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message… (Enter to send)"
          rows={2}
          disabled={isProcessing}
        />
        <button
          className="btn btn--primary"
          onClick={handleSend}
          disabled={isProcessing || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
