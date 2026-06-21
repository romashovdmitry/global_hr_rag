import { useCallback, useEffect, useRef, useState } from "react";

export type WsEvent =
  | { type: "status"; text: string }
  | { type: "answer"; text: string }
  | { type: "error"; text: string }
  | { type: "done" };

interface UseWebSocketReturn {
  isConnected: boolean;
  isProcessing: boolean;
  statusText: string;
  sendMessage: (content: string) => void;
  lastEvent: WsEvent | null;
}

const WS_URL = `ws://${window.location.hostname}:8000/api/v1/chat/ws`;
const RECONNECT_DELAY_MS = 2000;

export function useWebSocket(
  onAnswer: (text: string) => void
): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null);
  // Keep the latest onAnswer in a ref so the stable onmessage handler always
  // calls the current version without needing to reconnect.
  const onAnswerRef = useRef(onAnswer);
  useEffect(() => { onAnswerRef.current = onAnswer; }, [onAnswer]);

  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [lastEvent, setLastEvent] = useState<WsEvent | null>(null);

  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Epoch increments on every cleanup so stale onclose callbacks from the
  // previous mount never trigger a reconnect in the new mount's context.
  const epochRef = useRef(0);

  useEffect(() => {
    const epoch = ++epochRef.current;

    function connect() {
      if (epochRef.current !== epoch) return;

      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (epochRef.current !== epoch) return;
        setIsConnected(true);
      };

      ws.onclose = () => {
        if (epochRef.current !== epoch) return;
        setIsConnected(false);
        setIsProcessing(false);
        // Auto-reconnect after a short delay.
        reconnectTimer.current = setTimeout(() => {
          if (epochRef.current === epoch) connect();
        }, RECONNECT_DELAY_MS);
      };

      ws.onerror = () => {
        if (epochRef.current !== epoch) return;
        setIsConnected(false);
      };

      ws.onmessage = (event) => {
        if (epochRef.current !== epoch) return;
        const evt: WsEvent = JSON.parse(event.data);
        setLastEvent(evt);

        if (evt.type === "status") {
          // Restore isProcessing in case the WS reconnected mid-processing
          // (onclose resets it to false, so status events must re-enable it).
          setIsProcessing(true);
          setStatusText(evt.text);
        } else if (evt.type === "answer") {
          onAnswerRef.current(evt.text);
          // Clear status after the answer is delivered.
          setStatusText("");
          setIsProcessing(false);
        } else if (evt.type === "error") {
          setStatusText(`Error: ${evt.text}`);
          setIsProcessing(false);
        } else if (evt.type === "done") {
          setIsProcessing(false);
          // Keep the last status text visible briefly so users can read it,
          // then clear it.
          setTimeout(() => setStatusText(""), 1500);
        }
      };
    }

    connect();

    return () => {
      // Invalidate this epoch — all stale onclose/onmessage callbacks will
      // see epochRef.current !== epoch and become no-ops.
      epochRef.current++;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, []);

  const sendMessage = useCallback(
    (content: string) => {
      const ws = wsRef.current;
      if (ws?.readyState === WebSocket.OPEN) {
        // Mark processing immediately so the spinner appears in this render
        // cycle, before any WebSocket reply arrives.
        setIsProcessing(true);
        setStatusText("Sending…");
        ws.send(JSON.stringify({ content }));
      } else {
        // WebSocket not ready yet — surface the problem to the user instead
        // of silently dropping the message.
        setStatusText("Connection lost. Reconnecting…");
      }
    },
    []
  );

  return { isConnected, isProcessing, statusText, sendMessage, lastEvent };
}
