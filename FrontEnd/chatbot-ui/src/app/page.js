"use client";

import { useEffect, useRef, useState } from "react";

// Animated text component for bot responses
function AnimatedText({ text, animate }) {
  const [displayed, setDisplayed] = useState(animate ? "" : text);

  useEffect(() => {
    if (!animate) {
      setDisplayed(text);
      return;
    }
    setDisplayed("");
    let i = 0;
    const interval = setInterval(() => {
      setDisplayed(text.slice(0, i + 1));
      i++;
      if (i >= text.length) clearInterval(interval);
    }, 18);
    return () => clearInterval(interval);
  }, [text, animate]);

  return <span>{displayed}</span>;
}

const FLASK = "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const [chatList, setChatList] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [currentTitle, setCurrentTitle] = useState("");

  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [animatedMessageIndex, setAnimatedMessageIndex] = useState(-1);
  const [showUploadPopup, setShowUploadPopup] = useState(false);
  const [pendingFiles, setPendingFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  /* Scroll to bottom on new messages */
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  /* Load chats + uploaded files on mount */
  useEffect(() => {
    fetchChats();
    fetchUploadedFiles();
  }, []);

  const fetchChats = async () => {
    const res = await fetch(`${FLASK}/chats`);
    const data = await res.json();
    setChatList(data.reverse());
  };

  /* Fetch already-uploaded files from server */
  const fetchUploadedFiles = async () => {
    const res = await fetch(`${FLASK}/upload`);
    const data = await res.json();
    if (data.files) {
      const restored = data.files.map((name) => ({
        file: { name },
        enabled: true,
      }));
      setUploadedFiles(restored);
    }
  };

  /* Save / Update Chat */
  const saveChat = async (chatData) => {
    await fetch(`${FLASK}/chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(chatData),
    });
    fetchChats();
  };

  /* Send */
  const handleSend = async () => {
    if (!input.trim()) return;

    let chatId = currentChatId;
    let title = currentTitle;

    if (!chatId) {
      chatId = Date.now();
      title = input.slice(0, 25);
      setCurrentChatId(chatId);
      setCurrentTitle(title);
    }

    const userMsg = { role: "user", text: input };

    // Get reply from Flask
    const enabledDocs = uploadedFiles
      .filter((f) => f.enabled)
      .map((f) => f.file.name);

    const res = await fetch(`${FLASK}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input, docs: enabledDocs }),
    });
    const data = await res.json();
    const botMsg = { role: "bot", text: data.reply };
    const updatedMessages = [...messages, userMsg, botMsg];

    setMessages(updatedMessages);
    setAnimatedMessageIndex(updatedMessages.length - 1);

    const chatData = {
      id: chatId,
      title,
      messages: updatedMessages,
      updatedAt: new Date(),
    };

    await saveChat(chatData);
    setInput("");
  };

  /* New Chat */
  const handleNewChat = () => {
    setMessages([]);
    setCurrentChatId(null);
    setCurrentTitle("");
  };

  /* Load Chat */
  const loadChat = (chat) => {
    setMessages(chat.messages);
    setCurrentChatId(chat.id);
    setCurrentTitle(chat.title);
    setAnimatedMessageIndex(-1);
  };

  /* File Upload */
  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files);
    setPendingFiles((prev) => [...prev, ...selected]);
    if (!showUploadPopup) setShowUploadPopup(true);
    e.target.value = "";
  };

  const handleConfirmUpload = async () => {
    setUploading(true);
    const formData = new FormData();
    pendingFiles.forEach((file) => formData.append("files", file));

    await fetch(`${FLASK}/upload`, { method: "POST", body: formData });

    const confirmed = pendingFiles.map((file) => ({ file, enabled: true }));
    setUploadedFiles((prev) => [...prev, ...confirmed]);
    setPendingFiles([]);
    setShowUploadPopup(false);
    setUploading(false);
  };

  const handleCancelUpload = () => {
    setPendingFiles([]);
    setShowUploadPopup(false);
  };

  const toggleFile = (index) => {
    setUploadedFiles((prev) =>
      prev.map((f, i) => (i === index ? { ...f, enabled: !f.enabled } : f))
    );
  };

  return (
    <main className="h-screen w-full flex" style={{ fontFamily: "system-ui, sans-serif" }}>

      <style>{`
        .sidebar-scroll::-webkit-scrollbar { width: 4px; }
        .sidebar-scroll::-webkit-scrollbar-track { background: transparent; }
        .sidebar-scroll::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
        .sidebar-scroll::-webkit-scrollbar-thumb:hover { background: #555; }

        .chat-scroll::-webkit-scrollbar { width: 6px; }
        .chat-scroll::-webkit-scrollbar-track { background: transparent; }
        .chat-scroll::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #3a3a3a, #2a2a2a); border-radius: 99px; }
        .chat-scroll::-webkit-scrollbar-thumb:hover { background: #10a37f; transition: background 0.2s; }
      `}</style>

      {/* LEFT SIDEBAR */}
      <div style={{ width: "260px", background: "#171717", color: "#ececec", display: "flex", flexDirection: "column", flexShrink: 0 }}>

        {/* Header */}
        <div style={{ padding: "16px", borderBottom: "1px solid #2a2a2a", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2 style={{ fontWeight: 600, fontSize: "15px", margin: 0 }}>Chats</h2>
          <button
            onClick={handleNewChat}
            style={{ background: "#2f2f2f", border: "1px solid #3a3a3a", color: "#ececec", padding: "5px 10px", borderRadius: "6px", fontSize: "13px", cursor: "pointer" }}
          >
            + New
          </button>
        </div>

        {/* Docs Upload */}
        <div style={{ padding: "14px 16px", borderBottom: "1px solid #2a2a2a" }}>
          <p style={{ fontSize: "12px", color: "#888", margin: "0 0 10px 0" }}>📄 Documents</p>

          <button
            onClick={() => fileInputRef.current.click()}
            style={{ background: "#2f2f2f", border: "1px solid #3a3a3a", color: "#ececec", padding: "6px 12px", borderRadius: "6px", fontSize: "13px", cursor: "pointer", width: "100%" }}
          >
            Upload File
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            style={{ display: "none" }}
            onChange={handleFileChange}
          />

          {uploadedFiles.length > 0 && (
            <div className="sidebar-scroll" style={{ marginTop: "10px", display: "flex", flexDirection: "column", gap: "6px", maxHeight: "168px", overflowY: "auto" }}>
              {uploadedFiles.map((item, index) => (
                <div
                  key={index}
                  style={{ display: "flex", alignItems: "center", gap: "8px", padding: "6px 8px", background: "#2a2a2a", borderRadius: "6px", cursor: "pointer" }}
                  onClick={() => toggleFile(index)}
                >
                  <div style={{
                    width: "16px", height: "16px", borderRadius: "4px", border: "1.5px solid #555",
                    background: item.enabled ? "#10a37f" : "transparent",
                    display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0
                  }}>
                    {item.enabled && <span style={{ color: "#fff", fontSize: "11px", lineHeight: 1 }}>✓</span>}
                  </div>
                  <span style={{ fontSize: "12px", color: "#ccc", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {item.file.name}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Chat History */}
        <div style={{ padding: "12px 16px 4px 16px", borderBottom: "1px solid #2a2a2a" }}>
          <p style={{ fontSize: "12px", color: "#888", margin: 0 }}>🕘 Chat History</p>
        </div>
        <div className="sidebar-scroll" style={{ flex: 1, overflowY: "auto", padding: "10px 8px" }}>
          {chatList.map((chat) => (
            <div
              key={chat.id}
              onClick={() => loadChat(chat)}
              style={{
                padding: "9px 12px",
                borderRadius: "6px",
                cursor: "pointer",
                fontSize: "13px",
                marginBottom: "2px",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                background: chat.id === currentChatId ? "#2f2f2f" : "transparent",
                color: chat.id === currentChatId ? "#ececec" : "#aaa",
              }}
              onMouseEnter={(e) => { if (chat.id !== currentChatId) e.currentTarget.style.background = "#212121"; }}
              onMouseLeave={(e) => { if (chat.id !== currentChatId) e.currentTarget.style.background = "transparent"; }}
            >
              {chat.title}
            </div>
          ))}
        </div>
      </div>

      {/* RIGHT — Chat Area */}
      <div style={{ flex: 1, background: "#212121", display: "flex", flexDirection: "column", minWidth: 0 }}>

        {/* Messages */}
        <div className="chat-scroll" style={{ flex: 1, padding: "32px 20%", overflowY: "auto", minWidth: 0 }}>
          {messages.length === 0 ? (
            <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#555", fontSize: "18px" }}>
              How can I help you today?
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
              {messages.map((msg, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                    minWidth: 0,
                  }}
                >
                  <div style={{
                    maxWidth: msg.role === "user" ? "75%" : "100%",
                    width: msg.role === "bot" ? "100%" : "auto",
                    minWidth: 0,
                    padding: "12px 16px",
                    borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                    background: msg.role === "user" ? "#2f2f2f" : "transparent",
                    color: "#ececec",
                    fontSize: "14px",
                    lineHeight: "1.6",
                    wordBreak: "break-word",
                    overflowWrap: "anywhere",
                    whiteSpace: "pre-wrap",
                  }}>
                    {msg.role === "bot"
                      ? <AnimatedText key={i} text={msg.text} animate={i === animatedMessageIndex} />
                      : msg.text}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div style={{ padding: "16px 20%", borderTop: "1px solid #2a2a2a" }}>
          <div style={{ display: "flex", gap: "8px", background: "#2f2f2f", border: "1px solid #3a3a3a", borderRadius: "12px", padding: "10px 14px", alignItems: "center" }}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Message..."
              style={{ flex: 1, background: "transparent", border: "none", outline: "none", color: "#ececec", fontSize: "14px", minWidth: 0 }}
            />
            <button
              onClick={handleSend}
              style={{ background: "#10a37f", border: "none", color: "#fff", padding: "6px 14px", borderRadius: "8px", fontSize: "13px", cursor: "pointer", flexShrink: 0 }}
            >
              Send
            </button>
          </div>
        </div>

      </div>

      {/* Upload Popup */}
      {showUploadPopup && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)", display: "flex", alignItems: "center", justifyContent: "center", zIndex: 100 }}>
          <div style={{ background: "#1e1e1e", border: "1px solid #2a2a2a", borderRadius: "12px", padding: "24px", width: "400px", maxWidth: "90vw" }}>

            <h3 style={{ color: "#ececec", fontSize: "15px", fontWeight: 600, margin: "0 0 16px 0" }}>Upload Documents</h3>

            <div className="sidebar-scroll" style={{ display: "flex", flexDirection: "column", gap: "8px", maxHeight: "240px", overflowY: "auto", marginBottom: "20px" }}>
              {pendingFiles.length === 0 ? (
                <p style={{ color: "#555", fontSize: "13px" }}>No files selected.</p>
              ) : (
                pendingFiles.map((file, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: "10px", padding: "8px 10px", background: "#2a2a2a", borderRadius: "8px" }}>
                    <span style={{ fontSize: "16px" }}>📄</span>
                    <div style={{ flex: 1, overflow: "hidden" }}>
                      <p style={{ color: "#ececec", fontSize: "13px", margin: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{file.name}</p>
                      <p style={{ color: "#666", fontSize: "11px", margin: 0 }}>{(file.size / 1024).toFixed(1)} KB</p>
                    </div>
                  </div>
                ))
              )}
            </div>

            <div style={{ display: "flex", gap: "8px", marginBottom: "16px" }}>
              <button
                onClick={() => fileInputRef.current.click()}
                style={{ flex: 1, background: "#2f2f2f", border: "1px solid #3a3a3a", color: "#ccc", padding: "7px", borderRadius: "8px", fontSize: "13px", cursor: "pointer" }}
              >
                + Add More
              </button>
            </div>

            <div style={{ display: "flex", gap: "8px" }}>
              <button
                onClick={handleCancelUpload}
                style={{ flex: 1, background: "transparent", border: "1px solid #3a3a3a", color: "#aaa", padding: "8px", borderRadius: "8px", fontSize: "13px", cursor: "pointer" }}
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmUpload}
                disabled={uploading || pendingFiles.length === 0}
                style={{ flex: 1, background: pendingFiles.length === 0 ? "#1a3a32" : "#10a37f", border: "none", color: "#fff", padding: "8px", borderRadius: "8px", fontSize: "13px", cursor: pendingFiles.length === 0 ? "not-allowed" : "pointer", opacity: uploading ? 0.7 : 1 }}
              >
                {uploading ? "Uploading..." : "Confirm"}
              </button>
            </div>

          </div>
        </div>
      )}

    </main>
  );
}