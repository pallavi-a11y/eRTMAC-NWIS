import { useEffect, useState } from "react";
import {
  Brain,
  Send,
  FileText,
  Sparkles,
  User,
  Bot,
  MapPin,
  Database,
  BarChart3,
  Plus,
  MessageSquare,
  Trash2,
  X,
} from "lucide-react";

import api from "../api/client";

import "./DrillMind.css";

// A corpus id like "WCR-0016" encodes the real numeric document_id the
// backend's /api/chat needs for its document_ids filter - parsed the exact
// same way Corpus.jsx does (and the backend itself does, in
// routers/corpus.py's _resolve_document_id): split on "-", take the last
// part, int().
const getRawDocumentId = (corpusId) => parseInt(corpusId.split("-").pop(), 10);

// All chat history lives in the browser's localStorage, not the backend -
// it's per-device/per-browser only (won't follow you to a different machine,
// and disappears if that browser's site data gets cleared), which is fine
// for a single-user local tool. Two keys: the full list of saved
// conversations, and which one was open last time so reopening this page
// returns to it instead of always starting fresh.
const CONVERSATIONS_KEY = "drillmind_conversations";
const ACTIVE_ID_KEY = "drillmind_active_conversation_id";
// The not-yet-sent text sitting in the input box. Without this, navigating
// to another page unmounts DrillMind entirely (plain React Router routing,
// no keep-alive), which reset the "message" useState back to "" - so
// half-typed questions were silently lost just by clicking Dashboard and
// coming back. Persisted the same way conversations already are.
const DRAFT_KEY = "drillmind_draft_message";
// Where chat history lived before conversations existed (a single running
// log) - read once below to migrate anything already saved there into a
// real conversation, so switching to this model doesn't erase history.
const LEGACY_HISTORY_KEY = "drillmind_chat_history";

const WELCOME_MESSAGE = {
  id: 1,
  sender: "bot",
  text:
    "Hello! I'm Drill Mind. I can help you analyse wells, drilling reports and nearby well intelligence from the documents that have been ingested.",
};

function makeConversation(messages = [WELCOME_MESSAGE]) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    title: "New conversation",
    messages,
    updatedAt: Date.now(),
  };
}

// A conversation is titled after its first real question, the same way
// ChatGPT/Claude name a chat off what you actually asked - truncated so a
// long question doesn't blow out the sidebar's width.
function deriveTitle(messages) {
  const firstUser = messages.find((m) => m.sender === "user");
  if (!firstUser || !firstUser.text) return "New conversation";
  const text = firstUser.text.trim();
  return text.length > 42 ? `${text.slice(0, 42)}...` : text;
}

// Reads saved conversations back out of localStorage, migrating the old
// single-log format into one real conversation the first time this runs
// after the upgrade. Wrapped in try/catch because localStorage can throw
// (private browsing, site data blocked, storage full) - every failure case
// here just falls back to a single fresh conversation instead of crashing.
function loadInitialState() {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) {
        const savedActiveId = localStorage.getItem(ACTIVE_ID_KEY);
        const activeId = parsed.some((c) => c.id === savedActiveId)
          ? savedActiveId
          : parsed[0].id;
        return { conversations: parsed, activeId };
      }
    }

    const legacyRaw = localStorage.getItem(LEGACY_HISTORY_KEY);
    if (legacyRaw) {
      const legacyMessages = JSON.parse(legacyRaw);
      if (Array.isArray(legacyMessages) && legacyMessages.length > 0) {
        const conversation = {
          ...makeConversation(legacyMessages),
          title: deriveTitle(legacyMessages),
        };
        localStorage.removeItem(LEGACY_HISTORY_KEY);
        return { conversations: [conversation], activeId: conversation.id };
      }
    }
  } catch {
    // fall through to a fresh conversation below
  }

  const fresh = makeConversation();
  return { conversations: [fresh], activeId: fresh.id };
}

// Reads back whatever draft text was sitting in the input box last time this
// page was open. Wrapped in try/catch for the same reasons as
// loadInitialState above (private browsing, blocked storage, etc.).
function loadDraftMessage() {
  try {
    return localStorage.getItem(DRAFT_KEY) || "";
  } catch {
    return "";
  }
}

function DrillMind() {
  const [message, setMessage] = useState(loadDraftMessage);
  // Which conversation currently has a reply in flight (null = none) -
  // NOT a plain boolean. A plain "sending" flag shared across every
  // conversation caused a real bug found by testing: starting a New Chat
  // while the previous conversation's reply was still loading silently
  // dropped the new message (the old flag was still true, so the guard
  // below returned early with zero feedback - no error, no cleared input,
  // nothing). Tracking the specific conversation id fixes both halves of
  // that: the reply always lands back in the conversation that actually
  // asked, even if you've switched away by the time it arrives, and the
  // "Thinking..." bubble only shows in that same conversation instead of
  // bleeding into whichever one you're currently looking at.
  const [sendingForId, setSendingForId] = useState(null);

  // Computed once on mount (the function form of useState only ever runs
  // once) so localStorage is read/migrated a single time, not on every render.
  const [initial] = useState(loadInitialState);
  const [conversations, setConversations] = useState(initial.conversations);
  const [activeId, setActiveId] = useState(initial.activeId);

  // "+" file picker - lets the user scope the next question to specific
  // documents instead of the whole corpus (POST /api/chat's document_ids).
  // corpusDocuments is fetched lazily, only the first time the picker opens,
  // not on page load - most visits to Drill Mind never open it.
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [showFilePicker, setShowFilePicker] = useState(false);
  const [corpusDocuments, setCorpusDocuments] = useState(null);
  const [corpusLoadError, setCorpusLoadError] = useState(null);

  const activeConversation =
    conversations.find((c) => c.id === activeId) || conversations[0];
  const messages = activeConversation.messages;
  // Only true when THIS conversation is the one waiting on a reply -
  // switching to a different conversation while one loads shows that
  // conversation's own real state, not a borrowed "Thinking..." indicator.
  const sending = sendingForId === activeConversation.id;

  // Persists every change (new message, new conversation, deletion, switch)
  // straight back to localStorage.
  useEffect(() => {
    try {
      localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
      localStorage.setItem(ACTIVE_ID_KEY, activeId);
    } catch {
      // Storage full/blocked - the chat still works for this session, it
      // just won't persist. Not worth interrupting the user over.
    }
  }, [conversations, activeId]);

  // Persists the current draft on every keystroke, so it survives navigating
  // to another page and back. Cleared automatically once the draft becomes
  // "" (message sent, New Chat, or switching conversations all already
  // call setMessage("")), so a stale draft never resurfaces in the wrong
  // conversation.
  useEffect(() => {
    try {
      localStorage.setItem(DRAFT_KEY, message);
    } catch {
      // Storage full/blocked - same as above, not worth interrupting over.
    }
  }, [message]);

  // Applies an update function to one specific conversation's messages (by
  // id, not "whatever's active right now" - see sendMessage below for why
  // that distinction matters), and refreshes its title (once real content
  // exists) and updatedAt (so the sidebar list re-sorts to show the most
  // recently used chat first).
  const updateConversationMessages = (conversationId, updateFn) => {
    setConversations((prev) =>
      prev.map((conv) => {
        if (conv.id !== conversationId) return conv;
        const nextMessages = updateFn(conv.messages);
        return {
          ...conv,
          messages: nextMessages,
          title: deriveTitle(nextMessages),
          updatedAt: Date.now(),
        };
      })
    );
  };

  /* =====================================================
     SEND MESSAGE
  ===================================================== */

  const sendMessage = async () => {
    const text = message.trim();
    // Blocked while ANY conversation has a reply pending, not just this
    // one - the backend serializes VLM calls one at a time regardless
    // (see utils.py's _VLM_LOCK), so a second concurrent send would just
    // queue behind the first anyway. Disabling here gives honest feedback
    // instead of a click that silently does nothing.
    if (!text || sendingForId) return;

    // Captured now, not read again later - if the user switches to a
    // different conversation before this reply comes back, it must still
    // land in the conversation that actually asked the question.
    const conversationId = activeId;
    // Also captured now, and carried on the message itself (not just used
    // for the request) - so scrolling back through history later still
    // shows which files a given question was scoped to.
    const filesForThisMessage = attachedFiles;

    const userMessage = {
      id: Date.now(),
      sender: "user",
      text,
      files: filesForThisMessage,
    };

    updateConversationMessages(conversationId, (prev) => [...prev, userMessage]);
    setMessage("");
    setAttachedFiles([]);
    setSendingForId(conversationId);

    try {
      const res = await api.post("/api/chat", {
        message: text,
        document_ids: filesForThisMessage.map((f) => f.documentId),
      });

      updateConversationMessages(conversationId, (prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: "bot",
          text: res.data.text,
          citations: res.data.citations || [],
        },
      ]);
    } catch {
      updateConversationMessages(conversationId, (prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: "bot",
          text: "Drill Mind is unreachable right now. Is the backend running?",
          citations: [],
        },
      ]);
    } finally {
      setSendingForId(null);
    }
  };

  /* =====================================================
     FILE PICKER - scope the next question to specific documents
  ===================================================== */

  const toggleFilePicker = () => {
    const opening = !showFilePicker;
    setShowFilePicker(opening);

    if (opening && corpusDocuments === null) {
      api
        .get("/api/corpus")
        .then((res) => setCorpusDocuments(res.data))
        .catch(() =>
          setCorpusLoadError("Could not load documents. Is the backend running?")
        );
    }
  };

  const toggleAttachedFile = (doc) => {
    setAttachedFiles((prev) => {
      const already = prev.some((f) => f.corpusId === doc.id);
      if (already) return prev.filter((f) => f.corpusId !== doc.id);
      return [...prev, { corpusId: doc.id, documentId: getRawDocumentId(doc.id), name: doc.name }];
    });
  };

  const removeAttachedFile = (corpusId) => {
    setAttachedFiles((prev) => prev.filter((f) => f.corpusId !== corpusId));
  };

  /* =====================================================
     CONVERSATIONS - new / switch / delete
  ===================================================== */

  const startNewChat = () => {
    const fresh = makeConversation();
    setConversations((prev) => [fresh, ...prev]);
    setActiveId(fresh.id);
    setMessage("");
  };

  const switchConversation = (id) => {
    if (id === activeId) return;
    setActiveId(id);
    setMessage("");
  };

  const deleteConversation = (id) => {
    if (!window.confirm("Delete this conversation? This cannot be undone.")) {
      return;
    }

    setConversations((prev) => {
      const remaining = prev.filter((c) => c.id !== id);

      if (id === activeId) {
        if (remaining.length > 0) {
          setActiveId(remaining[0].id);
        } else {
          const fresh = makeConversation();
          setActiveId(fresh.id);
          return [fresh];
        }
      }

      return remaining;
    });
  };

  /* =====================================================
     ENTER KEY
  ===================================================== */

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  /* =====================================================
     SUGGESTED QUESTIONS
  ===================================================== */

  const askQuestion = (question) => {
    setMessage(question);
  };

  // Most recently active conversation first, matching how every chat app's
  // history sidebar orders itself.
  const sortedConversations = [...conversations].sort(
    (a, b) => b.updatedAt - a.updatedAt
  );

  return (
    <div className="drill-page">

      <div className="drill-layout">

        {/* =================================================
            SIDEBAR - past conversations
        ================================================= */}

        <aside className="drill-sidebar">

          <button className="new-chat-btn" onClick={startNewChat}>
            <Plus size={16} />
            New Chat
          </button>

          <div className="conversation-list">

            {sortedConversations.map((conv) => (

              <div
                key={conv.id}
                className={`conversation-item ${
                  conv.id === activeId ? "active" : ""
                }`}
                onClick={() => switchConversation(conv.id)}
              >

                <MessageSquare size={14} />

                <span className="conversation-title">
                  {conv.title}
                </span>

                <button
                  className="delete-conversation-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(conv.id);
                  }}
                  title="Delete conversation"
                >
                  <Trash2 size={13} />
                </button>

              </div>

            ))}

          </div>

        </aside>


        {/* =================================================
            MAIN CHAT AREA
        ================================================= */}

        <div className="drill-main">

          {/* =================================================
              HEADER
          ================================================= */}

          <div className="drill-header">

            <div className="drill-title-section">

              <div className="drill-logo">
                <Brain size={26} />
              </div>

              <div>
                <span className="drill-label">
                  AI INTELLIGENCE
                </span>

                <h1>Drill Mind</h1>

                <p>
                  Your intelligent assistant for well and
                  drilling intelligence.
                </p>
              </div>

            </div>

            <div className="drill-status">
              <span></span>
              Intelligence Engine Ready
            </div>

          </div>


          {/* =================================================
              CHAT AREA
          ================================================= */}

          <div className="drill-chat-container">

            <div className="chat-messages">

              {messages.map((item) => (

                <div
                  key={item.id}
                  className={`chat-message ${
                    item.sender === "user"
                      ? "user-message"
                      : "bot-message"
                  }`}
                >

                  <div className="message-avatar">

                    {item.sender === "bot" ? (
                      <Bot size={17} />
                    ) : (
                      <User size={17} />
                    )}

                  </div>

                  <div className="message-content">

                    <span className="message-name">
                      {item.sender === "bot"
                        ? "Drill Mind"
                        : "You"}
                    </span>

                    {item.text && (
                      <div className="message-bubble">
                        {item.text}
                      </div>
                    )}

                    {item.files && item.files.length > 0 && (
                      <div className="citation-list">
                        <span className="citation-label">Scoped to</span>
                        {item.files.map((f) => (
                          <span className="citation-chip" key={f.corpusId}>
                            <FileText size={11} />
                            {f.name}
                          </span>
                        ))}
                      </div>
                    )}

                    {item.citations && item.citations.length > 0 && (
                      <div className="citation-list">
                        <span className="citation-label">Pages searched</span>
                        {item.citations.map((c, i) => (
                          <span className="citation-chip" key={i}>
                            <FileText size={11} />
                            {c.well_name} · Doc {c.document_id} · p.{c.page_num}
                          </span>
                        ))}
                      </div>
                    )}

                  </div>

                </div>

              ))}

              {sending && (

                <div className="chat-message bot-message">

                  <div className="message-avatar">
                    <Bot size={17} />
                  </div>

                  <div className="message-content">
                    <span className="message-name">Drill Mind</span>
                    <div className="message-bubble">Thinking...</div>
                  </div>

                </div>

              )}

            </div>


            {/* =================================================
                WELCOME / SUGGESTIONS
            ================================================= */}

            {messages.length === 1 && (

              <div className="drill-welcome">

                <div className="welcome-icon">
                  <Sparkles size={23} />
                </div>

                <h2>
                  What would you like to know?
                </h2>

                <p>
                  Ask Drill Mind about wells, drilling
                  operations or ingested reports.
                </p>


                <div className="suggestion-grid">

                  <button
                    onClick={() =>
                      askQuestion(
                        "Show me nearby wells with similar drilling depth."
                      )
                    }
                  >

                    <MapPin size={17} />

                    <div>
                      <strong>
                        Nearby Wells
                      </strong>

                      <span>
                        Find wells around a location
                      </span>
                    </div>

                  </button>


                  <button
                    onClick={() =>
                      askQuestion(
                        "Summarise the hazards found in the Ankleshwar field."
                      )
                    }
                  >

                    <FileText size={17} />

                    <div>
                      <strong>
                        Drilling Reports
                      </strong>

                      <span>
                        Analyse WCR and DDR reports
                      </span>
                    </div>

                  </button>


                  <button
                    onClick={() =>
                      askQuestion(
                        "What hazards have been recorded at similar drilling depths?"
                      )
                    }
                  >

                    <BarChart3 size={17} />

                    <div>
                      <strong>
                        Risk Analysis
                      </strong>

                      <span>
                        Analyse drilling hazards
                      </span>
                    </div>

                  </button>


                  <button
                    onClick={() =>
                      askQuestion(
                        "Compare nearby wells and their drilling data."
                      )
                    }
                  >

                    <Database size={17} />

                    <div>
                      <strong>
                        Well Comparison
                      </strong>

                      <span>
                        Compare nearby well intelligence
                      </span>
                    </div>

                  </button>

                </div>

              </div>

            )}

          </div>


          {/* =================================================
              CHAT INPUT
          ================================================= */}

          <div className="drill-input-area">

            {attachedFiles.length > 0 && (

              <div className="attached-file-row">
                {attachedFiles.map((f) => (
                  <span className="attached-file-chip" key={f.corpusId}>
                    <FileText size={11} />
                    {f.name}
                    <button
                      type="button"
                      onClick={() => removeAttachedFile(f.corpusId)}
                      title="Remove"
                    >
                      <X size={11} />
                    </button>
                  </span>
                ))}
              </div>

            )}

            <div className="chat-input-wrapper">

              <div className="file-picker-wrapper">

                <button
                  type="button"
                  className="attach-file-btn"
                  onClick={toggleFilePicker}
                  title="Scope this question to specific files"
                >
                  <Plus size={18} />
                </button>

                {showFilePicker && (

                  <div className="file-picker-popover">

                    <div className="file-picker-header">
                      <span>Scope to files</span>
                      <button type="button" onClick={() => setShowFilePicker(false)}>
                        <X size={13} />
                      </button>
                    </div>

                    {corpusDocuments === null ? (
                      <p className="file-picker-empty">Loading documents...</p>
                    ) : corpusLoadError ? (
                      <p className="file-picker-empty">{corpusLoadError}</p>
                    ) : corpusDocuments.length === 0 ? (
                      <p className="file-picker-empty">No documents in the corpus yet.</p>
                    ) : (
                      <div className="file-picker-list">
                        {corpusDocuments.map((doc) => (
                          <label className="file-picker-item" key={doc.id}>
                            <input
                              type="checkbox"
                              checked={attachedFiles.some((f) => f.corpusId === doc.id)}
                              onChange={() => toggleAttachedFile(doc)}
                            />
                            <div>
                              <strong>{doc.name}</strong>
                              <span>{doc.well} · {doc.id}</span>
                            </div>
                          </label>
                        ))}
                      </div>
                    )}

                  </div>

                )}

              </div>

              <textarea
                value={message}
                onChange={(e) =>
                  setMessage(e.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder="Ask Drill Mind..."
                rows="1"
              />

              <button
                className="send-btn"
                onClick={sendMessage}
                disabled={!message.trim() || Boolean(sendingForId)}
                title="Send message"
              >
                <Send size={18} />
              </button>

            </div>


            <p className="input-disclaimer">
              {attachedFiles.length > 0
                ? `Answering only from ${attachedFiles.length} selected file${attachedFiles.length === 1 ? "" : "s"}.`
                : "Drill Mind answers from documents already ingested into the corpus."}
            </p>

          </div>

        </div>

      </div>

    </div>
  );
}

export default DrillMind;
