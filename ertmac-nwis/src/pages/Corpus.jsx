import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  Search,
  Filter,
  FileText,
  Drill,
  Database,
  Upload,
  Eye,
  Download,
  MoreVertical,
  CheckCircle2,
  Clock,
  AlertTriangle,
  FileSearch,
  X,
  ChevronLeft,
  ChevronRight,
  FileCheck2,
  Layers,
  Save,
} from "lucide-react";

import api, { API_BASE_URL } from "../api/client";

import "./Corpus.css";

/* A corpus id like "WCR-0010" encodes the real numeric document_id the
   backend needs for the review/OCR endpoints - has to be parsed exactly
   the way the backend itself does it (routers/corpus.py's
   _resolve_document_id: split on "-", take the last part, int()). */
const getRawDocumentId = (corpusId) =>
  parseInt(corpusId.split("-").pop(), 10);

function Corpus() {
  const navigate = useNavigate();

  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedType, setSelectedType] = useState("All");
  const [selectedStatus, setSelectedStatus] = useState("All");

  const [selectedDocument, setSelectedDocument] = useState(null);
  const [currentPage, setCurrentPage] = useState(1);

  const [pageIndex, setPageIndex] = useState([]);
  const [pageData, setPageData] = useState(null);
  const [pageLoading, setPageLoading] = useState(false);
  const [imageError, setImageError] = useState(false);

  // The OCR text is edited directly in the DOM (contentEditable), not through
  // React state - same approach as the standalone /review tool this reuses
  // the save endpoint from. A controlled value would fight the browser's own
  // cursor/selection handling on every keystroke; instead this ref is only
  // read from (on save) and only written to (seeding fresh content) below.
  const ocrEditableRef = useRef(null);
  const [savingCorrection, setSavingCorrection] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  /* =====================================================
     LOAD REAL DOCUMENTS
  ===================================================== */

  useEffect(() => {
    let cancelled = false;

    api
      .get("/api/corpus")
      .then((res) => {
        if (!cancelled) {
          setDocuments(res.data);
          setLoadError(null);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setLoadError(
            "Could not reach the backend. Is the server running?"
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  /* =====================================================
     LOAD PAGE INDEX WHEN A DOCUMENT IS OPENED
  ===================================================== */

  useEffect(() => {
    if (!selectedDocument) return;

    let cancelled = false;
    const rawId = getRawDocumentId(selectedDocument.id);

    api
      .get(`/api/review/${rawId}/pages`)
      .then((res) => {
        if (!cancelled) setPageIndex(res.data.pages || []);
      })
      .catch(() => {
        if (!cancelled) setPageIndex([]);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDocument]);

  /* =====================================================
     LOAD THE CURRENT PAGE'S TEXT/IMAGE DATA
  ===================================================== */

  useEffect(() => {
    if (!selectedDocument) return;

    let cancelled = false;
    // Resetting the per-request loading/error flags when the page changes,
    // before the fetch starts, is the standard data-fetching idiom - not the
    // "derive during render" case this rule otherwise guards against.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setPageLoading(true);
    setImageError(false);
    // Cleared here (on an actual page/document change), NOT in the seed
    // effect below - handleSaveCorrection's own refetch also lands in
    // pageData, and clearing it there raced with setSaveStatus({ok:true,...})
    // and immediately erased the "Saved." message before it was ever seen.
    setSaveStatus(null);

    const rawId = getRawDocumentId(selectedDocument.id);

    api
      .get(`/api/review/${rawId}/${currentPage}`)
      .then((res) => {
        if (!cancelled) setPageData(res.data);
      })
      .catch(() => {
        if (!cancelled) setPageData(null);
      })
      .finally(() => {
        if (!cancelled) setPageLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedDocument, currentPage]);

  /* =====================================================
     SEED THE EDITABLE OCR BOX
     Runs only when a fresh pageData object arrives (a new page/document was
     fetched) - never on a re-render caused by typing, so it doesn't stomp
     on in-progress edits. Real <table> markup, not markdown text, same
     reasoning as the standalone review tool: a plain string can never
     visually render as a bordered grid.
  ===================================================== */

  useEffect(() => {
    if (ocrEditableRef.current && pageData && pageData.rendered_html) {
      ocrEditableRef.current.innerHTML = pageData.rendered_html;
    }
  }, [pageData]);

  // Converts the editable box's current DOM (real <table> elements included)
  // back into a single plain-text string for saving - the inverse of the
  // server's render_page_as_html(). A <table> becomes markdown pipe rows
  // (matching table_parser.py's own format), so re-opening this saved text
  // still round-trips through the table detector correctly. Ported directly
  // from routers/review.py's standalone tool so both editors save in the
  // exact same format.
  const serializeEditableContent = (container) => {
    const lines = [];
    const walk = (node) => {
      if (node.nodeType === Node.TEXT_NODE) {
        if (node.textContent) lines.push(node.textContent);
        return;
      }
      if (node.nodeName === "TABLE") {
        const rows = Array.from(node.querySelectorAll("tr"));
        rows.forEach((tr, idx) => {
          const cells = Array.from(tr.children).map((cell) =>
            cell.textContent.trim()
          );
          lines.push("| " + cells.join(" | ") + " |");
          if (idx === 0) {
            lines.push("| " + cells.map(() => "---").join(" | ") + " |");
          }
        });
        lines.push("");
        return;
      }
      if (node.nodeName === "BR") {
        lines.push("");
        return;
      }
      if (typeof node.querySelector === "function" && node.querySelector("table")) {
        for (const child of node.childNodes) walk(child);
        return;
      }
      const text = node.textContent;
      if (text) lines.push(text);
    };
    for (const child of container.childNodes) walk(child);
    return lines.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  };

  const handleSaveCorrection = async () => {
    if (!ocrEditableRef.current || !selectedDocument) return;

    const text = serializeEditableContent(ocrEditableRef.current);
    const rawId = getRawDocumentId(selectedDocument.id);

    setSavingCorrection(true);
    setSaveStatus(null);

    try {
      await api.post(`/api/review/${rawId}/${currentPage}/correction`, { text });
      // Re-fetch so the "Human-corrected" badge and corrected_text flag
      // reflect what was actually saved, not an assumed success state.
      const res = await api.get(`/api/review/${rawId}/${currentPage}`);
      setPageData(res.data);
      setSaveStatus({ ok: true, message: "Saved." });
    } catch {
      setSaveStatus({ ok: false, message: "Save failed - is the backend running?" });
    } finally {
      setSavingCorrection(false);
    }
  };

  /* =====================================================
     FILTER DOCUMENTS
  ===================================================== */

  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      const query = searchQuery.toLowerCase().trim();

      const matchesSearch =
        !query ||
        doc.name.toLowerCase().includes(query) ||
        doc.id.toLowerCase().includes(query) ||
        doc.well.toLowerCase().includes(query) ||
        doc.region.toLowerCase().includes(query);

      const matchesType =
        selectedType === "All" ||
        doc.type === selectedType;

      const matchesStatus =
        selectedStatus === "All" ||
        doc.status === selectedStatus;

      return (
        matchesSearch &&
        matchesType &&
        matchesStatus
      );
    });
  }, [
    documents,
    searchQuery,
    selectedType,
    selectedStatus,
  ]);

  /* =====================================================
     SUMMARY COUNTS - derived from the real fetched documents
  ===================================================== */

  const wcrCount = documents.filter((doc) => doc.type === "WCR").length;
  const ddrCount = documents.filter((doc) => doc.type === "DDR").length;
  const reviewCount = documents.filter((doc) => doc.status === "Review").length;

  /* =====================================================
     TYPE ICON / CLASS
  ===================================================== */

  const getTypeIcon = (type) => {
    if (type === "WCR") {
      return <FileText size={17} />;
    }

    return <Drill size={17} />;
  };

  const getTypeClass = (type) => {
    if (type === "WCR") {
      return "wcr";
    }

    return "ddr";
  };

  /* =====================================================
     STATUS ICON
  ===================================================== */

  const getStatusIcon = (status) => {
    if (status === "Processed") {
      return <CheckCircle2 size={12} />;
    }

    if (status === "Processing") {
      return <Clock size={12} />;
    }

    return <AlertTriangle size={12} />;
  };

  /* =====================================================
     OPEN / CLOSE DOCUMENT
  ===================================================== */

  const handleOpenDocument = (doc) => {
    setSelectedDocument(doc);
    setCurrentPage(1);
  };

  const closeDocument = () => {
    setSelectedDocument(null);
    setCurrentPage(1);
    setPageData(null);
    setPageIndex([]);
  };

  /* =====================================================
     UPLOAD / DOWNLOAD
  ===================================================== */

  const handleUpload = () => {
    navigate("/upload");
  };

  const handleDownload = (doc) => {
    window.open(`${API_BASE_URL}/api/corpus/${doc.id}/file`, "_blank");
  };

  const handleDelete = async (doc) => {
    if (!window.confirm(`Delete "${doc.name}"? This cannot be undone.`)) {
      return;
    }

    try {
      await api.delete(`/api/corpus/${doc.id}`);
      setDocuments((prev) => prev.filter((item) => item.id !== doc.id));
      if (selectedDocument?.id === doc.id) {
        closeDocument();
      }
    } catch {
      alert("Could not delete this document. Is the backend running?");
    }
  };

  /* =====================================================
     PAGE STATUS BADGE - real, derived from what the backend
     actually knows about this page, not a fabricated number
  ===================================================== */

  const pageStatusBadge = () => {
    // Checked first, ahead of has_extraction: a hard-failed page still gets a
    // page_extractions row (see backend's extract_page_content), so
    // has_extraction alone can't tell "attempted and failed" apart from
    // "genuinely not reached yet" - parse_ok is the real signal. Found as a
    // real bug: without this check, a fully failed page (empty text) fell
    // through to the final "Extracted" branch below and showed a misleading
    // green badge while the panel body said "No extraction available".
    if (pageData && pageData.parse_ok === 0) {
      return { label: "Failed", tone: "bad" };
    }
    if (!pageData || !pageData.has_extraction) {
      if (selectedDocument?.status === "Processing") {
        return { label: "Processing", tone: "pending" };
      }
      return { label: "Not yet extracted", tone: "pending" };
    }
    if (pageData.corrected_text) {
      return { label: "Human-corrected", tone: "good" };
    }
    if (pageData.text_possibly_truncated) {
      return { label: "Possibly incomplete", tone: "warn" };
    }
    return { label: "Extracted", tone: "good" };
  };

  return (
    <div className="corpus-page">

      {/* =====================================================
          HEADER
      ===================================================== */}

      <div className="corpus-header">

        <div>

          <span className="corpus-eyebrow">
            eRTMAC-NWIS / DOCUMENT INTELLIGENCE
          </span>

          <h1>
            Document Corpus
          </h1>

          <p>
            Search, manage and analyse oil &amp; gas
            drilling and well intelligence documents.
          </p>

        </div>

        <button
          className="corpus-upload-btn"
          onClick={handleUpload}
        >
          <Upload size={16} />
          Upload Document
        </button>

      </div>


      {/* =====================================================
          SUMMARY
      ===================================================== */}

      <div className="corpus-summary-grid">

        <div className="corpus-summary-card blue">

          <div className="corpus-summary-icon">
            <Database size={21} />
          </div>

          <div>
            <span>Total Documents</span>
            <strong>{documents.length}</strong>
            <small>Across the corpus</small>
          </div>

        </div>


        <div className="corpus-summary-card green">

          <div className="corpus-summary-icon">
            <FileText size={21} />
          </div>

          <div>
            <span>WCR Reports</span>
            <strong>{wcrCount}</strong>
            <small>Well completion records</small>
          </div>

        </div>


        <div className="corpus-summary-card purple">

          <div className="corpus-summary-icon">
            <Drill size={21} />
          </div>

          <div>
            <span>DDR Logs</span>
            <strong>{ddrCount}</strong>
            <small>Daily drilling records</small>
          </div>

        </div>


        <div className="corpus-summary-card orange">

          <div className="corpus-summary-icon">
            <FileSearch size={21} />
          </div>

          <div>
            <span>Needs Review</span>
            <strong>{reviewCount}</strong>
            <small>Flagged for manual check</small>
          </div>

        </div>

      </div>


      {/* =====================================================
          DOCUMENT REPOSITORY
      ===================================================== */}

      <div className="corpus-card">

        <div className="corpus-card-header">

          <div>

            <span>
              DOCUMENT REPOSITORY
            </span>

            <h2>
              All Documents
            </h2>

          </div>

          <span className="document-count">
            {filteredDocuments.length} results
          </span>

        </div>


        {/* ===================================================
            SEARCH + FILTER
        =================================================== */}

        <div className="corpus-filter-bar">

          <div className="corpus-search">

            <Search size={16} />

            <input
              type="text"
              placeholder="Search documents, wells or regions..."
              value={searchQuery}
              onChange={(e) =>
                setSearchQuery(e.target.value)
              }
            />

            {searchQuery && (
              <button
                className="clear-search"
                onClick={() =>
                  setSearchQuery("")
                }
              >
                <X size={13} />
              </button>
            )}

          </div>


          <div className="corpus-select">

            <Filter size={14} />

            <select
              value={selectedType}
              onChange={(e) =>
                setSelectedType(e.target.value)
              }
            >

              <option value="All">
                All Types
              </option>

              <option value="WCR">
                WCR
              </option>

              <option value="DDR">
                DDR
              </option>

            </select>

          </div>


          <div className="corpus-select">

            <select
              value={selectedStatus}
              onChange={(e) =>
                setSelectedStatus(e.target.value)
              }
            >

              <option value="All">
                All Status
              </option>

              <option value="Processed">
                Processed
              </option>

              <option value="Review">
                Review
              </option>

              <option value="Processing">
                Processing
              </option>

              <option value="Failed">
                Failed
              </option>

            </select>

          </div>

        </div>


        {/* ===================================================
            DOCUMENT LIST
        =================================================== */}

        <div className="document-list">

          <div className="document-list-header">

            <span>DOCUMENT</span>
            <span>TYPE</span>
            <span>WELL / REGION</span>
            <span>DATE</span>
            <span>STATUS</span>
            <span>ACTION</span>

          </div>

          {loading ? (

            <div className="corpus-empty">
              <FileSearch size={30} />
              <strong>Loading documents...</strong>
            </div>

          ) : loadError ? (

            <div className="corpus-empty">
              <FileSearch size={30} />
              <strong>{loadError}</strong>
              <p>Start the backend, then refresh this page.</p>
            </div>

          ) : filteredDocuments.length === 0 ? (

            <div className="corpus-empty">

              <FileSearch size={30} />

              <strong>
                No documents found
              </strong>

              <p>
                {documents.length === 0
                  ? "No documents uploaded yet."
                  : "Try changing your search or filters."}
              </p>

            </div>

          ) : (

            filteredDocuments.map((doc) => (

              <div
                className="document-row"
                key={doc.id}
              >

                {/* DOCUMENT */}

                <div className="document-name">

                  <div
                    className={`document-icon ${getTypeClass(
                      doc.type
                    )}`}
                  >
                    {getTypeIcon(doc.type)}
                  </div>

                  <div>

                    <strong>
                      {doc.name}
                    </strong>

                    <span>
                      {doc.id} · {doc.pages} pages · {doc.size}
                    </span>

                  </div>

                </div>


                {/* TYPE */}

                <span
                  className={`document-type ${getTypeClass(
                    doc.type
                  )}`}
                >
                  {doc.type}
                </span>


                {/* LOCATION */}

                <div className="document-location">

                  <strong>
                    {doc.well}
                  </strong>

                  <span>
                    {doc.region}
                  </span>

                </div>


                {/* DATE */}

                <span className="document-date">
                  {doc.date}
                </span>


                {/* STATUS */}

                <span
                  className={`document-status ${doc.status
                    .toLowerCase()
                    .replace(" ", "-")}`}
                >

                  {getStatusIcon(doc.status)}

                  {doc.status}

                </span>


                {/* ACTION */}

                <div className="document-actions">

                  <button
                    title="View document"
                    onClick={() =>
                      handleOpenDocument(doc)
                    }
                  >
                    <Eye size={15} />
                  </button>

                  <button
                    title="Download"
                    onClick={() =>
                      handleDownload(doc)
                    }
                  >
                    <Download size={15} />
                  </button>

                  <button
                    title="Delete document"
                    onClick={() =>
                      handleDelete(doc)
                    }
                  >
                    <MoreVertical size={15} />
                  </button>

                </div>

              </div>

            ))

          )}

        </div>

      </div>


      {/* =====================================================
          PROCESSING STATUS
      ===================================================== */}

      <div className="corpus-processing">

        <div className="processing-left">

          <div className="processing-icon">
            <FileCheck2 size={19} />
          </div>

          <div>

            <strong>
              Document Intelligence Engine
            </strong>

            <p>
              OCR, metadata extraction and document
              indexing are operational.
            </p>

          </div>

        </div>

        <div className="processing-status">

          <span></span>

          System Ready

        </div>

      </div>


      {/* =====================================================
          DOCUMENT VIEWER
      ===================================================== */}

      {selectedDocument && (

        <div className="document-viewer-overlay">

          <div className="document-viewer">

            {/* VIEWER HEADER */}

            <div className="viewer-header">

              <div className="viewer-title">

                <div className="viewer-title-icon">
                  {getTypeIcon(
                    selectedDocument.type
                  )}
                </div>

                <div>

                  <span>
                    DOCUMENT VIEWER
                  </span>

                  <h2>
                    {selectedDocument.name}
                  </h2>

                  <small>
                    {selectedDocument.id} ·{" "}
                    {selectedDocument.pages} pages ·{" "}
                    {selectedDocument.size}
                  </small>

                </div>

              </div>


              <div className="viewer-header-actions">

                <button
                  onClick={() =>
                    handleDownload(
                      selectedDocument
                    )
                  }
                >
                  <Download size={15} />
                  Download
                </button>

                <button
                  className="viewer-close"
                  onClick={closeDocument}
                >
                  <X size={19} />
                </button>

              </div>

            </div>


            {/* VIEWER INFO */}

            <div className="viewer-info-bar">

              <div>
                <Layers size={15} />

                <strong>
                  {selectedDocument.pages}
                </strong>

                pages
              </div>

              <div>
                <FileCheck2 size={15} />
                {pageIndex.filter((p) => p.has_extraction).length} of{" "}
                {pageIndex.length || selectedDocument.pages} pages extracted
              </div>

              <div>
                <span className="ocr-status-dot"></span>
                {selectedDocument.status}
              </div>

            </div>


            {/* VIEWER CONTENT */}

            <div className="viewer-content">

              {/* PAGE */}

              <div className="viewer-page-section">

                <div className="viewer-section-header">

                  <div>

                    <span>
                      DOCUMENT PAGE
                    </span>

                    <strong>
                      Page {currentPage} of{" "}
                      {selectedDocument.pages}
                    </strong>

                  </div>

                </div>


                <div className="page-canvas">

                  <div className="real-page-image-wrapper">

                    {imageError ? (
                      <div className="real-page-image-fallback">
                        <FileSearch size={26} />
                        <span>No page image available</span>
                      </div>
                    ) : (
                      <img
                        className="real-page-image"
                        src={`${API_BASE_URL}/api/review/${getRawDocumentId(
                          selectedDocument.id
                        )}/${currentPage}/image`}
                        alt={`Page ${currentPage} of ${selectedDocument.name}`}
                        onError={() => setImageError(true)}
                      />
                    )}

                  </div>

                </div>


                {/* NAVIGATION */}

                <div className="page-navigation">

                  <button
                    disabled={currentPage === 1}
                    onClick={() =>
                      setCurrentPage((prev) =>
                        Math.max(
                          1,
                          prev - 1
                        )
                      )
                    }
                  >

                    <ChevronLeft size={16} />

                    Previous

                  </button>


                  <span>
                    Page{" "}
                    <strong>
                      {currentPage}
                    </strong>{" "}
                    /{" "}
                    {selectedDocument.pages}
                  </span>


                  <button
                    disabled={
                      currentPage ===
                      selectedDocument.pages
                    }
                    onClick={() =>
                      setCurrentPage((prev) =>
                        Math.min(
                          selectedDocument.pages,
                          prev + 1
                        )
                      )
                    }
                  >

                    Next

                    <ChevronRight size={16} />

                  </button>

                </div>

              </div>


              {/* OCR */}

              <div className="ocr-section">

                <div className="ocr-header">

                  <div>

                    <span>
                      EXTRACTED CONTENT
                    </span>

                    <h3>
                      OCR Text
                    </h3>

                  </div>

                  <span
                    className="ocr-confidence"
                    style={
                      pageStatusBadge().tone === "bad"
                        ? { background: "#fef2f2", color: "#dc2626" }
                        : pageStatusBadge().tone === "warn"
                        ? { background: "#fff7ed", color: "#c2410c" }
                        : pageStatusBadge().tone === "pending"
                        ? { background: "var(--hover)", color: "var(--text-muted)" }
                        : undefined
                    }
                  >
                    {pageStatusBadge().label}
                  </span>

                </div>


                <div className="ocr-page-info">

                  <FileText size={15} />

                  Page {currentPage}
                  {pageData && pageData.hazards.length > 0 && (
                    <> · {pageData.hazards.length} hazard(s) flagged</>
                  )}

                </div>


                <div className="ocr-text">

                  {pageLoading ? (

                    <p>Loading...</p>

                  ) : pageData && pageData.rendered_html ? (

                    // Trusted, server-built HTML (routers/review.py escapes
                    // every cell on the way in), seeded imperatively by the
                    // effect above - editable directly (type in a table cell
                    // or the surrounding text), same as the standalone
                    // /review tool. Not a React-controlled value: contentEditable
                    // fights a controlled value on every keystroke.
                    <div
                      ref={ocrEditableRef}
                      className="ocr-editable-content"
                      contentEditable
                      suppressContentEditableWarning
                    />

                  ) : pageData && pageData.parse_ok === 0 ? (

                    <p style={{ color: "#dc2626" }}>
                      This page failed to extract.
                      {pageData.failure_reason
                        ? ` ${pageData.failure_reason}`
                        : " No further detail was recorded for this attempt."}
                    </p>

                  ) : selectedDocument.status === "Processing" ? (

                    <p>
                      Processing - this page hasn't been extracted yet.
                      Check back shortly.
                    </p>

                  ) : (

                    <p>
                      No extraction available for this page.
                    </p>

                  )}

                </div>


                {pageData && pageData.rendered_html && (

                  <div className="ocr-edit-actions">

                    <button
                      className="ocr-save-btn"
                      onClick={handleSaveCorrection}
                      disabled={savingCorrection}
                    >
                      <Save size={13} />
                      {savingCorrection ? "Saving..." : "Save Correction"}
                    </button>

                    {saveStatus && (
                      <span
                        className="ocr-save-status"
                        style={{ color: saveStatus.ok ? "#16a34a" : "#dc2626" }}
                      >
                        {saveStatus.message}
                      </span>
                    )}

                  </div>

                )}


                <div className="ocr-footer">

                  <CheckCircle2 size={14} />

                  {pageData && pageData.has_extraction
                    ? "OCR extraction completed"
                    : "Awaiting extraction"}

                </div>

              </div>

            </div>


            {/* ALL PAGES */}

            <div className="page-strip">

              <div className="page-strip-title">

                <span>
                  ALL PAGES
                </span>

                <small>
                  {selectedDocument.pages} pages
                </small>

              </div>


              <div className="page-thumbnails">

                {Array.from(
                  {
                    length:
                      selectedDocument.pages,
                  },
                  (_, index) => {

                    const page = index + 1;
                    const indexEntry = pageIndex.find(
                      (p) => p.page_num === page
                    );
                    const hasExtraction = indexEntry
                      ? indexEntry.has_extraction
                      : null;
                    const pageFailed = indexEntry
                      ? indexEntry.parse_ok === 0
                      : false;

                    return (

                      <button
                        key={page}
                        className={`page-thumbnail ${
                          currentPage === page
                            ? "active"
                            : ""
                        }`}
                        onClick={() =>
                          setCurrentPage(page)
                        }
                        title={
                          pageFailed
                            ? "Extraction failed on this page"
                            : hasExtraction === false
                            ? "No extraction yet"
                            : undefined
                        }
                      >

                        <div
                          className="thumbnail-paper"
                          style={{
                            background: pageFailed
                              ? "#fef2f2"
                              : hasExtraction === false
                              ? "var(--hover)"
                              : "#ffffff",
                          }}
                        >

                          <div></div>
                          <div></div>
                          <div></div>
                          <div></div>

                        </div>

                        <span>
                          {page}
                        </span>

                      </button>

                    );
                  }
                )}

              </div>

            </div>

          </div>

        </div>

      )}

    </div>
  );
}

export default Corpus;
