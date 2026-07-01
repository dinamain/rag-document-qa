import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return alert("Please select a PDF first");
    const formData = new FormData();
    formData.append("file", file);
    setUploadStatus("Uploading...");
    const response = await fetch("https://rag-document-qa-yrtf.onrender.com/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    setUploadStatus(data.message);
  };

  const handleAsk = async () => {
    if (!question) return alert("Please type a question");
    setLoading(true);
    setAnswer("");
    setSources([]);
    const response = await fetch("https://rag-document-qa-yrtf.onrender.com/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    setAnswer(data.answer);
    setSources(data.sources || []);
    setLoading(false);
  };

  return (
    <div style={{
      maxWidth: "750px",
      margin: "40px auto",
      fontFamily: "'Segoe UI', Arial, sans-serif",
      padding: "0 20px",
      color: "#1a1a2e"
    }}>
      {/* Header */}
      <div style={{
        textAlign: "center",
        marginBottom: "32px"
      }}>
        <h1 style={{
          fontSize: "28px",
          fontWeight: "700",
          color: "#1a1a2e",
          margin: "0 0 8px 0"
        }}>📄 RAG Document Q&A</h1>
        <p style={{ color: "#666", margin: 0, fontSize: "14px" }}>
          Upload a PDF and ask questions about it
        </p>
      </div>

      {/* Upload Section */}
      <div style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: "12px",
        padding: "24px",
        marginBottom: "20px",
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)"
      }}>
        <h2 style={{ fontSize: "16px", fontWeight: "600", margin: "0 0 16px 0" }}>
          Upload PDF
        </h2>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <input
            type="file"
            accept=".pdf"
            onChange={(e) => setFile(e.target.files[0])}
            style={{ fontSize: "14px", flex: 1 }}
          />
          <button
            onClick={handleUpload}
            style={{
              background: "#4f46e5",
              color: "#fff",
              border: "none",
              borderRadius: "8px",
              padding: "8px 20px",
              fontSize: "14px",
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            Upload
          </button>
        </div>
        {uploadStatus && (
          <p style={{
            margin: "12px 0 0 0",
            fontSize: "14px",
            color: uploadStatus.includes("successfully") ? "#16a34a" : "#666"
          }}>
            {uploadStatus.includes("successfully") ? "✅ " : ""}{uploadStatus}
          </p>
        )}
      </div>

      {/* Question Section */}
      <div style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: "12px",
        padding: "24px",
        marginBottom: "20px",
        boxShadow: "0 1px 3px rgba(0,0,0,0.08)"
      }}>
        <h2 style={{ fontSize: "16px", fontWeight: "600", margin: "0 0 16px 0" }}>
          Ask a Question
        </h2>
        <div style={{ display: "flex", gap: "12px" }}>
          <input
            type="text"
            placeholder="Ask something about the document..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAsk()}
            style={{
              flex: 1,
              padding: "10px 14px",
              fontSize: "14px",
              border: "1px solid #e2e8f0",
              borderRadius: "8px",
              outline: "none"
            }}
          />
          <button
            onClick={handleAsk}
            style={{
              background: "#4f46e5",
              color: "#fff",
              border: "none",
              borderRadius: "8px",
              padding: "10px 24px",
              fontSize: "14px",
              fontWeight: "600",
              cursor: "pointer"
            }}
          >
            Ask
          </button>
        </div>
        {loading && (
          <p style={{ margin: "12px 0 0 0", color: "#666", fontSize: "14px" }}>
            ⏳ Thinking...
          </p>
        )}
      </div>

      {/* Answer Section */}
      {answer && (
        <div style={{
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: "12px",
          padding: "24px",
          marginBottom: "20px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.08)"
        }}>
          <h2 style={{ fontSize: "16px", fontWeight: "600", margin: "0 0 12px 0" }}>
            Answer
          </h2>
          <div style={{
            borderTop: "1px solid #e2e8f0",
            paddingTop: "16px",
            fontSize: "15px",
            lineHeight: "1.7",
            color: "#1a1a2e"
          }}>
            {answer}
          </div>
        </div>
      )}

      {/* Sources Section */}
      {sources.length > 0 && (
        <div style={{
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: "12px",
          padding: "24px",
          boxShadow: "0 1px 3px rgba(0,0,0,0.08)"
        }}>
          <h2 style={{ fontSize: "16px", fontWeight: "600", margin: "0 0 16px 0" }}>
            Retrieved Context
          </h2>
          {sources.map((source, index) => (
            <div key={index} style={{
              background: "#f8fafc",
              border: "1px solid #e2e8f0",
              borderRadius: "8px",
              padding: "12px 16px",
              marginBottom: "10px"
            }}>
              <div style={{
                fontSize: "13px",
                fontWeight: "600",
                color: "#4f46e5",
                marginBottom: "6px"
              }}>
                📄 {source.filename} (Page {source.page + 1})
              </div>
              <div style={{
                fontSize: "13px",
                color: "#555",
                lineHeight: "1.5"
              }}>
                {source.text}...
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;