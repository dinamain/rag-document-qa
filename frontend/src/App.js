import { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);

  // Handle PDF upload
  const handleUpload = async () => {
    if (!file) return alert("Please select a PDF first");
    
    const formData = new FormData();
    formData.append("file", file);
    
    setUploadStatus("Uploading...");
    const response = await fetch("http://localhost:8000/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    setUploadStatus(data.message);
  };

  // Handle question
  const handleAsk = async () => {
    if (!question) return alert("Please type a question");
    
    setLoading(true);
    setAnswer("");
    const response = await fetch("http://localhost:8000/ask", {
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
    <div style={{ maxWidth: "700px", margin: "50px auto", fontFamily: "Arial" }}>
      <h1>RAG Document Q&A</h1>

      {/* Upload Section */}
      <div style={{ marginBottom: "30px" }}>
        <h2>Upload PDF</h2>
        <input type="file" accept=".pdf" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload} style={{ marginLeft: "10px" }}>Upload</button>
        {uploadStatus && <p style={{ color: "green" }}>{uploadStatus}</p>}
      </div>

      {/* Question Section */}
      <div>
        <h2>Ask a Question</h2>
        <input
          type="text"
          placeholder="Ask something about the document..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{ width: "80%", padding: "8px" }}
        />
        <button onClick={handleAsk} style={{ marginLeft: "10px" }}>Ask</button>
        {loading && <p>Thinking...</p>}
        {answer && (
          <div style={{ marginTop: "20px", padding: "15px", background: "#f0f0f0" }}>
            <h3>Answer:</h3>
            <p>{answer}</p>
          </div>
        )}
        {sources.length > 0 && (
  <div style={{ marginTop: "15px" }}>
    <h4>Sources:</h4>
    {sources.map((source, index) => (
      <div key={index} style={{ 
        padding: "8px", 
        marginBottom: "8px", 
        background: "#e8e8e8",
        fontSize: "0.85em"
      }}>
        <strong>Page {source.page + 1}</strong>: {source.text}...
      </div>
    ))}
  </div>
)}
      </div>
    </div>
  );
}

export default App;