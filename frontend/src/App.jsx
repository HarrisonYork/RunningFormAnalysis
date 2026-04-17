import { useState, useRef } from 'react';
import './App.css';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  
  const fileInputRef = useRef(null);

  const MAX_FILE_SIZE_MB = 50;
  const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

  const handleFileChange = (event) => {
    setErrorMsg(''); // Clear previous errors
    const file = event.target.files[0];
    
    if (file) {
      if (file.size > MAX_FILE_SIZE_BYTES) {
        setErrorMsg(`File size exceeds the ${MAX_FILE_SIZE_MB}MB limit. Please upload a smaller video.`);
        setVideoFile(null);
        setPreviewUrl(null);
        event.target.value = null;
        return;
      }
      
      setVideoFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleAnalyze = async () => {
    if (!videoFile) return;
    
    setIsUploading(true);
    console.log("Simulating upload and analysis for:", videoFile.name);
    
    // TODO: Connect this to your Python backend using fetch() or axios
    setTimeout(() => {
      alert(`Successfully analyzed ${videoFile.name}! (Placeholder)`);
      setIsUploading(false);
    }, 2000);
  };

  return (
    <div id="app-container">
      <section className="upload-section">
        <h1>Running Form Analysis</h1>
        <p>Upload a video to analyze your running form and receive tailored feedback.</p>

        
        <div className="upload-box" onClick={triggerFileInput}>
          <input 
            type="file" 
            accept="video/mp4,video/quicktime,video/x-msvideo" 
            onChange={handleFileChange} 
            ref={fileInputRef}
          />
          <div className="upload-icon">📁</div>
          <p>{videoFile ? videoFile.name : "Click to browse or drag and drop your video here"}</p>
          <span>MP4, MOV, or AVI (Max {MAX_FILE_SIZE_MB}MB)</span>
        </div>

        {/* Error Message Display */}
        {errorMsg && <div className="error-msg">{errorMsg}</div>}

        {/* Local Video Preview */}
        {previewUrl && (
          <div className="video-preview">
            <video src={previewUrl} controls>
              Your browser does not support the video tag.
            </video>
          </div>
        )}

        <button 
          onClick={handleAnalyze} 
          disabled={!videoFile || isUploading}
        >
          {isUploading ? 'Analyzing...' : 'Analyze Form'}
        </button>
      </section>

      <section id="project-details">
        <h2>About</h2>
        <p>
          This tool uses machine learning to estimate a runner's pose and analyze their form.
          Upload a clear, side-profile video of your run to get started.
        </p>
        <ul>
          {/* <li></li>
          <li></li>
          <li></li> */}
        </ul>
      </section>
    </div>
  );
}

export default App;