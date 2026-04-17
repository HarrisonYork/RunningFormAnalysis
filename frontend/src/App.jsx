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
    setErrorMsg('');
    const file = event.target.files[0];
    
    if (file) {
      if (file.size > MAX_FILE_SIZE_BYTES) {
        setErrorMsg(`File size exceeds the ${MAX_FILE_SIZE_MB}MB limit. Please upload a smaller video.`);
        handleClear(); 
        return;
      }
      
      setVideoFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

const triggerFileInput = () => {
    // Only trigger if we aren't currently analyzing
    if (fileInputRef.current && !isUploading) {
      fileInputRef.current.click();
    }
  };

  const handleClear = (e) => {
    if (e) e.stopPropagation(); // Prevents triggering the file input click if event bubbles

    // Clean up the object URL to avoid memory leaks
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setVideoFile(null);
    setPreviewUrl(null);
    setErrorMsg('');
    
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Reset the actual input element
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

        {/* Large Placeholder Upload Box */}
        {!videoFile && (
          <div className="upload-box" onClick={triggerFileInput}>
            <input 
              type="file" 
              accept="video/mp4,video/quicktime,video/x-msvideo" 
              onChange={handleFileChange} 
              ref={fileInputRef}
            />
            <div className="upload-icon">📁</div>
            <p>Click to browse or drag and drop your video here</p>
            <span>MP4, MOV, or AVI (Max {MAX_FILE_SIZE_MB}MB)</span>
          </div>
        )}

        {/* Error Message Display */}
        {errorMsg && <div className="error-msg">{errorMsg}</div>}

        {/* Local Video Preview & Controls */}
        {videoFile && (
          <div className="active-video-container">
            <div className="video-preview">
              <video src={previewUrl} controls>
                Your browser does not support the video tag.
              </video>
              <p className="file-name"><strong>Selected:</strong> {videoFile.name}</p>
            </div>

            <div className="action-buttons">
              <button 
                onClick={handleClear} 
                disabled={isUploading}
                className="clear-btn"
              >
                Clear Video
              </button>
              <button 
                onClick={handleAnalyze} 
                disabled={!videoFile || isUploading}
                className="analyze-btn"
              >
                {isUploading ? 'Analyzing...' : 'Analyze Form'}
              </button>
            </div>
          </div>
        )}
      </section>

      <section id="project-details">
        <h2>About</h2>
        <p>
          This tool uses machine learning to estimate a runner's pose and analyze their form.
        </p>
        <p>
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