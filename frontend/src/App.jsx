import { useState, useRef } from 'react';
import './App.css';
import FormFeedback from './FormFeedback';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [videoUrl, setVideoUrl] = useState(null);
  const [confidences, setConfidences] = useState(null);
  
  const fileInputRef = useRef(null);

  const MAX_FILE_SIZE_MB = 50;
  const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;

const handleFileChange = (event) => {
    setErrorMsg('');
    const file = event.target.files[0];
    
    if (file) {
      if (file.size > MAX_FILE_SIZE_BYTES) {
        setErrorMsg(`File size exceeds the ${MAX_FILE_SIZE_MB}MB limit. Please upload a smaller video.`); 
        return;
      }
      
      setVideoFile(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

const triggerFileInput = () => {
    if (fileInputRef.current && !isUploading) {
      fileInputRef.current.click();
    }
  };

  const handleClear = (e) => {
    if (e) e.stopPropagation();

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }

    setVideoFile(null);
    setPreviewUrl(null);
    setVideoUrl(null);
    setConfidences(null);
    setErrorMsg('');
    
    if (fileInputRef.current) {
      fileInputRef.current.value = ''; // Reset the actual input element
    }
  };

  const handleAnalyze = async () => {
    if (!videoFile) return;
    
    setIsUploading(true);
    setErrorMsg('');
    
    const formData = new FormData();
    formData.append("video", videoFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/api/pose_estimate", {
          method: 'POST',
          body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
          throw new Error(data.error || "Failed to analyze video.");
      }

      setConfidences(data.confidences);
      setVideoUrl("http://127.0.0.1:5000/" + data.video_url);
    } catch (err) {
      console.error("Upload error:", err);
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div id="app-container">
      <section className="upload-section">
        <h1>Running Form Analysis</h1>
        <p>Upload a video to analyze your running form and receive tailored feedback.</p>
        <p>Videos should be a side profile.</p>

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
            <p>Click to upload a video</p>
            <span>MP4, MOV, or AVI (Max {MAX_FILE_SIZE_MB}MB)</span>
          </div>
        )}

        {/* Error Message Display */}
        {errorMsg && <div className="error-msg">{errorMsg}</div>}

        {videoFile && (
          <div className="active-video-container">
            
            {!videoUrl && previewUrl && (
              <div className="video-preview">
                <video 
                  src={previewUrl} 
                  controls={!isUploading} // Disable controls while analyzing
                  style={{ 
                    opacity: isUploading ? 0.2 : 1, // Dim the video while loading
                    transition: 'opacity 0.5s ease'
                  }} 
                >
                  Your browser does not support the video tag.
                </video>

                {isUploading && (
                  <div 
                    className="loading-overlay"
                    style={{
                      position: 'absolute',
                      top: '-100px', left: 0, right: 0, bottom: 0,
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      pointerEvents: 'none' // Prevents overlay from blocking clicks
                    }}
                  >
                    <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>⏳</div>
                    <h3 style={{ color: 'white', margin: 0, textShadow: '0 2px 4px rgba(0,0,0,0.8)' }}>
                      Analyzing your running form...
                    </h3>
                  </div>
                )}
                
                <p className="file-name"><strong>Selected:</strong> {videoFile.name}</p>
                
              </div>
            )}

            {!isUploading && videoUrl && (
              <div className="results-container">
                <video src={videoUrl} controls>
                  Your browser does not support the video tag.
                </video>
                <p className="file-name"><strong>Analyzed:</strong> {videoFile.name}</p>
              </div>
            )}

            <div className="action-buttons">
              <button 
                onClick={handleClear} 
                disabled={isUploading}
                className="clear-btn"
              >
                Clear Video
              </button>
              
              {!videoUrl && (
                <button 
                  onClick={handleAnalyze} 
                  disabled={!videoFile || isUploading}
                  className="analyze-btn"
                >
                  {isUploading ? 'Analyzing...' : 'Analyze Form'}
                </button>
              )}
            </div>
          </div>
        )}
      </section>

      <section id="form-analysis">
        {confidences && (
          <div className="metrics-wrapper" style={{ padding: '1rem', borderRadius: '8px' }}>
              <FormFeedback confidences={confidences} />
          </div>
        )}
      </section>

      <section id="project-details">
        <h2>About</h2>
        <p>
          This tool uses machine learning to estimate a runner's pose and analyze their form.
        </p>
        {/* <p>
          Upload a clear, side-profile video of your run to get started.
        </p> */}
        <p>
          <a href="https://github.com/HarrisonYork/RunningForm">Project Code</a>
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