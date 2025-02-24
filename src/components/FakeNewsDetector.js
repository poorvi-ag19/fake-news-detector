import { useState, useEffect } from 'react';
import axios from 'axios';
import '../styles/FakeNewsDetector.css';

export default function FakeNewsDetector() {
  const [newsText, setNewsText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fakeNews, setFakeNews] = useState([]);
  const [realNews, setRealNews] = useState([]);

  // Fetch Fake and Real news samples from JSON files in the public/ folder
  useEffect(() => {
    fetch('/fake.json')
      .then((response) => response.json())
      .then((data) => setFakeNews(data))
      .catch((error) => console.error('Error loading fake news:', error));

    fetch('/true.json')
      .then((response) => response.json())
      .then((data) => setRealNews(data))
      .catch((error) => console.error('Error loading real news:', error));
  }, []);

  // Function to analyze user input
  const handleSubmit = async () => {
    if (!newsText) return;
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5000/predict', { text: newsText });
      setResult(response.data);
    } catch (err) {
      setError('Error analyzing the news. Please try again.');
    }
    setLoading(false);
  };

  return (
    <div className="detector-container">
      <h2>Fake News Detector</h2>

      {/* Input Box for User to Enter News */}
      <textarea
        className="input-box"
        rows="4"
        placeholder="Enter news text here..."
        value={newsText}
        onChange={(e) => setNewsText(e.target.value)}
      />
      <button className="check-button" onClick={handleSubmit} disabled={loading}>
        {loading ? 'Analyzing...' : 'Check News'}
      </button>

      {/* Display Prediction Result */}
      {result && (
        <p className={`result ${result.label === 'FAKE' ? 'fake' : 'real'}`}>
          {result.label === 'FAKE' ? '🚨 Fake News Detected!' : '✅ This news seems real!'}
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {/* Display Sample Fake and Real News */}
      <div className="news-samples">
        <div className="news-column">
          <h3>Fake News Samples</h3>
          <ul>
            {fakeNews.slice(0, 5).map((news, index) => (
              <li key={index}>{news.title}</li>
            ))}
          </ul>
        </div>
        <div className="news-column">
          <h3>Real News Samples</h3>
          <ul>
            {realNews.slice(0, 5).map((news, index) => (
              <li key={index}>{news.title}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
