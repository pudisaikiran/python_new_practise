import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [isLoginView, setIsLoginView] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [user, setUser] = useState(null);

  // Check if we are already logged in via localStorage
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    const storedEmail = localStorage.getItem('email');
    if (token && storedEmail) {
      setUser({ email: storedEmail });
    }
  }, []);

  const handleRegister = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');
    
    try {
      const response = await fetch('http://localhost:8000/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, full_name: fullName })
      });
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail || 'Registration failed');
      
      setMessage('Registration successful! Please log in.');
      setIsLoginView(true);
      setPassword(''); // Clear password field
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogin = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');
    
    try {
      const response = await fetch('http://localhost:8000/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.detail || 'Login failed');
      
      // Store tokens securely in local storage
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);
      localStorage.setItem('email', data.email);
      
      setUser({ email: data.email, fullName: data.full_name });
      setMessage('Logged in successfully!');
      setPassword('');
    } catch (err) {
      setError(err.message);
    }
  };

  const handleLogout = async () => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (refreshToken) {
      try {
        // Ping backend to kill the refresh token from DB
        await fetch('http://localhost:8000/auth/logout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken })
        });
      } catch (err) {
        console.error("Logout failed", err);
      }
    }
    
    // Clear local storage regardless of backend success
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('email');
    setUser(null);
    setMessage('Logged out successfully.');
  };

  if (user) {
    return (
      <div className="container dashboard">
        <div className="card">
          <h1>Welcome, {user.email}!</h1>
          {message && <div className="success">{message}</div>}
          <p>You are successfully logged in and your tokens are secured.</p>
          <div className="tokens-display">
            <strong>Access Token:</strong> <br/>
            <span className="token-text">{localStorage.getItem('access_token')}</span>
            <br/><br/>
            <strong>Refresh Token:</strong> <br/>
            <span className="token-text">{localStorage.getItem('refresh_token')}</span>
          </div>
          <button onClick={handleLogout} className="btn-logout">Logout</button>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="card">
        <h1>{isLoginView ? 'Welcome Back' : 'Create an Account'}</h1>
        
        {error && <div className="error">{error}</div>}
        {message && <div className="success">{message}</div>}

        <form onSubmit={isLoginView ? handleLogin : handleRegister}>
          {!isLoginView && (
            <div className="input-group">
              <label>Full Name</label>
              <input 
                type="text" 
                value={fullName} 
                onChange={(e) => setFullName(e.target.value)} 
                required 
              />
            </div>
          )}
          
          <div className="input-group">
            <label>Email Address</label>
            <input 
              type="email" 
              value={email} 
              onChange={(e) => setEmail(e.target.value)} 
              required 
            />
          </div>
          
          <div className="input-group">
            <label>Password</label>
            <input 
              type="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              required 
            />
          </div>

          <button type="submit" className="btn-primary">
            {isLoginView ? 'Login' : 'Register'}
          </button>
        </form>

        <p className="toggle-view">
          {isLoginView ? "Don't have an account? " : "Already have an account? "}
          <span onClick={() => setIsLoginView(!isLoginView)}>
            {isLoginView ? 'Sign up here' : 'Login here'}
          </span>
        </p>
      </div>
    </div>
  );
}

export default App;
