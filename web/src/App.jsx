import { Navigate, Route, Routes, Link } from 'react-router-dom';
import { useAuth } from './auth.jsx';
import { Spinner } from './components/Bits.jsx';
import { Shell } from './components/Shell.jsx';
import Account from './pages/Account.jsx';
import About from './pages/About.jsx';
import AuthPage from './pages/AuthPage.jsx';
import History from './pages/History.jsx';
import Home from './pages/Home.jsx';
import NewScan from './pages/NewScan.jsx';
import ScanPage from './pages/ScanPage.jsx';

function RequireAuth({ children }) {
  const { user } = useAuth();
  if (user === undefined) return <div style={{ padding: 40 }}><Spinner /> Loading...</div>;
  if (user === null) return <Navigate to="/login" replace />;
  return children;
}

const NotFound = () => (
  <>
    <h1>Page not found</h1>
    <p className="muted">That page does not exist. <Link to="/">Go home</Link>.</p>
  </>
);

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<AuthPage />} />
      <Route element={<RequireAuth><Shell /></RequireAuth>}>
        <Route index element={<Home />} />
        <Route path="scan/new" element={<NewScan />} />
        <Route path="scan/:id" element={<ScanPage />} />
        <Route path="history" element={<History />} />
        <Route path="about" element={<About />} />
        <Route path="account" element={<Account />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
