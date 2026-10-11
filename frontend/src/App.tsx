import React from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Database, Activity, Beaker, Play, Box } from 'lucide-react';
import clsx from 'clsx';

// Pages
import Dashboard from './pages/Dashboard';
import Dataset from './pages/Dataset';
import Training from './pages/Training';
import Evaluation from './pages/Evaluation';
import Predict from './pages/Predict';

function Sidebar() {
  const location = useLocation();
  const path = location.pathname;

  const navItems = [
    { path: '/', icon: <LayoutDashboard size={20} />, label: 'Dashboard' },
    { path: '/dataset', icon: <Database size={20} />, label: 'Dataset' },
    { path: '/training', icon: <Activity size={20} />, label: 'Training' },
    { path: '/evaluation', icon: <Beaker size={20} />, label: 'Evaluation' },
    { path: '/predict', icon: <Play size={20} />, label: 'Few-Shot Predict' },
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Box size={28} className="text-primary" />
        <span className="sidebar-logo">MetaLearnHub</span>
      </div>
      <nav className="nav-links">
        {navItems.map((item) => (
          <Link
            key={item.path}
            to={item.path}
            className={clsx('nav-item', { 'active': path === item.path })}
          >
            {item.icon}
            {item.label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}

function App() {
  return (
    <Router>
      <div className="app-container">
        <Sidebar />
        <main className="main-content">
          <header className="topbar">
            <h1 className="text-lg font-semibold">MetaLearnHub Prototype</h1>
            <div className="flex items-center gap-4 text-sm text-secondary">
              <span>Status: <span className="text-success font-semibold">Online</span></span>
            </div>
          </header>
          <div className="page-container">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dataset" element={<Dataset />} />
              <Route path="/training" element={<Training />} />
              <Route path="/evaluation" element={<Evaluation />} />
              <Route path="/predict" element={<Predict />} />
            </Routes>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App;
