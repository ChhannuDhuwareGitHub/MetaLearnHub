import React, { useEffect, useState } from 'react';
import { Activity, Database, Server, Beaker, CheckCircle } from 'lucide-react';
import api from '../api';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/status')
      .then(res => {
        setStatus(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching status', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="flex justify-center items-center h-64"><div className="animate-pulse text-lg">Loading...</div></div>;
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-2">System Overview</h1>
        <p className="text-secondary">Monitor the overall status of the MetaLearnHub prototype pipeline.</p>
      </div>

      <div className="grid grid-cols-4 gap-6 mb-8">
        <div className="stat-card">
          <div className="flex justify-between items-center text-primary mb-2">
            <Database size={24} />
            <span className="badge badge-success">Loaded</span>
          </div>
          <div className="stat-value">{status?.breeds?.length || 0}</div>
          <div className="stat-label">Cattle Breeds Found</div>
        </div>

        <div className="stat-card border-t-[#8b5cf6]">
          <div className="flex justify-between items-center text-[#8b5cf6] mb-2">
            <Activity size={24} />
            <span className="badge badge-secondary">{status?.training_status || 'Idle'}</span>
          </div>
          <div className="stat-value">{status?.total_experiments || 0}</div>
          <div className="stat-label">Total Experiments</div>
        </div>

        <div className="stat-card border-t-[#f59e0b]">
          <div className="flex justify-between items-center text-[#f59e0b] mb-2">
            <Beaker size={24} />
            <span className="badge badge-secondary">{status?.total_evaluations > 0 ? 'Evaluated' : 'Pending'}</span>
          </div>
          <div className="stat-value">{status?.total_evaluations || 0}</div>
          <div className="stat-label">Evaluations Run</div>
        </div>

        <div className="stat-card border-t-[#10b981]">
          <div className="flex justify-between items-center text-[#10b981] mb-2">
            <Server size={24} />
            {status?.cuda_available ? (
              <span className="badge badge-success">GPU</span>
            ) : (
              <span className="badge badge-secondary">CPU</span>
            )}
          </div>
          <div className="stat-value">{status?.device || 'Unknown'}</div>
          <div className="stat-label">Compute Target</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Pipeline Status</h2>
          </div>
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3">
              <CheckCircle className={status?.dataset_loaded ? "text-success" : "text-muted"} size={20} />
              <div className={status?.dataset_loaded ? "text-primary" : "text-muted"}>Dataset Inspected & Grouped</div>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle className={status?.total_experiments > 0 ? "text-success" : "text-muted"} size={20} />
              <div className={status?.total_experiments > 0 ? "text-primary" : "text-muted"}>Meta-Training Model Checkpoint</div>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle className={status?.total_evaluations > 0 ? "text-success" : "text-muted"} size={20} />
              <div className={status?.total_evaluations > 0 ? "text-primary" : "text-muted"}>Few-Shot Evaluation Results</div>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle className={status?.model_loaded ? "text-success" : "text-muted"} size={20} />
              <div className={status?.model_loaded ? "text-primary" : "text-muted"}>Inference Engine Loaded</div>
            </div>
          </div>
          <div className="mt-6">
            <Link to="/training" className="btn btn-primary w-full text-center">Start New Experiment</Link>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Project Objective</h2>
          </div>
          <p className="text-secondary mb-4 text-sm leading-relaxed">
            Determine whether a meta-learning model (Prototypical Networks) can learn transferable visual knowledge from several cattle-breed classes and then adapt to a previously unseen class using only a few labeled examples (1, 3, 5, or 10 shots).
          </p>
          <div className="p-4 bg-[var(--bg-surface-elevated)] rounded-md border border-[var(--border-light)] text-sm">
            <strong className="text-primary block mb-1">Data Leakage Prevention:</strong>
            Original images and their transformed counterparts (_c) are strictly grouped to ensure they always land in the same partition (Train/Val/Test), preventing catastrophic data leakage.
          </div>
        </div>
      </div>
    </div>
  );
}
