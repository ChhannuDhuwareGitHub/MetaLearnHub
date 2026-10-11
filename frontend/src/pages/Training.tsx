import React, { useState, useEffect } from 'react';
import api from '../api';
import { Activity, Play, Square, Settings, ChevronRight } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function Training() {
  const [breeds, setBreeds] = useState<string[]>([]);
  const [status, setStatus] = useState<any>(null);
  
  const [config, setConfig] = useState({
    held_out_class: 'Red Dane cattle',
    n_way: 3,
    k_shot: 5,
    query_per_class: 5,
    episodes_per_epoch: 100,
    num_epochs: 30,
    learning_rate: 0.001
  });

  useEffect(() => {
    api.get('/dataset/classes').then(res => setBreeds(res.data.classes)).catch(console.error);
    
    const interval = setInterval(() => {
      api.get('/training/status').then(res => setStatus(res.data)).catch(console.error);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleStart = () => {
    api.post('/training/start', config).catch(console.error);
  };

  const handleCancel = () => {
    api.post('/training/cancel').catch(console.error);
  };

  const isRunning = status?.status === 'running';

  // Prepare chart data
  const chartData = [];
  if (status?.epoch_losses && status?.val_losses) {
    for (let i = 0; i < status.epoch_losses.length; i++) {
      chartData.push({
        epoch: i + 1,
        train_loss: status.epoch_losses[i],
        val_loss: status.val_losses[i],
        train_acc: status.epoch_accuracies[i],
        val_acc: status.val_accuracies[i],
      });
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold mb-2">Meta-Training</h1>
          <p className="text-secondary">Episodic training of Prototypical Networks.</p>
        </div>
        <div className="flex gap-4">
          {isRunning ? (
            <button className="btn btn-danger" onClick={handleCancel}>
              <Square size={16} /> Stop Training
            </button>
          ) : (
            <button className="btn btn-primary" onClick={handleStart}>
              <Play size={16} /> Start Training
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-8">
        <div className="col-span-1 card">
          <div className="card-header border-b border-[var(--border-light)] pb-4">
            <h2 className="card-title flex items-center gap-2">
              <Settings size={20} className="text-primary" />
              Experiment Config
            </h2>
          </div>
          
          <div className="pt-4 flex flex-col gap-4">
            <div className="form-group mb-0">
              <label className="form-label">Held-out Class (Leave-One-Out)</label>
              <select 
                className="form-control"
                value={config.held_out_class}
                onChange={e => setConfig({...config, held_out_class: e.target.value})}
                disabled={isRunning}
              >
                {breeds.map(b => <option key={b} value={b}>{b}</option>)}
              </select>
              <div className="text-xs text-warning mt-1">This class will NOT be seen during training.</div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="form-group mb-0">
                <label className="form-label">N-Way</label>
                <input type="number" className="form-control" value={config.n_way} onChange={e => setConfig({...config, n_way: parseInt(e.target.value)})} disabled={isRunning} />
              </div>
              <div className="form-group mb-0">
                <label className="form-label">K-Shot</label>
                <input type="number" className="form-control" value={config.k_shot} onChange={e => setConfig({...config, k_shot: parseInt(e.target.value)})} disabled={isRunning} />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="form-group mb-0">
                <label className="form-label">Epochs</label>
                <input type="number" className="form-control" value={config.num_epochs} onChange={e => setConfig({...config, num_epochs: parseInt(e.target.value)})} disabled={isRunning} />
              </div>
              <div className="form-group mb-0">
                <label className="form-label">Episodes/Epoch</label>
                <input type="number" className="form-control" value={config.episodes_per_epoch} onChange={e => setConfig({...config, episodes_per_epoch: parseInt(e.target.value)})} disabled={isRunning} />
              </div>
            </div>
          </div>
        </div>

        <div className="col-span-2 card">
          <div className="card-header border-b border-[var(--border-light)] pb-4">
            <h2 className="card-title flex items-center gap-2">
              <Activity size={20} className="text-primary" />
              Training Progress
            </h2>
            {status?.status && (
              <span className={`badge ${isRunning ? 'badge-primary animate-pulse' : 'badge-secondary'}`}>
                {status.status.toUpperCase()}
              </span>
            )}
          </div>

          {status?.status === 'running' || status?.status === 'completed' ? (
            <div className="pt-4">
              <div className="grid grid-cols-4 gap-4 mb-6 text-center">
                <div className="bg-[var(--bg-main)] p-3 rounded-md border border-[var(--border-light)]">
                  <div className="text-xs text-secondary mb-1">Epoch</div>
                  <div className="text-xl font-bold">{status.current_epoch} / {status.total_epochs}</div>
                </div>
                <div className="bg-[var(--bg-main)] p-3 rounded-md border border-[var(--border-light)]">
                  <div className="text-xs text-secondary mb-1">Episode (Current)</div>
                  <div className="text-xl font-bold">{status.current_episode} / {status.episodes_per_epoch}</div>
                </div>
                <div className="bg-[var(--bg-main)] p-3 rounded-md border border-[var(--border-light)]">
                  <div className="text-xs text-secondary mb-1">Val Accuracy</div>
                  <div className="text-xl font-bold text-success">{(status.val_accuracy * 100).toFixed(1)}%</div>
                </div>
                <div className="bg-[var(--bg-main)] p-3 rounded-md border border-[var(--border-light)]">
                  <div className="text-xs text-secondary mb-1">Best Accuracy</div>
                  <div className="text-xl font-bold text-primary">{(status.best_val_accuracy * 100).toFixed(1)}%</div>
                </div>
              </div>

              {chartData.length > 0 && (
                <div className="h-64 mt-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={12} />
                      <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                      <Tooltip contentStyle={{ backgroundColor: '#1a1f2e', borderColor: '#2d354a' }} />
                      <Legend />
                      <Line type="monotone" dataKey="train_acc" stroke="#3b82f6" strokeWidth={2} name="Train Acc" dot={false} />
                      <Line type="monotone" dataKey="val_acc" stroke="#10b981" strokeWidth={2} name="Val Acc" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          ) : (
            <div className="pt-16 flex flex-col items-center justify-center text-muted">
              <Activity size={48} className="mb-4 opacity-50" />
              <p>Configure and start training to see progress.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
